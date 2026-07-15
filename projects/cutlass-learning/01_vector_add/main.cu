#include "cuda_helpers.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// 这里的 16 单位是 byte。float4 不是“四个 float4”，而是一个包含 x/y/z/w
// 四个 float 的 CUDA 向量类型：4*4=16 byte；这条 128-bit 访问路径还要求
// 起始地址按 16 byte 对齐。half2 则是 2 个 half、4 byte，对齐要求也不同。
static_assert(sizeof(float4) == 16, "float4 must contain four 32-bit floats");
static_assert(alignof(float4) == 16, "float4 access requires 16-byte alignment");

// __restrict__ 是调用者对编译器的承诺：A/B/C 指向的逻辑区不重叠，编译器因而
// 可少做 alias 防守并更积极优化；违反承诺会产生不可依赖的结果。count 用
// size_t 是因为元素数非负，并与 vector::size、sizeof、cudaMalloc 的长度类型一致；
// int 也能覆盖本实验的小 N，但大数组的 index*stride 可能溢出 32 bit。
// Vector Add 是一维数组，所以只使用 x。二维图像常用 x/y，三维体数据再用 z；
// blockIdx 表示 block 坐标，blockDim 是每 block 的线程数，threadIdx 是块内坐标。
__global__ void vector_add_scalar_kernel(float const* __restrict__ a,
                                         float const* __restrict__ b,
                                         float* __restrict__ c,
                                         std::size_t count) {
  std::size_t const index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    c[index] = a[index] + b[index];
  }
}

// 每个线程处理一个 float4（4 个连续 float）。前 tail_count 个线程（全局编号
// 0..tail_count-1，数量为 0–3）还会用标量指令处理 N % 4 的尾部，因此向量
// 主体和标量尾部只需要一次 kernel launch。
// Host 端计算 pack_count=N/4、tail_count=N%4，所以
// N=4*pack_count+tail_count，且 tail_count 只可能是 0..3。两个 if 不是互斥的：
// 前几个线程可能既处理一个完整 pack，又顺手处理一个尾元素，它们写的位置不同。
// a[index]/b[index] 读出的是 float4 向量，av.x 等成员才是四个标量 float。
// reinterpret_cast 只改变“如何解释同一地址”的指针类型，不搬运也不转换数值；
// 因此在 cast 成 float4* 之前必须先证明地址满足 float4 对齐。
__global__ void vector_add_float4_kernel(float4 const* __restrict__ a,
                                         float4 const* __restrict__ b,
                                         float4* __restrict__ c,
                                         std::size_t pack_count,
                                         std::size_t tail_count) {
  std::size_t const index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (index < pack_count) {
    float4 const av = a[index];
    float4 const bv = b[index];
    c[index] = make_float4(av.x + bv.x, av.y + bv.y, av.z + bv.z,
                           av.w + bv.w);
  }

  // 尾部不足四个元素，若继续做 float4 load/store 会越过逻辑边界。因此把同一
  // allocation 临时按 float 元素查看，只读写最后 0..3 个有效标量。index 是
  // 一维“全局线程编号”，不是只能取 0..3 的 x 坐标；tail_count=N%4 才只能取
  // 0..3。条件 index<tail_count 只挑出全局编号 0..tail_count-1 的几个线程，
  // 再用 scalar_index 把它们映射到所有完整 float4 pack 之后的尾部位置。
  if (index < tail_count) {
    std::size_t const scalar_index = pack_count * 4 + index;
    float const* scalar_a = reinterpret_cast<float const*>(a);
    float const* scalar_b = reinterpret_cast<float const*>(b);
    float* scalar_c = reinterpret_cast<float*>(c);
    scalar_c[scalar_index] = scalar_a[scalar_index] + scalar_b[scalar_index];
  }
}

// 本文件上方的 static_assert 已验证 alignof(float4)==16 byte。指针本身不能直接
// 做取模；uintptr_t 是标准库提供的、足以无损保存地址数值的无符号整数类型，
// 转成它以后才能用 address % 16 检查对齐。
bool is_aligned_for_float4(void const* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

int grid_size(std::size_t work_items, int threads) {
  return static_cast<int>((work_items + static_cast<std::size_t>(threads) - 1) /
                          static_cast<std::size_t>(threads));
}

struct VectorizedPlan {
  bool use_float4 = false;
  std::size_t pack_count = 0;
  std::size_t tail_count = 0;
  // path 只用于把 dispatcher 的“实际执行路径”写进输出。requested vector 可能
  // 因 N 太小或地址未对齐而回退 scalar；记录它可避免把 fallback 性能误标成
  // float4 性能。char const* 指向静态字符串，不参与 kernel 计算。
  char const* path = "scalar";
};

VectorizedPlan make_vectorized_plan(float const* a, float const* b, float* c,
                                    std::size_t count) {
  bool const aligned = is_aligned_for_float4(a) &&
                       is_aligned_for_float4(b) &&
                       is_aligned_for_float4(c);
  if (!aligned) {
    return {false, 0, 0, "scalar (16-byte alignment fallback)"};
  }
  if (count < 4) {
    // count/4 为 0 时不能发射 <<<0, threads>>>，直接走标量路径。
    return {false, 0, count, "scalar (N < 4 fallback)"};
  }

  std::size_t const packs = count / 4;
  std::size_t const tail = count % 4;
  return {true, packs, tail,
          tail == 0 ? "float4" : "float4 + scalar tail"};
}

struct Verification {
  ErrorStats errors;
  bool guards_unchanged = true;
};

// output_offset 是 allocation 开头到逻辑 C[0] 之间的 float 元素数；logical_count
// 是有效输出 N；allocation_count 是含 offset 和尾部 guard 的总 float 元素数。
// vector iterator 的位移类型是有符号 difference_type（通常就是 ptrdiff_t），所以
// 这里把无符号 size_t 计数显式转换后再做 begin()+offset；本实验的参数上限保证
// 该转换可表示。guard 检查用于发现 kernel 写出逻辑区但仍落在 allocation 内的
// 越界；写到 allocation 外仍应交给 Compute Sanitizer 检测。基础版 offset=0 时
// 没有前 guard，advanced.cu 才固定在逻辑区两侧都放 guard。
Verification copy_and_verify(DeviceBuffer<float> const& device_output,
                             std::size_t output_offset,
                             std::size_t logical_count,
                             std::size_t allocation_count,
                             std::vector<float> const& expected) {
  std::vector<float> storage(allocation_count);
  device_output.copy_to_host(storage.data(), allocation_count);

  std::vector<float> actual(
      storage.begin() + static_cast<std::ptrdiff_t>(output_offset),
      storage.begin() +
          static_cast<std::ptrdiff_t>(output_offset + logical_count));
  ErrorStats const errors =
      compare_vectors(actual, expected, 1.0e-6, 1.0e-6);

  // cudaMemset(..., 0xff) 生成 NaN canary。逻辑区前后的值若不再是 NaN，说明
  // kernel 写出了 [offset, offset+N)；仅靠更大的 allocation 会掩盖这种错误。
  bool guards_unchanged = true;
  for (std::size_t i = 0; i < output_offset; ++i) {
    guards_unchanged = guards_unchanged && std::isnan(storage[i]);
  }
  for (std::size_t i = output_offset + logical_count;
       i < allocation_count; ++i) {
    guards_unchanged = guards_unchanged && std::isnan(storage[i]);
  }
  return {errors, guards_unchanged};
}

struct BenchmarkRow {
  int threads = 0;
  float scalar_ms = 0.0F;
  float vectorized_ms = 0.0F;
  Verification scalar_verification;
  Verification vectorized_verification;
};

double effective_gb_per_second(std::size_t count, float milliseconds) {
  // A、B 各读一次，C 写一次，所以逻辑流量是 3*N*sizeof(float)。3.0 是
  // double；count 显式转成 double 后，整条表达式按 double 计算，不会有问题。
  double const transferred_bytes =
      3.0 * static_cast<double>(count) * sizeof(float);
  return transferred_bytes / (static_cast<double>(milliseconds) * 1.0e6);
}

void print_usage(char const* program) {
  std::cout
      << "Usage: " << program
      << " [N] [iterations] [a_offset] [b_offset] [c_offset] [block_size]\n"
      << "  offsets are measured in float elements; offset=1 forces a 4-byte "
         "but not 16-byte aligned pointer\n"
      << "  omit block_size to sweep 64, 128, 256, and 512 threads\n";
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--help") {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  if (argc > 7) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  // argc 是参数个数，argv 保存参数字符串。count 是向量元素数；iterations 是
  // 每个配置计时的重复次数。未提供参数时使用大于本机 L2 的默认问题。
  std::size_t const count = argc > 1
                                ? static_cast<std::size_t>(
                                      parse_positive_int(argv[1], "element count"))
                                : 16'777'219;
  int const iterations = argc > 2 ? parse_positive_int(argv[2], "iterations")
                                  : 100;
  // offset 的单位是 float 元素。cudaMalloc 返回充分对齐的基址；加 1 会把逻辑
  // 首地址移动 4 byte，使它仍满足 float 对齐、却不再满足 float4 的 16-byte
  // 对齐。分别改变 A/B/C offset，可以验证三者任意一个未对齐都会安全回退；
  // host 数据也复制到同一 offset，所以逻辑上的 A[i]/B[i]/C[i] 并没有错位。
  std::size_t const a_offset =
      argc > 3 ? static_cast<std::size_t>(
                     parse_nonnegative_int(argv[3], "A offset"))
               : 0;
  std::size_t const b_offset =
      argc > 4 ? static_cast<std::size_t>(
                     parse_nonnegative_int(argv[4], "B offset"))
               : 0;
  std::size_t const c_offset =
      argc > 5 ? static_cast<std::size_t>(
                     parse_nonnegative_int(argv[5], "C offset"))
               : 0;
  if (a_offset > 1024 || b_offset > 1024 || c_offset > 1024) {
    std::cerr << "Offsets are a teaching aid and must be <= 1024 floats.\n";
    return EXIT_FAILURE;
  }

  std::vector<int> thread_counts{64, 128, 256, 512};
  if (argc > 6) {
    thread_counts = {parse_positive_int(argv[6], "block size")};
  }

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  for (int threads : thread_counts) {
    if (threads > properties.maxThreadsPerBlock) {
      std::cerr << "block size " << threads << " exceeds device maximum "
                << properties.maxThreadsPerBlock << '\n';
      return EXIT_FAILURE;
    }
  }

  std::vector<float> a(count);
  std::vector<float> b(count);
  std::vector<float> expected(count);

  // 这不是随机数，而是确定性、可复现的周期序列。两个不同的质数周期避免 A/B
  // 完全同形；减去中点让数值有正有负；除以 251/127 把值缩放到约 [-0.5,0.5]，
  // 便于检查且避免不必要的溢出。expected 是 CPU reference。
  for (std::size_t i = 0; i < count; ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 251) - 125) / 251.0F;
    b[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 127.0F;
    expected[i] = a[i] + b[i];
  }

  // constexpr 表示 guard_elements 是编译期常量。4 个 float 正好是 16 byte，也
  // 就是一个 float4 的宽度，便于捕获多写一个 vector pack 的错误；它不是能证明
  // 任意越界都安全的“魔法长度”，所以还要运行 Compute Sanitizer。DeviceBuffer
  // 是 host 侧 RAII 包装器，内部 cudaMalloc 得到的 storage 位于 device global
  // memory。这里总有 4 个 suffix guard；offset>0 的前置区也兼作 prefix guard。
  std::size_t constexpr guard_elements = 4;
  std::size_t const a_allocation = a_offset + count + guard_elements;
  std::size_t const b_allocation = b_offset + count + guard_elements;
  std::size_t const c_allocation = c_offset + count + guard_elements;
  DeviceBuffer<float> device_a(a_allocation);
  DeviceBuffer<float> device_b(b_allocation);
  DeviceBuffer<float> device_c(c_allocation);

  CUDA_CHECK(cudaMemset(device_a.data(), 0xff, a_allocation * sizeof(float)));
  CUDA_CHECK(cudaMemset(device_b.data(), 0xff, b_allocation * sizeof(float)));
  // std::vector 不能直接传给 cudaMemcpy：vector 是带 size/capacity 等信息的
  // C++ 容器对象，a.data() 才是其连续元素区首地址。第三个参数是 device offset。
  device_a.copy_from_host(a.data(), count, a_offset);
  device_b.copy_from_host(b.data(), count, b_offset);

  // 这三个确实都是指向 device global memory 的指针。前面只把 host 的 A/B 数据
  // 复制到了 device_a/device_b；下面的赋值既不搬数据也不做加法，只是让指针跳过
  // guard/offset，指向 kernel 应看到的逻辑起点。device_c 此时尚无有效输出；每条
  // benchmark 路径运行前才把它 poison，kernel 执行后它才成为输出。
  float const* input_a = device_a.data() + a_offset;
  float const* input_b = device_b.data() + b_offset;
  float* output_c = device_c.data() + c_offset;
  VectorizedPlan const plan =
      make_vectorized_plan(input_a, input_b, output_c, count);

  std::vector<BenchmarkRow> rows;
  bool all_passed = true;
  for (int threads : thread_counts) {
    BenchmarkRow row;
    row.threads = threads;

    CUDA_CHECK(cudaMemset(device_c.data(), 0xff,
                          c_allocation * sizeof(float)));
    int const scalar_blocks = grid_size(count, threads);
    // [&] 是按引用捕获：lambda 可直接使用本作用域中的指针、count、grid 等变量，
    // 避免逐个写入捕获列表；它只在这些变量仍存活的 main 作用域内使用。
    // 定义 lambda 本身不会 launch。benchmark_cuda_ms 随后会调用它做 warm-up 和
    // 计时，所以每种 block size 都一定先测一次 scalar baseline；若 float4 条件
    // 不满足，下面 requested-vector benchmark 还会再次调用同一个 scalar kernel。
    auto launch_scalar = [&] {
      vector_add_scalar_kernel<<<scalar_blocks, threads>>>(input_a, input_b,
                                                            output_c, count);
    };
    row.scalar_ms = benchmark_cuda_ms(launch_scalar, 5, iterations);
    row.scalar_verification =
        copy_and_verify(device_c, c_offset, count, c_allocation, expected);

    // 是的，下面要独立测 requested-vector 路径。两条路径复用 device_c，所以先
    // 重新 poison 整个 allocation，才能证明向量 kernel 自己写全了结果、没有借用
    // scalar 的旧输出，并能为这次运行重新检查 guard；memset 不计入 kernel 时间。
    CUDA_CHECK(cudaMemset(device_c.data(), 0xff,
                          c_allocation * sizeof(float)));
    // `[&] { ... }` 是 lambda 表达式：花括号是这个可调用对象的函数体，不是立刻
    // 执行的普通代码块。它被保存为 launch_vectorized，随后由 benchmark_cuda_ms
    // 反复调用；host 端 if 每次选择 scalar fallback 或 float4 kernel。
    auto launch_vectorized = [&] {
      if (!plan.use_float4) {
        vector_add_scalar_kernel<<<scalar_blocks, threads>>>(
            input_a, input_b, output_c, count);
        return;
      }
      int const vector_blocks = grid_size(plan.pack_count, threads);
      vector_add_float4_kernel<<<vector_blocks, threads>>>(
          reinterpret_cast<float4 const*>(input_a),
          reinterpret_cast<float4 const*>(input_b),
          reinterpret_cast<float4*>(output_c), plan.pack_count,
          plan.tail_count);
    };
    row.vectorized_ms = benchmark_cuda_ms(launch_vectorized, 5, iterations);
    row.vectorized_verification =
        copy_and_verify(device_c, c_offset, count, c_allocation, expected);

    bool const row_passed =
        row.scalar_verification.errors.failures == 0 &&
        row.vectorized_verification.errors.failures == 0 &&
        row.scalar_verification.guards_unchanged &&
        row.vectorized_verification.guards_unchanged;
    all_passed = all_passed && row_passed;
    rows.push_back(row);
  }

  // left/right 控制字段左/右对齐；setw(n) 只为“下一次”输出设置最小字段宽度。
  // fixed 选择定点小数格式，此时 setprecision(n) 表示小数点后 n 位；defaultfloat
  // 在表格后恢复默认浮点格式（此时 precision 又表示有效数字数）。这些 manipulator
  // 只影响输出格式，不改变 row 中保存的数值。
  std::cout << "Vector Add boundary and float4 lab\n"
            << "elements=" << count << ", iterations=" << iterations
            << ", offsets(A,B,C)=(" << a_offset << ',' << b_offset << ','
            << c_offset << ")\n"
            << "aligned16(A,B,C)=(" << is_aligned_for_float4(input_a) << ','
            << is_aligned_for_float4(input_b) << ','
            << is_aligned_for_float4(output_c) << ")\n"
            << "vectorized dispatch: " << plan.path << "\n\n"
            << std::left << std::setw(8) << "block" << std::right
            << std::setw(13) << "scalar_ms" << std::setw(15) << "scalar_GB/s"
            << std::setw(14) << "vector_ms" << std::setw(15) << "vector_GB/s"
            << std::setw(11) << "speedup" << '\n';

  for (BenchmarkRow const& row : rows) {
    std::cout << std::left << std::setw(8) << row.threads << std::right
              << std::fixed << std::setprecision(6) << std::setw(13)
              << row.scalar_ms << std::setprecision(2) << std::setw(15)
              << effective_gb_per_second(count, row.scalar_ms)
              << std::setprecision(6) << std::setw(14) << row.vectorized_ms
              << std::setprecision(2) << std::setw(15)
              << effective_gb_per_second(count, row.vectorized_ms)
              << std::setprecision(3) << std::setw(10)
              << row.scalar_ms / row.vectorized_ms << "x\n";
  }

  // min_element 返回 iterator，所以前面的 unary `*` 把 iterator 解引用为真正的
  // BenchmarkRow，再以 const& 绑定到 vector 内元素。`[]` 是不捕获外部变量的
  // comparator lambda；两个 const& 参数避免复制且不允许修改，返回 true 表示
  // left.vectorized_ms 更小、应排在 right 前面，最终选出耗时最短的一行。
  BenchmarkRow const& best = *std::min_element(
      rows.begin(), rows.end(), [](BenchmarkRow const& left,
                                   BenchmarkRow const& right) {
        return left.vectorized_ms < right.vectorized_ms;
      });
  std::cout << std::defaultfloat
            << "\nbest requested-vector path: block=" << best.threads
            << ", " << best.vectorized_ms << " ms, "
            << effective_gb_per_second(count, best.vectorized_ms)
            << " effective GB/s (cache-sensitive)\n";

  if (!all_passed) {
    for (BenchmarkRow const& row : rows) {
      std::string const scalar_label =
          "block " + std::to_string(row.threads) + " scalar correctness";
      std::string const vector_label =
          "block " + std::to_string(row.threads) + " vector correctness";
      print_error_stats(scalar_label.c_str(), row.scalar_verification.errors);
      print_error_stats(vector_label.c_str(),
                        row.vectorized_verification.errors);
      std::cout << "guards(block " << row.threads << "): scalar="
                << row.scalar_verification.guards_unchanged
                << ", vector=" << row.vectorized_verification.guards_unchanged
                << '\n';
    }
  }
  std::cout << "correctness and guard canaries: "
            << (all_passed ? "PASS\n" : "FAIL\n");
  return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
