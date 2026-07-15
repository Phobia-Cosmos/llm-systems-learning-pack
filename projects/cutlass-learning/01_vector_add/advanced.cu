#include "cuda_helpers.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

__global__ void vector_add_float_scalar_advanced(
    float const* __restrict__ a, float const* __restrict__ b,
    float* __restrict__ c, std::size_t count) {
  std::size_t const index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    c[index] = a[index] + b[index];
  }
}

__global__ void vector_add_float4_advanced(
    float4 const* __restrict__ a, float4 const* __restrict__ b,
    float4* __restrict__ c, std::size_t pack_count,
    std::size_t tail_count) {
  std::size_t const index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < pack_count) {
    float4 const av = a[index];
    float4 const bv = b[index];
    c[index] = make_float4(av.x + bv.x, av.y + bv.y, av.z + bv.z,
                           av.w + bv.w);
  }
  if (index < tail_count) {
    std::size_t const scalar_index = pack_count * 4 + index;
    auto const* scalar_a = reinterpret_cast<float const*>(a);
    auto const* scalar_b = reinterpret_cast<float const*>(b);
    auto* scalar_c = reinterpret_cast<float*>(c);
    scalar_c[scalar_index] = scalar_a[scalar_index] + scalar_b[scalar_index];
  }
}

__global__ void vector_add_half_scalar_advanced(
    __half const* __restrict__ a, __half const* __restrict__ b,
    __half* __restrict__ c, std::size_t count) {
  std::size_t const index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    c[index] = __hadd(a[index], b[index]);
  }
}

__global__ void vector_add_half2_advanced(
    __half2 const* __restrict__ a, __half2 const* __restrict__ b,
    __half2* __restrict__ c, std::size_t pack_count,
    std::size_t tail_count) {
  std::size_t const index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < pack_count) {
    c[index] = __hadd2(a[index], b[index]);
  }
  if (index < tail_count) {
    std::size_t const scalar_index = pack_count * 2 + index;
    auto const* scalar_a = reinterpret_cast<__half const*>(a);
    auto const* scalar_b = reinterpret_cast<__half const*>(b);
    auto* scalar_c = reinterpret_cast<__half*>(c);
    scalar_c[scalar_index] = __hadd(scalar_a[scalar_index],
                                    scalar_b[scalar_index]);
  }
}

__global__ void vector_add_int_scalar_advanced(
    int const* __restrict__ a, int const* __restrict__ b,
    int* __restrict__ c, std::size_t count) {
  std::size_t const index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    c[index] = a[index] + b[index];
  }
}

// CUDA int4 是四个 int32（共 16 byte）的向量类型，不是量化中的 4-bit INT4。
__global__ void vector_add_int4_advanced(
    int4 const* __restrict__ a, int4 const* __restrict__ b,
    int4* __restrict__ c, std::size_t pack_count,
    std::size_t tail_count) {
  std::size_t const index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < pack_count) {
    int4 const av = a[index];
    int4 const bv = b[index];
    c[index] = make_int4(av.x + bv.x, av.y + bv.y, av.z + bv.z,
                         av.w + bv.w);
  }
  if (index < tail_count) {
    std::size_t const scalar_index = pack_count * 4 + index;
    auto const* scalar_a = reinterpret_cast<int const*>(a);
    auto const* scalar_b = reinterpret_cast<int const*>(b);
    auto* scalar_c = reinterpret_cast<int*>(c);
    scalar_c[scalar_index] = scalar_a[scalar_index] + scalar_b[scalar_index];
  }
}

struct Options {
  std::size_t count = 16'777'219;
  int iterations = 20;
  int rounds = 10;
  int warmup = 3;
  int min_block = 32;
  int max_block = 1024;
  int block_step = 32;
  std::size_t offset = 0;
  std::string types = "all";
  std::string variants = "all";
  std::string csv_path = "results/vector_add_advanced.csv";
};

void print_usage(char const* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --n N                 element count\n"
      << "  --iterations N        launches timed inside each round\n"
      << "  --rounds N            independent rounds (default 10)\n"
      << "  --warmup N            warm-up launches before every round\n"
      << "  --min-block N         first threads/block value\n"
      << "  --max-block N         last threads/block value\n"
      << "  --step N              block-size step; all values must be warp multiples\n"
      << "  --block N             shorthand for min=max=N\n"
      << "  --offset N            element offset; 1 deliberately misaligns packed types\n"
      << "  --types all|float|half|int\n"
      << "  --variants all|scalar|vector\n"
      << "  --csv PATH            output CSV\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  auto require_value = [&](int& index, char const* flag) -> char const* {
    if (index + 1 >= argc) {
      throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    return argv[++index];
  };

  for (int i = 1; i < argc; ++i) {
    std::string const flag = argv[i];
    if (flag == "--help") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (flag == "--n") {
      options.count = static_cast<std::size_t>(
          parse_positive_int(require_value(i, "--n"), "N"));
    } else if (flag == "--iterations") {
      options.iterations = parse_positive_int(
          require_value(i, "--iterations"), "iterations");
    } else if (flag == "--rounds") {
      options.rounds =
          parse_positive_int(require_value(i, "--rounds"), "rounds");
    } else if (flag == "--warmup") {
      options.warmup = parse_nonnegative_int(
          require_value(i, "--warmup"), "warmup");
    } else if (flag == "--min-block") {
      options.min_block = parse_positive_int(
          require_value(i, "--min-block"), "minimum block size");
    } else if (flag == "--max-block") {
      options.max_block = parse_positive_int(
          require_value(i, "--max-block"), "maximum block size");
    } else if (flag == "--step") {
      options.block_step =
          parse_positive_int(require_value(i, "--step"), "block step");
    } else if (flag == "--block") {
      int const block =
          parse_positive_int(require_value(i, "--block"), "block size");
      options.min_block = block;
      options.max_block = block;
      options.block_step = 32;
    } else if (flag == "--offset") {
      options.offset = static_cast<std::size_t>(
          parse_nonnegative_int(require_value(i, "--offset"), "offset"));
    } else if (flag == "--types") {
      options.types = require_value(i, "--types");
    } else if (flag == "--variants") {
      options.variants = require_value(i, "--variants");
    } else if (flag == "--csv") {
      options.csv_path = require_value(i, "--csv");
    } else {
      throw std::invalid_argument("unknown option: " + flag);
    }
  }

  auto valid_choice = [](std::string const& value,
                         std::vector<std::string> const& choices) {
    return std::find(choices.begin(), choices.end(), value) != choices.end();
  };
  if (!valid_choice(options.types, {"all", "float", "half", "int"})) {
    throw std::invalid_argument("--types must be all, float, half, or int");
  }
  if (!valid_choice(options.variants, {"all", "scalar", "vector"})) {
    throw std::invalid_argument("--variants must be all, scalar, or vector");
  }
  if (options.offset > 1024) {
    throw std::invalid_argument("--offset must be <= 1024 elements");
  }
  return options;
}

int grid_size(std::size_t work_items, int threads) {
  return static_cast<int>((work_items + static_cast<std::size_t>(threads) - 1) /
                          static_cast<std::size_t>(threads));
}

bool is_aligned(void const* pointer, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

double quantile(std::vector<double> values, double probability) {
  if (values.empty()) {
    throw std::invalid_argument("quantile requires at least one sample");
  }
  std::sort(values.begin(), values.end());
  double const position = probability * static_cast<double>(values.size() - 1);
  std::size_t const lower = static_cast<std::size_t>(std::floor(position));
  std::size_t const upper = static_cast<std::size_t>(std::ceil(position));
  double const fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

struct Statistics {
  double median_ms = 0.0;
  double p95_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  std::vector<double> samples_ms;
};

class NvtxRange {
 public:
  explicit NvtxRange(std::string const& label) { nvtxRangePushA(label.c_str()); }
  NvtxRange(NvtxRange const&) = delete;
  NvtxRange& operator=(NvtxRange const&) = delete;
  ~NvtxRange() { nvtxRangePop(); }
};

template <typename Launch>
Statistics collect_statistics(Launch&& launch, Options const& options,
                              std::string const& label_prefix) {
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(options.rounds));

  for (int round = 0; round < options.rounds; ++round) {
    for (int i = 0; i < options.warmup; ++i) {
      launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::string const label =
        label_prefix + "/round=" + std::to_string(round);
    NvtxRange range(label);
    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < options.iterations; ++i) {
      launch();
    }
    double const average_ms =
        static_cast<double>(timer.stop_ms()) / options.iterations;
    CUDA_CHECK(cudaGetLastError());
    samples.push_back(average_ms);
  }

  return {quantile(samples, 0.50), quantile(samples, 0.95),
          *std::min_element(samples.begin(), samples.end()),
          *std::max_element(samples.begin(), samples.end()), samples};
}

struct CsvRow {
  std::string dtype;
  std::string requested_variant;
  std::string actual_path;
  std::size_t element_bytes = 0;
  std::size_t vector_width = 0;
  std::size_t required_alignment = 0;
  bool actual_aligned = false;
  std::size_t offset = 0;
  std::size_t count = 0;
  int block_size = 0;
  int rounds = 0;
  int iterations = 0;
  Statistics timing;
  double median_gbps = 0.0;
  double p95_latency_gbps = 0.0;
  bool correctness = false;
};

class CsvWriter {
 public:
  explicit CsvWriter(std::string const& path) : path_(path) {
    std::filesystem::path const output_path(path);
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    output_.open(path);
    if (!output_) {
      throw std::runtime_error("cannot open CSV: " + path);
    }
    output_ << "dtype,requested_variant,actual_path,element_bytes,vector_width,"
               "required_alignment_bytes,actual_aligned,offset_elements,N,"
               "block_size,rounds,iterations,median_ms,p95_ms,min_ms,max_ms,"
               "median_effective_gbps,p95_latency_effective_gbps,correctness,"
               "samples_ms\n";
  }

  void write(CsvRow const& row) {
    output_ << row.dtype << ',' << row.requested_variant << ','
            << row.actual_path << ',' << row.element_bytes << ','
            << row.vector_width << ',' << row.required_alignment << ','
            << (row.actual_aligned ? 1 : 0) << ',' << row.offset << ','
            << row.count << ',' << row.block_size << ',' << row.rounds << ','
            << row.iterations << ',' << std::setprecision(10)
            << row.timing.median_ms << ',' << row.timing.p95_ms << ','
            << row.timing.min_ms << ',' << row.timing.max_ms << ','
            << row.median_gbps << ',' << row.p95_latency_gbps << ','
            << (row.correctness ? "PASS" : "FAIL") << ',';
    for (std::size_t i = 0; i < row.timing.samples_ms.size(); ++i) {
      if (i != 0) output_ << ';';
      output_ << row.timing.samples_ms[i];
    }
    output_ << '\n';
    output_.flush();
  }

  std::string const& path() const { return path_; }

 private:
  std::string path_;
  std::ofstream output_;
};

struct Summary {
  std::string dtype;
  std::string variant;
  std::string actual_path;
  int best_block = 0;
  double best_median_gbps = 0.0;
  double best_median_ms = std::numeric_limits<double>::infinity();
  bool correctness = false;
};

double effective_gbps(std::size_t count, std::size_t element_bytes,
                      double milliseconds) {
  double const bytes = 3.0 * static_cast<double>(count) * element_bytes;
  return bytes / (milliseconds * 1.0e6);
}

template <typename Launch, typename Prepare, typename Verify>
void run_variant(std::string const& dtype, std::string const& requested_variant,
                 std::string const& actual_path, std::size_t element_bytes,
                 std::size_t vector_width, std::size_t required_alignment,
                 bool actual_aligned, Options const& options,
                 std::vector<int> const& blocks, Launch&& launch,
                 Prepare&& prepare, Verify&& verify, CsvWriter& csv,
                 std::vector<Summary>& summaries) {
  Summary summary{dtype, requested_variant, actual_path, 0, 0.0,
                  std::numeric_limits<double>::infinity(), true};
  for (int block : blocks) {
    // Reset poison/guards for every block configuration. Verification after
    // the measured launches therefore belongs to this exact CSV row rather
    // than being copied from blocks.front().
    prepare();
    std::string const label = dtype + '/' + requested_variant + "/path=" +
                              actual_path + "/block=" +
                              std::to_string(block);
    Statistics const timing = collect_statistics(
        [&] { launch(block); }, options, label);
    bool const correctness = verify();
    summary.correctness = summary.correctness && correctness;
    double const median_bandwidth =
        effective_gbps(options.count, element_bytes, timing.median_ms);
    double const p95_bandwidth =
        effective_gbps(options.count, element_bytes, timing.p95_ms);
    csv.write({dtype,
               requested_variant,
               actual_path,
               element_bytes,
               vector_width,
               required_alignment,
               actual_aligned,
               options.offset,
               options.count,
               block,
               options.rounds,
               options.iterations,
               timing,
               median_bandwidth,
               p95_bandwidth,
               correctness});
    if (timing.median_ms < summary.best_median_ms) {
      summary.best_median_ms = timing.median_ms;
      summary.best_median_gbps = median_bandwidth;
      summary.best_block = block;
    }
  }
  summaries.push_back(summary);
}

template <typename T>
bool guard_bits_equal(T const& value, unsigned char expected_byte) {
  unsigned char bytes[sizeof(T)]{};
  std::memcpy(bytes, &value, sizeof(T));
  return std::all_of(std::begin(bytes), std::end(bytes),
                     [=](unsigned char byte) { return byte == expected_byte; });
}

void run_float(Options const& options, std::vector<int> const& blocks,
               CsvWriter& csv, std::vector<Summary>& summaries) {
  std::size_t constexpr prefix_guard = 8;
  std::size_t constexpr suffix_guard = 8;
  std::size_t const logical_offset = prefix_guard + options.offset;
  std::size_t const allocation = logical_offset + options.count + suffix_guard;
  std::vector<float> a(options.count);
  std::vector<float> b(options.count);
  std::vector<float> expected(options.count);
  for (std::size_t i = 0; i < options.count; ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 251) - 125) / 251.0F;
    b[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 127.0F;
    expected[i] = a[i] + b[i];
  }
  DeviceBuffer<float> device_a(allocation);
  DeviceBuffer<float> device_b(allocation);
  DeviceBuffer<float> device_c(allocation);
  CUDA_CHECK(cudaMemset(device_a.data(), 0xa5, allocation * sizeof(float)));
  CUDA_CHECK(cudaMemset(device_b.data(), 0xa5, allocation * sizeof(float)));
  device_a.copy_from_host(a.data(), options.count, logical_offset);
  device_b.copy_from_host(b.data(), options.count, logical_offset);
  float const* input_a = device_a.data() + logical_offset;
  float const* input_b = device_b.data() + logical_offset;
  float* output = device_c.data() + logical_offset;

  auto prepare = [&] {
    CUDA_CHECK(cudaMemset(device_c.data(), 0xa5,
                          allocation * sizeof(float)));
  };
  auto verify = [&] {
    std::vector<float> storage(allocation);
    device_c.copy_to_host(storage.data(), allocation);
    for (std::size_t i = 0; i < logical_offset; ++i) {
      if (!guard_bits_equal(storage[i], 0xa5)) return false;
    }
    for (std::size_t i = 0; i < options.count; ++i) {
      float const got = storage[logical_offset + i];
      if (!std::isfinite(got) || std::abs(got - expected[i]) > 1.0e-6F)
        return false;
    }
    for (std::size_t i = logical_offset + options.count; i < allocation; ++i) {
      if (!guard_bits_equal(storage[i], 0xa5)) return false;
    }
    return true;
  };
  auto scalar = [&](int block) {
    vector_add_float_scalar_advanced<<<grid_size(options.count, block), block>>>(
        input_a, input_b, output, options.count);
  };

  bool const aligned = is_aligned(input_a, alignof(float4)) &&
                       is_aligned(input_b, alignof(float4)) &&
                       is_aligned(output, alignof(float4));
  bool const use_vector = aligned && options.count >= 4;
  std::string const path = use_vector
                               ? (options.count % 4 == 0 ? "float4"
                                                        : "float4_scalar_tail")
                               : (aligned ? "scalar_small_N_fallback"
                                          : "scalar_alignment_fallback");
  auto vectorized = [&](int block) {
    if (!use_vector) {
      scalar(block);
      return;
    }
    std::size_t const packs = options.count / 4;
    vector_add_float4_advanced<<<grid_size(packs, block), block>>>(
        reinterpret_cast<float4 const*>(input_a),
        reinterpret_cast<float4 const*>(input_b),
        reinterpret_cast<float4*>(output), packs, options.count % 4);
  };

  if (options.variants == "all" || options.variants == "scalar") {
    run_variant("float32", "scalar", "scalar", sizeof(float), 1,
                alignof(float), true, options, blocks, scalar, prepare, verify,
                csv, summaries);
  }
  if (options.variants == "all" || options.variants == "vector") {
    run_variant("float32", "float4", path, sizeof(float), 4,
                alignof(float4), aligned, options, blocks, vectorized, prepare,
                verify, csv, summaries);
  }
}

void run_half(Options const& options, std::vector<int> const& blocks,
              CsvWriter& csv, std::vector<Summary>& summaries) {
  std::size_t constexpr prefix_guard = 8;
  std::size_t constexpr suffix_guard = 8;
  std::size_t const logical_offset = prefix_guard + options.offset;
  std::size_t const allocation = logical_offset + options.count + suffix_guard;
  std::vector<__half> a(options.count);
  std::vector<__half> b(options.count);
  std::vector<__half> expected(options.count);
  for (std::size_t i = 0; i < options.count; ++i) {
    a[i] = __float2half_rn(
        static_cast<float>(static_cast<int>(i % 251) - 125) / 251.0F);
    b[i] = __float2half_rn(
        static_cast<float>(static_cast<int>(i % 127) - 63) / 127.0F);
    expected[i] = __float2half_rn(__half2float(a[i]) + __half2float(b[i]));
  }
  DeviceBuffer<__half> device_a(allocation);
  DeviceBuffer<__half> device_b(allocation);
  DeviceBuffer<__half> device_c(allocation);
  CUDA_CHECK(cudaMemset(device_a.data(), 0xa5, allocation * sizeof(__half)));
  CUDA_CHECK(cudaMemset(device_b.data(), 0xa5, allocation * sizeof(__half)));
  device_a.copy_from_host(a.data(), options.count, logical_offset);
  device_b.copy_from_host(b.data(), options.count, logical_offset);
  __half const* input_a = device_a.data() + logical_offset;
  __half const* input_b = device_b.data() + logical_offset;
  __half* output = device_c.data() + logical_offset;

  auto prepare = [&] {
    CUDA_CHECK(cudaMemset(device_c.data(), 0xa5,
                          allocation * sizeof(__half)));
  };
  auto verify = [&] {
    std::vector<__half> storage(allocation);
    device_c.copy_to_host(storage.data(), allocation);
    for (std::size_t i = 0; i < logical_offset; ++i) {
      if (!guard_bits_equal(storage[i], 0xa5)) return false;
    }
    for (std::size_t i = 0; i < options.count; ++i) {
      float const got = __half2float(storage[logical_offset + i]);
      float const want = __half2float(expected[i]);
      if (!std::isfinite(got) || std::abs(got - want) > 1.0e-3F)
        return false;
    }
    for (std::size_t i = logical_offset + options.count; i < allocation; ++i) {
      if (!guard_bits_equal(storage[i], 0xa5)) return false;
    }
    return true;
  };
  auto scalar = [&](int block) {
    vector_add_half_scalar_advanced<<<grid_size(options.count, block), block>>>(
        input_a, input_b, output, options.count);
  };

  bool const aligned = is_aligned(input_a, alignof(__half2)) &&
                       is_aligned(input_b, alignof(__half2)) &&
                       is_aligned(output, alignof(__half2));
  bool const use_vector = aligned && options.count >= 2;
  std::string const path = use_vector
                               ? (options.count % 2 == 0 ? "half2"
                                                        : "half2_scalar_tail")
                               : (aligned ? "scalar_small_N_fallback"
                                          : "scalar_alignment_fallback");
  auto vectorized = [&](int block) {
    if (!use_vector) {
      scalar(block);
      return;
    }
    std::size_t const packs = options.count / 2;
    vector_add_half2_advanced<<<grid_size(packs, block), block>>>(
        reinterpret_cast<__half2 const*>(input_a),
        reinterpret_cast<__half2 const*>(input_b),
        reinterpret_cast<__half2*>(output), packs, options.count % 2);
  };

  if (options.variants == "all" || options.variants == "scalar") {
    run_variant("float16", "scalar", "scalar", sizeof(__half), 1,
                alignof(__half), true, options, blocks, scalar, prepare, verify,
                csv, summaries);
  }
  if (options.variants == "all" || options.variants == "vector") {
    run_variant("float16", "half2", path, sizeof(__half), 2,
                alignof(__half2), aligned, options, blocks, vectorized, prepare,
                verify, csv, summaries);
  }
}

void run_int(Options const& options, std::vector<int> const& blocks,
             CsvWriter& csv, std::vector<Summary>& summaries) {
  std::size_t constexpr prefix_guard = 8;
  std::size_t constexpr suffix_guard = 8;
  std::size_t const logical_offset = prefix_guard + options.offset;
  std::size_t const allocation = logical_offset + options.count + suffix_guard;
  std::vector<int> a(options.count);
  std::vector<int> b(options.count);
  std::vector<int> expected(options.count);
  for (std::size_t i = 0; i < options.count; ++i) {
    a[i] = static_cast<int>(i % 101) - 50;
    b[i] = static_cast<int>((i * 3) % 97) - 48;
    expected[i] = a[i] + b[i];
  }
  DeviceBuffer<int> device_a(allocation);
  DeviceBuffer<int> device_b(allocation);
  DeviceBuffer<int> device_c(allocation);
  CUDA_CHECK(cudaMemset(device_a.data(), 0xa5, allocation * sizeof(int)));
  CUDA_CHECK(cudaMemset(device_b.data(), 0xa5, allocation * sizeof(int)));
  device_a.copy_from_host(a.data(), options.count, logical_offset);
  device_b.copy_from_host(b.data(), options.count, logical_offset);
  int const* input_a = device_a.data() + logical_offset;
  int const* input_b = device_b.data() + logical_offset;
  int* output = device_c.data() + logical_offset;

  auto prepare = [&] {
    CUDA_CHECK(cudaMemset(device_c.data(), 0xa5,
                          allocation * sizeof(int)));
  };
  auto verify = [&] {
    std::vector<int> storage(allocation);
    device_c.copy_to_host(storage.data(), allocation);
    for (std::size_t i = 0; i < logical_offset; ++i) {
      if (!guard_bits_equal(storage[i], 0xa5)) return false;
    }
    for (std::size_t i = 0; i < options.count; ++i) {
      if (storage[logical_offset + i] != expected[i]) return false;
    }
    for (std::size_t i = logical_offset + options.count; i < allocation; ++i) {
      if (!guard_bits_equal(storage[i], 0xa5)) return false;
    }
    return true;
  };
  auto scalar = [&](int block) {
    vector_add_int_scalar_advanced<<<grid_size(options.count, block), block>>>(
        input_a, input_b, output, options.count);
  };

  bool const aligned = is_aligned(input_a, alignof(int4)) &&
                       is_aligned(input_b, alignof(int4)) &&
                       is_aligned(output, alignof(int4));
  bool const use_vector = aligned && options.count >= 4;
  std::string const path = use_vector
                               ? (options.count % 4 == 0 ? "int4"
                                                        : "int4_scalar_tail")
                               : (aligned ? "scalar_small_N_fallback"
                                          : "scalar_alignment_fallback");
  auto vectorized = [&](int block) {
    if (!use_vector) {
      scalar(block);
      return;
    }
    std::size_t const packs = options.count / 4;
    vector_add_int4_advanced<<<grid_size(packs, block), block>>>(
        reinterpret_cast<int4 const*>(input_a),
        reinterpret_cast<int4 const*>(input_b),
        reinterpret_cast<int4*>(output), packs, options.count % 4);
  };

  if (options.variants == "all" || options.variants == "scalar") {
    run_variant("int32", "scalar", "scalar", sizeof(int), 1, alignof(int),
                true, options, blocks, scalar, prepare, verify, csv, summaries);
  }
  if (options.variants == "all" || options.variants == "vector") {
    run_variant("int32", "int4", path, sizeof(int), 4, alignof(int4),
                aligned, options, blocks, vectorized, prepare, verify, csv,
                summaries);
  }
}

int main(int argc, char** argv) {
  try {
    Options const options = parse_options(argc, argv);
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    int const warp = properties.warpSize;
    if (options.min_block > options.max_block ||
        options.min_block % warp != 0 || options.max_block % warp != 0 ||
        options.block_step % warp != 0) {
      throw std::invalid_argument(
          "min/max/step must describe positive multiples of the warp size");
    }
    if (options.max_block > properties.maxThreadsPerBlock) {
      throw std::invalid_argument("maximum block size exceeds device limit");
    }

    std::vector<int> blocks;
    for (int block = options.min_block; block <= options.max_block;
         block += options.block_step) {
      blocks.push_back(block);
    }
    if (blocks.empty()) {
      throw std::invalid_argument("block-size sweep is empty");
    }

    CsvWriter csv(options.csv_path);
    std::vector<Summary> summaries;
    if (options.types == "all" || options.types == "float") {
      run_float(options, blocks, csv, summaries);
    }
    if (options.types == "all" || options.types == "half") {
      run_half(options, blocks, csv, summaries);
    }
    if (options.types == "all" || options.types == "int") {
      run_int(options, blocks, csv, summaries);
    }

    bool all_passed = true;
    std::cout << "Vector Add advanced summary\n"
              << "N=" << options.count << ", rounds=" << options.rounds
              << ", iterations/round=" << options.iterations
              << ", block range=" << options.min_block << ':'
              << options.block_step << ':' << options.max_block
              << ", offset=" << options.offset << " elements\n";
    for (Summary const& summary : summaries) {
      std::cout << std::left << std::setw(8) << summary.dtype << std::setw(9)
                << summary.variant << std::setw(28) << summary.actual_path
                << " best_block=" << std::setw(4) << summary.best_block
                << " median=" << std::fixed << std::setprecision(2)
                << summary.best_median_gbps << " GB/s, correctness="
                << (summary.correctness ? "PASS" : "FAIL") << '\n';
      all_passed = all_passed && summary.correctness;
    }
    std::cout << "CSV: " << csv.path() << '\n';
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (std::exception const& error) {
    std::cerr << "vector_add_advanced: " << error.what() << '\n';
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
}
