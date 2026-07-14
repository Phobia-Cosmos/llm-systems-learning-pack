#include "cuda_helpers.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

inline void cublas_check(cublasStatus_t status, char const* expression,
                         char const* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error " << static_cast<int>(status) << " at " << file
              << ':' << line << " for " << expression << '\n';
    std::exit(EXIT_FAILURE);
  }
}

#define CUBLAS_CHECK(expression) \
  cublas_check((expression), #expression, __FILE__, __LINE__)

inline void cutlass_check(cutlass::Status status, char const* expression,
                          char const* file, int line) {
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS error at " << file << ':' << line << " for "
              << expression << ": " << cutlassGetStatusString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

#define CUTLASS_CHECK(expression) \
  cutlass_check((expression), #expression, __FILE__, __LINE__)

double gemm_tflops(int m, int n, int k, float milliseconds) {
  return 2.0 * static_cast<double>(m) * n * k / (milliseconds * 1.0e9);
}

int main(int argc, char** argv) {
  int const m = argc > 1 ? parse_positive_int(argv[1], "M") : 513;
  int const n = argc > 2 ? parse_positive_int(argv[2], "N") : 509;
  int const k = argc > 3 ? parse_positive_int(argv[3], "K") : 257;
  int const iterations = argc > 4 ? parse_positive_int(argv[4], "iterations")
                                  : 30;

  std::size_t const a_count = static_cast<std::size_t>(m) * k;
  std::size_t const b_count = static_cast<std::size_t>(k) * n;
  std::size_t const c_count = static_cast<std::size_t>(m) * n;
  std::vector<float> a(a_count);
  std::vector<float> b(b_count);
  std::vector<float> c(c_count);
  for (std::size_t i = 0; i < a_count; ++i) {
    a[i] = static_cast<float>(static_cast<int>((i * 17) % 23) - 11) / 23.0F;
  }
  for (std::size_t i = 0; i < b_count; ++i) {
    b[i] = static_cast<float>(static_cast<int>((i * 13) % 19) - 9) / 19.0F;
  }
  for (std::size_t i = 0; i < c_count; ++i) {
    c[i] = static_cast<float>(static_cast<int>((i * 7) % 11) - 5) / 11.0F;
  }

  DeviceBuffer<float> device_a(a_count);
  DeviceBuffer<float> device_b(b_count);
  DeviceBuffer<float> device_c(c_count);
  DeviceBuffer<float> device_cutlass(c_count);
  DeviceBuffer<float> device_cublas(c_count);
  device_a.copy_from_host(a.data(), a_count);
  device_b.copy_from_host(b.data(), b_count);
  device_c.copy_from_host(c.data(), c_count);
  device_cublas.copy_from_host(c.data(), c_count);

  using RowMajor = cutlass::layout::RowMajor;
  using CutlassSgemm =
      cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float,
                                  RowMajor>;
  CutlassSgemm gemm;

  float const alpha = 1.25F;
  float const beta = 0.25F;
  CutlassSgemm::Arguments correctness_arguments(
      {m, n, k}, {device_a.data(), k}, {device_b.data(), n},
      {device_c.data(), n}, {device_cutlass.data(), n}, {alpha, beta});
  CUTLASS_CHECK(gemm.can_implement(correctness_arguments));
  std::size_t const workspace_bytes = gemm.get_workspace_size(correctness_arguments);
  DeviceBuffer<std::uint8_t> workspace(workspace_bytes);
  CUTLASS_CHECK(gemm.initialize(correctness_arguments, workspace.data()));
  CUTLASS_CHECK(gemm.run());

  cublasHandle_t cublas = nullptr;
  CUBLAS_CHECK(cublasCreate(&cublas));
  cublasStatus_t const math_mode_status =
      cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH);
  if (math_mode_status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "warning: cuBLAS pedantic math mode is unavailable; using default math\n";
  }

  // cuBLAS is column-major.  Swapping A/B and M/N computes the transpose of
  // row-major C without physically transposing any allocation.
  CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                           &alpha, device_b.data(), n, device_a.data(), k,
                           &beta, device_cublas.data(), n));

  std::vector<float> cutlass_result(c_count);
  std::vector<float> cublas_result(c_count);
  device_cutlass.copy_to_host(cutlass_result.data(), c_count);
  device_cublas.copy_to_host(cublas_result.data(), c_count);
  ErrorStats const errors =
      compare_vectors(cutlass_result, cublas_result, 5.0e-3, 5.0e-3);

  float const benchmark_alpha = 1.0F;
  float const benchmark_beta = 0.0F;
  CutlassSgemm::Arguments benchmark_arguments(
      {m, n, k}, {device_a.data(), k}, {device_b.data(), n},
      {device_c.data(), n}, {device_cutlass.data(), n},
      {benchmark_alpha, benchmark_beta});
  CUTLASS_CHECK(gemm.initialize(benchmark_arguments, workspace.data()));
  auto launch_cutlass = [&] { CUTLASS_CHECK(gemm.run()); };
  float const cutlass_ms = benchmark_cuda_ms(launch_cutlass, 5, iterations);

  auto launch_cublas = [&] {
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             &benchmark_alpha, device_b.data(), n,
                             device_a.data(), k, &benchmark_beta,
                             device_cublas.data(), n));
  };
  float const cublas_ms = benchmark_cuda_ms(launch_cublas, 5, iterations);

  CUBLAS_CHECK(cublasDestroy(cublas));

  std::cout << "CUTLASS SGEMM vs cuBLAS (row major, FP32 CUDA cores)\n"
            << "shape: M=" << m << ", N=" << n << ", K=" << k << '\n'
            << "CUTLASS: " << cutlass_ms << " ms, "
            << gemm_tflops(m, n, k, cutlass_ms) << " TFLOP/s\n"
            << "cuBLAS:   " << cublas_ms << " ms, "
            << gemm_tflops(m, n, k, cublas_ms) << " TFLOP/s\n";
  print_error_stats("CUTLASS vs cuBLAS correctness (alpha=1.25, beta=0.25)",
                    errors);
  std::cout << (errors.failures == 0 ? "PASS\n" : "FAIL\n");
  return errors.failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

