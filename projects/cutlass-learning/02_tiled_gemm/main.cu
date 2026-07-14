#include "cuda_helpers.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <vector>

__global__ void naive_gemm_kernel(float const* a, float const* b, float* c,
                                  int m, int n, int k) {
  int const column = blockIdx.x * blockDim.x + threadIdx.x;
  int const row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || column >= n) {
    return;
  }

  float accumulator = 0.0F;
  for (int inner = 0; inner < k; ++inner) {
    accumulator += a[row * k + inner] * b[inner * n + column];
  }
  c[row * n + column] = accumulator;
}

template <int Tile>
__global__ void tiled_gemm_kernel(float const* a, float const* b, float* c,
                                  int m, int n, int k) {
  __shared__ float tile_a[Tile][Tile];
  __shared__ float tile_b[Tile][Tile];

  int const column = blockIdx.x * Tile + threadIdx.x;
  int const row = blockIdx.y * Tile + threadIdx.y;
  float accumulator = 0.0F;

  for (int tile = 0; tile < (k + Tile - 1) / Tile; ++tile) {
    int const a_column = tile * Tile + threadIdx.x;
    int const b_row = tile * Tile + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && a_column < k) ? a[row * k + a_column] : 0.0F;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < k && column < n) ? b[b_row * n + column] : 0.0F;
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < Tile; ++inner) {
      accumulator += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && column < n) {
    c[row * n + column] = accumulator;
  }
}

std::vector<float> cpu_gemm(std::vector<float> const& a,
                            std::vector<float> const& b, int m, int n, int k) {
  std::vector<float> result(static_cast<std::size_t>(m) * n, 0.0F);
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      double accumulator = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        accumulator += static_cast<double>(a[row * k + inner]) *
                       static_cast<double>(b[inner * n + column]);
      }
      result[row * n + column] = static_cast<float>(accumulator);
    }
  }
  return result;
}

double gemm_tflops(int m, int n, int k, float milliseconds) {
  return 2.0 * static_cast<double>(m) * n * k / (milliseconds * 1.0e9);
}

int main(int argc, char** argv) {
  int const m = argc > 1 ? parse_positive_int(argv[1], "M") : 257;
  int const n = argc > 2 ? parse_positive_int(argv[2], "N") : 259;
  int const k = argc > 3 ? parse_positive_int(argv[3], "K") : 263;
  int const iterations = argc > 4 ? parse_positive_int(argv[4], "iterations")
                                  : 30;

  std::size_t const a_count = static_cast<std::size_t>(m) * k;
  std::size_t const b_count = static_cast<std::size_t>(k) * n;
  std::size_t const c_count = static_cast<std::size_t>(m) * n;
  std::vector<float> a(a_count);
  std::vector<float> b(b_count);
  for (std::size_t i = 0; i < a_count; ++i) {
    a[i] = static_cast<float>(static_cast<int>((i * 17) % 23) - 11) / 23.0F;
  }
  for (std::size_t i = 0; i < b_count; ++i) {
    b[i] = static_cast<float>(static_cast<int>((i * 13) % 19) - 9) / 19.0F;
  }
  std::vector<float> const expected = cpu_gemm(a, b, m, n, k);
  std::vector<float> naive(c_count);
  std::vector<float> tiled(c_count);

  DeviceBuffer<float> device_a(a_count);
  DeviceBuffer<float> device_b(b_count);
  DeviceBuffer<float> device_naive(c_count);
  DeviceBuffer<float> device_tiled(c_count);
  device_a.copy_from_host(a.data(), a_count);
  device_b.copy_from_host(b.data(), b_count);

  int constexpr tile = 16;
  dim3 const block(tile, tile);
  dim3 const grid((n + tile - 1) / tile, (m + tile - 1) / tile);
  auto launch_naive = [&] {
    naive_gemm_kernel<<<grid, block>>>(device_a.data(), device_b.data(),
                                      device_naive.data(), m, n, k);
  };
  auto launch_tiled = [&] {
    tiled_gemm_kernel<tile><<<grid, block>>>(
        device_a.data(), device_b.data(), device_tiled.data(), m, n, k);
  };

  float const naive_ms = benchmark_cuda_ms(launch_naive, 3, iterations);
  float const tiled_ms = benchmark_cuda_ms(launch_tiled, 3, iterations);
  device_naive.copy_to_host(naive.data(), c_count);
  device_tiled.copy_to_host(tiled.data(), c_count);

  ErrorStats const naive_errors =
      compare_vectors(naive, expected, 2.0e-4, 2.0e-4);
  ErrorStats const tiled_errors =
      compare_vectors(tiled, expected, 2.0e-4, 2.0e-4);

  std::cout << "naive_vs_tiled_sgemm (row major)\n"
            << "shape: M=" << m << ", N=" << n << ", K=" << k << '\n'
            << "naive: " << naive_ms << " ms, "
            << gemm_tflops(m, n, k, naive_ms) << " TFLOP/s\n"
            << "tiled: " << tiled_ms << " ms, "
            << gemm_tflops(m, n, k, tiled_ms) << " TFLOP/s\n"
            << "speedup: " << naive_ms / tiled_ms << "x\n";
  print_error_stats("naive correctness", naive_errors);
  print_error_stats("tiled correctness", tiled_errors);

  bool const passed = naive_errors.failures == 0 && tiled_errors.failures == 0;
  std::cout << (passed ? "PASS\n" : "FAIL\n");
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

