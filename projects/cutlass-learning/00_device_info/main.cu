#include "cuda_helpers.cuh"

#include <cuda_runtime.h>

#include <iostream>

// 这是最小 kernel launch smoke test。GPU 不能直接修改 main 的栈变量，所以传入
// device result 指针并写入一个已知值。只让全局第 0 个线程写，既避免多线程
// 同写产生数据竞争，也使以后把 launch 扩大为多个 block/thread 时仍然安全。
__global__ void smoke_kernel(int* result) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *result = 42;
  }
}

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "No CUDA device is visible.\n";
    return EXIT_FAILURE;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  int const architecture = properties.major * 10 + properties.minor;

  std::cout << "CUDA device " << device << ": " << properties.name << '\n'
            << "compute capability: " << properties.major << '.'
            << properties.minor << " (CMake architecture " << architecture
            << ")\n"
            << "streaming multiprocessors: " << properties.multiProcessorCount
            << '\n'
            << "global memory: "
            << properties.totalGlobalMem / (1024 * 1024) << " MiB\n"
            << "shared memory per block: "
            << properties.sharedMemPerBlock / 1024 << " KiB\n"
            << "opt-in shared memory per block: "
            << properties.sharedMemPerBlockOptin / 1024 << " KiB\n"
            << "shared memory per SM: "
            << properties.sharedMemPerMultiprocessor / 1024 << " KiB\n"
            << "L2 cache: " << properties.l2CacheSize / (1024 * 1024)
            << " MiB\n"
            << "max threads per block: " << properties.maxThreadsPerBlock
            << '\n'
            << "max resident threads per SM: "
            << properties.maxThreadsPerMultiProcessor << '\n'
            << "warp size: " << properties.warpSize << '\n';

  // DeviceBuffer<int>(1) 在 GPU 显存中分配 1 个 int。<<<1,1>>> 表示启动
  // 1 个 block、每个 block 1 个 thread，足够完成这次最小写入测试。
  DeviceBuffer<int> device_result(1);
  smoke_kernel<<<1, 1>>>(device_result.data());
  CUDA_CHECK(cudaGetLastError());

  int host_result = 0;
  // &host_result 是 CPU 目标地址；第二个 1 是复制 1 个 int（不是 1 byte）。
  device_result.copy_to_host(&host_result, 1);
  if (host_result != 42) {
    std::cerr << "Kernel smoke test failed: expected 42, got " << host_result
              << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "kernel smoke test: PASS\n";
  if (architecture == 89) {
    std::cout << "learning route: Ada / sm_89; use mma.sync, not Hopper WGMMA/TMA\n";
  } else {
    std::cout << "learning route: verify architecture-specific exercises before running\n";
  }
  return EXIT_SUCCESS;
}
