#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

inline void cuda_check(cudaError_t status, char const* expression,
                       char const* file, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ':' << line << " for "
              << expression << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

// expression 会先执行并产生 cudaError_t；#expression 把调用文本转成字符串，
// __FILE__ / __LINE__ 是宏展开（也就是调用 CUDA_CHECK）位置的文件名和行号，
// 因而错误信息会指向调用者，而不是统一指向 cuda_check 函数内部。
#define CUDA_CHECK(expression) \
  cuda_check((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ != 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        cudaFree(data_);
      }
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  T* data() { return data_; }
  T const* data() const { return data_; }
  std::size_t size() const { return count_; }

  void copy_from_host(T const* source, std::size_t count,
                      std::size_t destination_offset = 0) {
    if (destination_offset > count_ || count > count_ - destination_offset) {
      throw std::out_of_range("host-to-device copy exceeds DeviceBuffer");
    }
    CUDA_CHECK(cudaMemcpy(data_ + destination_offset, source, count * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  void copy_to_host(T* destination, std::size_t count,
                    std::size_t source_offset = 0) const {
    if (source_offset > count_ || count > count_ - source_offset) {
      throw std::out_of_range("device-to-host copy exceeds DeviceBuffer");
    }
    CUDA_CHECK(cudaMemcpy(destination, data_ + source_offset, count * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

class CudaEventTimer {
 public:
  CudaEventTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~CudaEventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { CUDA_CHECK(cudaEventRecord(start_)); }

  float stop_ms() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

// Launch 是由编译器从传入的 lambda/函数对象自动推导出的类型。Launch&& 在这里
// 是 forwarding reference：既能接临时对象也能接已有对象，且不必把闭包复制一份。
template <typename Launch>
float benchmark_cuda_ms(Launch&& launch, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaEventTimer timer;
  timer.start();
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  float const total_ms = timer.stop_ms();
  CUDA_CHECK(cudaGetLastError());
  return total_ms / static_cast<float>(iterations);
}

inline int parse_positive_int(char const* text, char const* name) {
  try {
    std::size_t consumed = 0;
    long long const value = std::stoll(text, &consumed);
    if (consumed != std::string(text).size() || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("range");
    }
    return static_cast<int>(value);
  } catch (std::exception const&) {
    std::cerr << name << " must be a positive 32-bit integer, got: " << text
              << '\n';
    std::exit(EXIT_FAILURE);
  }
}

inline int parse_nonnegative_int(char const* text, char const* name) {
  try {
    std::size_t consumed = 0;
    long long const value = std::stoll(text, &consumed);
    if (consumed != std::string(text).size() || value < 0 ||
        value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("range");
    }
    return static_cast<int>(value);
  } catch (std::exception const&) {
    std::cerr << name << " must be a non-negative 32-bit integer, got: "
              << text << '\n';
    std::exit(EXIT_FAILURE);
  }
}

struct ErrorStats {
  double max_abs = 0.0;
  double max_rel = 0.0;
  std::size_t max_abs_index = 0;
  std::size_t failures = 0;
};

// absolute_tolerance 保护接近 0 的结果；relative_tolerance 允许误差随参考值
// 量级增长。逐元素判据是 |got-want| <= atol + rtol*|want|。
template <typename T, typename U>
ErrorStats compare_vectors(std::vector<T> const& actual,
                           std::vector<U> const& expected,
                           double absolute_tolerance,
                           double relative_tolerance) {
  if (actual.size() != expected.size()) {
    throw std::invalid_argument("vector sizes differ");
  }

  ErrorStats stats;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    double const got = static_cast<double>(actual[i]);
    double const want = static_cast<double>(expected[i]);
    double const absolute = std::abs(got - want);
    // 相对误差 = 绝对误差 / 参考值的量级。参考值为 0 时不能除以 0，所以用
    // 1e-12 作分母下限；最终是否失败仍使用上面的 atol+rtol 联合判据。
    double const relative = absolute / std::max(std::abs(want), 1.0e-12);
    if (absolute > stats.max_abs) {
      stats.max_abs = absolute;
      stats.max_abs_index = i;
    }
    stats.max_rel = std::max(stats.max_rel, relative);
    if (!std::isfinite(got) ||
        absolute > absolute_tolerance + relative_tolerance * std::abs(want)) {
      ++stats.failures;
    }
  }
  return stats;
}

// max_abs 表示最坏的绝对偏差，max_abs_index 给出其位置，max_rel 表示最坏的
// 相对偏差，failures 是超过容差或出现 NaN/Inf 的元素数。它们比只输出
// PASS/FAIL 更利于定位数值问题。
inline void print_error_stats(char const* label, ErrorStats const& stats) {
  std::cout << label << ": max_abs=" << std::scientific << stats.max_abs
            << ", max_abs_index=" << stats.max_abs_index
            << ", max_rel=" << stats.max_rel
            << ", failures=" << stats.failures << std::defaultfloat << '\n';
}
