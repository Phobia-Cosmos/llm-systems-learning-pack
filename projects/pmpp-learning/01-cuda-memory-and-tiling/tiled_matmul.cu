#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                              \
        if (error != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)            \
                      << " (line " << __LINE__ << ")\n";                      \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

__global__ void tiled_matmul(const float* A, const float* B, float* P,
                             int height_a, int width_a, int width_b) {
    extern __shared__ float shared[];
    const int tile_width = blockDim.x;
    float* tile_a = shared;
    float* tile_b = shared + tile_width * tile_width;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * tile_width + ty;  // y -> matrix row
    const int col = blockIdx.x * tile_width + tx;  // x -> matrix column
    const int local = ty * tile_width + tx;
    float sum = 0.0f;

    const int phase_count = (width_a + tile_width - 1) / tile_width;
    for (int phase = 0; phase < phase_count; ++phase) {
        const int a_col = phase * tile_width + tx;
        const int b_row = phase * tile_width + ty;

        // In this simple square-tile version, each thread loads one element
        // from each input matrix during every phase.
        tile_a[local] = (row < height_a && a_col < width_a)
                            ? A[row * width_a + a_col]
                            : 0.0f;
        tile_b[local] = (b_row < width_a && col < width_b)
                            ? B[b_row * width_b + col]
                            : 0.0f;
        __syncthreads();

        for (int k = 0; k < tile_width; ++k) {
            sum += tile_a[ty * tile_width + k] *
                   tile_b[k * tile_width + tx];
        }
        __syncthreads();  // All users finish before the next phase overwrites.
    }

    if (row < height_a && col < width_b) {
        P[row * width_b + col] = sum;
    }
}

void print_matrix(const std::vector<float>& matrix, int rows, int cols,
                  const char* name) {
    std::cout << name << " (usual [row][col] layout):\n";
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout << std::setw(7) << matrix[row * cols + col];
        }
        std::cout << '\n';
    }
}

int main() {
    constexpr int height_a = 4;
    constexpr int width_a = 4;
    constexpr int width_b = 4;
    constexpr int tile = 2;

    std::vector<float> A(height_a * width_a);
    std::vector<float> B(width_a * width_b);
    std::vector<float> P(height_a * width_b);
    for (int i = 0; i < static_cast<int>(A.size()); ++i) A[i] = i + 1.0f;
    for (int row = 0; row < width_a; ++row) {
        for (int col = 0; col < width_b; ++col) {
            B[row * width_b + col] = (row == col) ? 1.0f : 0.0f;
        }
    }

    float *d_A = nullptr, *d_B = nullptr, *d_P = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P, P.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 threads(tile, tile);
    dim3 blocks((width_b + tile - 1) / tile,
                (height_a + tile - 1) / tile);
    const size_t dynamic_shared_bytes =
        2 * tile * tile * sizeof(float);
    tiled_matmul<<<blocks, threads, dynamic_shared_bytes>>>(
        d_A, d_B, d_P, height_a, width_a, width_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(P.data(), d_P, P.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    print_matrix(A, height_a, width_a, "A");
    print_matrix(B, width_a, width_b, "B (identity)");
    print_matrix(P, height_a, width_b, "P = A * B");

    bool correct = true;
    for (int i = 0; i < static_cast<int>(P.size()); ++i) {
        correct = correct && std::fabs(P[i] - A[i]) < 1e-5f;
    }
    std::cout << (correct ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
