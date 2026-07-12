#include <cuda_runtime.h>

#include <cstdlib>
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

// Visible to all kernels, writable by the host, read-only in device code.
__constant__ float c_scale;

__global__ void memory_spaces(cudaTextureObject_t input_texture,
                              float* output,
                              int n) {
    // The host chooses this array's byte size in the kernel launch.
    extern __shared__ float tile[];

    const int tx = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tx;

    // Each thread loads its own position. The texture reads global-memory
    // backing storage through the read-only texture path.
    tile[tx] = (i < n) ? tex1Dfetch<float>(input_texture, i) : 0.0f;
    __syncthreads();

    // Reuse a value loaded by a neighbor in the same block. This is safe only
    // after the block-wide barrier above.
    if (i < n) {
        const int neighbor = (tx + 1) % blockDim.x;
        output[i] = c_scale * (tile[tx] + tile[neighbor]);
    }
}

int main() {
    constexpr int n = 16;
    constexpr int threads = 8;

    std::vector<float> input(n);
    std::vector<float> output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    const float scale = 10.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_scale, &scale, sizeof(scale)));

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeLinear;
    resource.res.linear.devPtr = d_input;
    resource.res.linear.desc = cudaCreateChannelDesc<float>();
    resource.res.linear.sizeInBytes = n * sizeof(float);

    cudaTextureDesc texture{};
    texture.readMode = cudaReadModeElementType;

    cudaTextureObject_t input_texture = 0;
    CUDA_CHECK(cudaCreateTextureObject(&input_texture, &resource, &texture,
                                       nullptr));

    const int blocks = (n + threads - 1) / threads;
    const size_t dynamic_shared_bytes = threads * sizeof(float);
    memory_spaces<<<blocks, threads, dynamic_shared_bytes>>>(input_texture,
                                                             d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "output = scale * (my value + next value in my block)\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "thread " << i << ": " << output[i] << '\n';
    }

    CUDA_CHECK(cudaDestroyTextureObject(input_texture));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input));
}
