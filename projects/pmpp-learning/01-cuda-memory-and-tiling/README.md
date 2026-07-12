# 01：CUDA memory、tiling 与矩阵坐标

## 先运行坐标实验

这段代码只需要 Python：

```bash
cd /home/undefined/Desktop/ai/projects/pmpp-learning/01-cuda-memory-and-tiling
python3 index_mapping.py
```

它展示如下恒等关系：

```text
CUDA:  x = blockIdx.x * TILE_WIDTH + threadIdx.x  -> column
CUDA:  y = blockIdx.y * TILE_WIDTH + threadIdx.y  -> row

常见矩阵记号: P[row][col] = P[y][x]
本书图中记号: Pd[x,y]
```

所以书中 `thread(1,0)`、`block(0,1)`、`TILE_WIDTH=2` 得到 `(x,y)=(1,2)`：

- 书中写成 `Pd[1,2]`；
- 常见 row-first 矩阵记号写成 `P[2][1]`；
- row-major 一维地址为 `2 * width + 1`。

## 编译并运行 CUDA 实验

本机有 `nvcc` 时，最方便的方式是：

```bash
make
make run
```

也可分别编译：

```bash
nvcc -O2 -std=c++17 memory_spaces.cu -o memory_spaces
./memory_spaces

nvcc -O2 -std=c++17 tiled_matmul.cu -o tiled_matmul
./tiled_matmul
```

清理编译结果：

```bash
make clean
```

## 看代码时关注什么

### `memory_spaces.cu`

- `__constant__ float c_scale`：host 写入，kernel 只读。
- `cudaTextureObject_t` 与 `tex1Dfetch`：现代 texture object API。
- `extern __shared__ float tile[]`：数组长度不写在 kernel 源码中。
- kernel 启动的第三个参数 `threads * sizeof(float)`：在运行时指定每个 block 的动态 shared memory 字节数。
- 每个 thread 从 texture/global backing storage 加载一个值到自己的 `tile[tx]`，`__syncthreads()` 后，同一 block 的 thread 可读取彼此加载的值。

### `tiled_matmul.cu`

- `row` 来自 `.y`，`col` 来自 `.x`。
- 每个 thread 每个 phase 各加载一个 `A` 元素和一个 `B` 元素。
- 第一次 `__syncthreads()` 确保 tile 已全部装好。
- 第二次 `__syncthreads()` 确保所有 thread 用完 tile 后才覆盖它。
- shared memory 由两个动态数组共用一块存储，启动时分配 `2 * TILE * TILE * sizeof(float)` 字节。

本例故意采用常见矩阵记号 `P[row][col]`，便于与日常线性代数代码对照。
