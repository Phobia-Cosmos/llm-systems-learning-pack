NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17

.PHONY: all run clean

all: memory_spaces tiled_matmul

memory_spaces: memory_spaces.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

tiled_matmul: tiled_matmul.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	./memory_spaces
	./tiled_matmul

clean:
	rm -f memory_spaces tiled_matmul
