NVCC       := nvcc
CUDA_ARCH  ?= sm_120

N_RUNS      ?= 10
WARMUP_RUNS ?= 5

COMMON_NVCCFLAGS := -std=c++17 -arch=$(CUDA_ARCH) -O3
NVCCFLAGS_CUB   := $(COMMON_NVCCFLAGS) -DN_RUNS=$(N_RUNS) -DWARMUP_RUNS=$(WARMUP_RUNS)

TARGETS := bench_cub_radix

all: $(TARGETS)

bench_cub_radix: bench_cub_radix.cu
	$(NVCC) $(NVCCFLAGS_CUB) -o $@ $<

run-radix: bench_cub_radix
	./bench_cub_radix cub_sort_benchmark_results.csv

clean:
	rm -f $(TARGETS) cub_sort_benchmark_results.csv gpu_sort_benchmark_results.csv gpu_sort_benchmark_plot.png

.PHONY: all run-radix clean