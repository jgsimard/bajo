// bench_cub_radix.cu — cub::DeviceRadixSort benchmark aligned with Bajo/Mojo GPU sort
//
// Measures:
//   CUB_Radix_Pairs : uint32 keys + uint32 values
//   CUB_Radix_Keys  : uint32 keys only
//
// Methodology matches the Mojo benchmark more closely:
//   - same sizes: 1<<10 ... 1<<28
//   - same N_RUNS default: 10
//   - deterministic uint32 key pattern: (i * 1103515245 + 12345) & 0x7fffffff
//   - device-to-device reset copy overhead is measured and subtracted
//   - CSV columns match Mojo: Algorithm,N,Time_ms,Throughput_GKs
//
// Build:
//   cd benchmarks && make bench_cub_radix
//
// Run:
//   ./bench_cub_radix
//   ./bench_cub_radix ../cub_sort_benchmark_results.csv

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifndef N_RUNS
#define N_RUNS 10
#endif

#ifndef WARMUP_RUNS
#define WARMUP_RUNS 5
#endif

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d — %s\n",                          \
                   __FILE__, __LINE__, cudaGetErrorString(err__));              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

static int grid(uint32_t n, int block = 256) {
  return static_cast<int>((n + block - 1) / block);
}

__global__ void fill_lcg_u32(uint32_t *out, uint32_t n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  out[tid] = (tid * 1103515245u + 12345u) & 0x7fffffffu;
}

__global__ void fill_iota_u32(uint32_t *out, uint32_t n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  out[tid] = tid;
}

__global__ void check_sorted_u32_kernel(
    const uint32_t *keys,
    uint32_t n,
    uint32_t *first_bad
) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid + 1 >= n) return;

  if (keys[tid] > keys[tid + 1]) {
    atomicMin(first_bad, tid);
  }
}

static void check_sorted_u32(const uint32_t *d_keys, uint32_t n) {
  uint32_t *d_first_bad = nullptr;
  uint32_t h_first_bad = 0xffffffffu;

  CUDA_CHECK(cudaMalloc(&d_first_bad, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(
      d_first_bad,
      &h_first_bad,
      sizeof(uint32_t),
      cudaMemcpyHostToDevice
  ));

  check_sorted_u32_kernel<<<grid(n), 256>>>(d_keys, n, d_first_bad);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(
      &h_first_bad,
      d_first_bad,
      sizeof(uint32_t),
      cudaMemcpyDeviceToHost
  ));

  CUDA_CHECK(cudaFree(d_first_bad));

  if (h_first_bad != 0xffffffffu) {
    std::fprintf(
        stderr,
        "VALIDITY FAILED: keys are not sorted at index %u\n",
        h_first_bad
    );
    std::exit(1);
  }
}

template <typename Fn>
static double timed_ms(Fn &&fn) {
  CUDA_CHECK(cudaDeviceSynchronize());

  auto t0 = std::chrono::steady_clock::now();
  fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();

  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double throughput_gks(uint32_t n, double ms) {
  return static_cast<double>(n) / (ms * 1.0e6);
}

struct Result {
  std::string algorithm;
  uint32_t n;
  double time_ms;
  double throughput_gks;
};

static void print_result(const Result &r) {
  std::printf(
      "  %-16s N=%10u  time=%9.4f ms  throughput=%8.3f GK/s\n",
      r.algorithm.c_str(),
      r.n,
      r.time_ms,
      r.throughput_gks
  );
}

static Result bench_sort_keys(uint32_t n) {
  uint32_t *d_seed = nullptr;
  uint32_t *d_keys_in = nullptr;
  uint32_t *d_keys_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_seed,     n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_keys_in,  n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_keys_out, n * sizeof(uint32_t)));

  fill_lcg_u32<<<grid(n), 256>>>(d_seed, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  const int n_items = static_cast<int>(n);

  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      d_temp,
      temp_bytes,
      d_keys_in,
      d_keys_out,
      n_items,
      0,
      32
  ));

  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

  for (int w = 0; w < WARMUP_RUNS; ++w) {
    CUDA_CHECK(cudaMemcpy(
        d_keys_in,
        d_seed,
        n * sizeof(uint32_t),
        cudaMemcpyDeviceToDevice
    ));

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        d_temp,
        temp_bytes,
        d_keys_in,
        d_keys_out,
        n_items,
        0,
        32
    ));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  check_sorted_u32(d_keys_out, n);

  double copy_ms = timed_ms([&]() {
    for (int r = 0; r < N_RUNS; ++r) {
      CUDA_CHECK(cudaMemcpy(
          d_keys_in,
          d_seed,
          n * sizeof(uint32_t),
          cudaMemcpyDeviceToDevice
      ));
    }
  });

  double total_ms = timed_ms([&]() {
    for (int r = 0; r < N_RUNS; ++r) {
      CUDA_CHECK(cudaMemcpy(
          d_keys_in,
          d_seed,
          n * sizeof(uint32_t),
          cudaMemcpyDeviceToDevice
      ));

      CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
          d_temp,
          temp_bytes,
          d_keys_in,
          d_keys_out,
          n_items,
          0,
          32
      ));
    }
  });

  double sort_ms = std::max(1.0e-9, (total_ms - copy_ms) / double(N_RUNS));

  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_keys_out));
  CUDA_CHECK(cudaFree(d_keys_in));
  CUDA_CHECK(cudaFree(d_seed));

  return Result{
      "CUB_Radix_Keys",
      n,
      sort_ms,
      throughput_gks(n, sort_ms)
  };
}

static Result bench_sort_pairs(uint32_t n) {
  uint32_t *d_seed_k = nullptr;
  uint32_t *d_keys_in = nullptr;
  uint32_t *d_keys_out = nullptr;

  uint32_t *d_seed_v = nullptr;
  uint32_t *d_vals_in = nullptr;
  uint32_t *d_vals_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_seed_k,   n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_keys_in,  n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_keys_out, n * sizeof(uint32_t)));

  CUDA_CHECK(cudaMalloc(&d_seed_v,   n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_vals_in,  n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_vals_out, n * sizeof(uint32_t)));

  fill_lcg_u32 <<<grid(n), 256>>>(d_seed_k, n);
  fill_iota_u32<<<grid(n), 256>>>(d_seed_v, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  const int n_items = static_cast<int>(n);

  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      d_temp,
      temp_bytes,
      d_keys_in,
      d_keys_out,
      d_vals_in,
      d_vals_out,
      n_items,
      0,
      32
  ));

  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

  for (int w = 0; w < WARMUP_RUNS; ++w) {
    CUDA_CHECK(cudaMemcpy(
        d_keys_in,
        d_seed_k,
        n * sizeof(uint32_t),
        cudaMemcpyDeviceToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_vals_in,
        d_seed_v,
        n * sizeof(uint32_t),
        cudaMemcpyDeviceToDevice
    ));

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        d_temp,
        temp_bytes,
        d_keys_in,
        d_keys_out,
        d_vals_in,
        d_vals_out,
        n_items,
        0,
        32
    ));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  check_sorted_u32(d_keys_out, n);

  double copy_ms = timed_ms([&]() {
    for (int r = 0; r < N_RUNS; ++r) {
      CUDA_CHECK(cudaMemcpy(
          d_keys_in,
          d_seed_k,
          n * sizeof(uint32_t),
          cudaMemcpyDeviceToDevice
      ));
      CUDA_CHECK(cudaMemcpy(
          d_vals_in,
          d_seed_v,
          n * sizeof(uint32_t),
          cudaMemcpyDeviceToDevice
      ));
    }
  });

  double total_ms = timed_ms([&]() {
    for (int r = 0; r < N_RUNS; ++r) {
      CUDA_CHECK(cudaMemcpy(
          d_keys_in,
          d_seed_k,
          n * sizeof(uint32_t),
          cudaMemcpyDeviceToDevice
      ));
      CUDA_CHECK(cudaMemcpy(
          d_vals_in,
          d_seed_v,
          n * sizeof(uint32_t),
          cudaMemcpyDeviceToDevice
      ));

      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
          d_temp,
          temp_bytes,
          d_keys_in,
          d_keys_out,
          d_vals_in,
          d_vals_out,
          n_items,
          0,
          32
      ));
    }
  });

  double sort_ms = std::max(1.0e-9, (total_ms - copy_ms) / double(N_RUNS));

  CUDA_CHECK(cudaFree(d_temp));

  CUDA_CHECK(cudaFree(d_vals_out));
  CUDA_CHECK(cudaFree(d_vals_in));
  CUDA_CHECK(cudaFree(d_seed_v));

  CUDA_CHECK(cudaFree(d_keys_out));
  CUDA_CHECK(cudaFree(d_keys_in));
  CUDA_CHECK(cudaFree(d_seed_k));

  return Result{
      "CUB_Radix_Pairs",
      n,
      sort_ms,
      throughput_gks(n, sort_ms)
  };
}

static void save_csv(const char *path, const std::vector<Result> &results) {
  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "Failed to open CSV output: %s\n", path);
    std::exit(1);
  }

  out << "Algorithm,N,Time_ms,Throughput_GKs\n";
  for (const Result &r : results) {
    out << r.algorithm << ','
        << r.n << ','
        << r.time_ms << ','
        << r.throughput_gks << '\n';
  }
}

int main(int argc, char **argv) {
  const char *csv_path =
      argc >= 2 ? argv[1] : "cub_sort_benchmark_results.csv";

  std::printf("================================================================\n");
  std::printf("CUB DeviceRadixSort Benchmark — Bajo/Mojo comparison mode\n");
  std::printf("================================================================\n");
  std::printf("N_RUNS      : %d\n", N_RUNS);
  std::printf("WARMUP_RUNS : %d\n", WARMUP_RUNS);
  std::printf("key type    : uint32_t\n");
  std::printf("value type  : uint32_t\n");
  std::printf("CSV output  : %s\n", csv_path);

  const uint32_t sizes[] = {
      1u << 10,
      1u << 12,
      1u << 14,
      1u << 16,
      1u << 18,
      1u << 20,
      1u << 22,
      1u << 24,
      1u << 26,
      1u << 28,
  };

  std::vector<Result> results;
  results.reserve(20);

  std::printf("\n--- SORT PAIRS: u32 keys + u32 values ---\n");
  for (uint32_t n : sizes) {
    Result r = bench_sort_pairs(n);
    print_result(r);
    results.push_back(r);
  }

  std::printf("\n--- SORT KEYS: u32 keys only ---\n");
  for (uint32_t n : sizes) {
    Result r = bench_sort_keys(n);
    print_result(r);
    results.push_back(r);
  }

  save_csv(csv_path, results);

  std::printf("\n================================================================\n");
  std::printf("Saved CSV: %s\n", csv_path);
  std::printf("Done.\n");

  return 0;
}