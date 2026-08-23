#include <cuda_runtime.h>

#include <NXB/BVHBuilder.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NEXUSBVH_REVISION
#define NEXUSBVH_REVISION "unknown"
#endif

namespace {

constexpr int kWarmupBuilds = 1;
constexpr int kMeasuredBuilds = 11;
constexpr int kMeasuredTraces = 11;
constexpr int kRayWidth = 1024;
constexpr int kRayHeight = 576;
constexpr std::uint32_t kRayCount = kRayWidth * kRayHeight;
constexpr float kFovScale = 0.2f;
constexpr int kTraversalStackSize = 32;
constexpr int kBlockSize = 128;

struct PackedMesh {
  std::vector<NXB::Triangle> triangles;
  float3 lower;
  float3 upper;
};

struct Camera {
  float3 origin;
  float3 forward;
  float3 right;
  float3 up;
  float fov_scale;
};

struct HitRecord {
  float u;
  float v;
  std::uint32_t primitive;
  std::uint32_t instance;
  float nx;
  float ny;
  float nz;
  float t;
};

static_assert(sizeof(HitRecord) == 32);

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

std::uint32_t parse_obj_vertex_index(const std::string& token,
                                     std::size_t vertex_count) {
  const std::size_t slash = token.find('/');
  const int index = std::stoi(token.substr(0, slash));
  const std::int64_t resolved =
      index > 0 ? static_cast<std::int64_t>(index - 1)
                : static_cast<std::int64_t>(vertex_count) + index;

  if (resolved < 0 || resolved >= static_cast<std::int64_t>(vertex_count)) {
    throw std::runtime_error("OBJ face contains an invalid vertex index");
  }

  return static_cast<std::uint32_t>(resolved);
}

PackedMesh load_obj_triangles(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Could not open OBJ: " + path);
  }

  std::vector<float3> vertices;
  std::vector<NXB::Triangle> triangles;
  std::string line;
  float3 lower = make_float3(std::numeric_limits<float>::max());
  float3 upper = make_float3(-std::numeric_limits<float>::max());

  while (std::getline(input, line)) {
    if (line.compare(0, 2, "v ") == 0) {
      std::istringstream fields(line.substr(2));
      float x, y, z;
      if (!(fields >> x >> y >> z)) {
        throw std::runtime_error("Could not parse OBJ vertex");
      }
      const float3 vertex = make_float3(x, y, z);
      vertices.push_back(vertex);
      continue;
    }

    if (line.compare(0, 2, "f ") != 0) {
      continue;
    }

    std::istringstream fields(line.substr(2));
    std::vector<std::uint32_t> face;
    std::string token;
    while (fields >> token) {
      face.push_back(parse_obj_vertex_index(token, vertices.size()));
    }

    for (std::size_t i = 1; i + 1 < face.size(); ++i) {
      triangles.emplace_back(
          vertices[face[0]], vertices[face[i]], vertices[face[i + 1]]);
    }
  }

  if (triangles.empty()) {
    throw std::runtime_error("OBJ contains no triangles: " + path);
  }
  if (triangles.size() > UINT32_MAX) {
    throw std::runtime_error("NexusBVH only supports uint32 triangle counts");
  }

  for (const NXB::Triangle& triangle : triangles) {
    for (const float3 vertex : {triangle.v0, triangle.v1, triangle.v2}) {
      lower.x = std::min(lower.x, vertex.x);
      lower.y = std::min(lower.y, vertex.y);
      lower.z = std::min(lower.z, vertex.z);
      upper.x = std::max(upper.x, vertex.x);
      upper.y = std::max(upper.y, vertex.y);
      upper.z = std::max(upper.z, vertex.z);
    }
  }

  return {std::move(triangles), lower, upper};
}

Camera make_camera(float3 lower, float3 upper) {
  const float3 center = (lower + upper) * 0.5f;
  const float3 extent = upper - lower;
  const float scene_width =
      std::max(1.0f, std::max(extent.x, std::max(extent.y, extent.z)));
  const float3 origin =
      center + make_float3(0.0f, extent.y * 0.2f, -scene_width * 2.5f);
  const float3 forward = normalize(center - origin);
  const float3 right = normalize(cross(forward, make_float3(0, 1, 0)));
  const float3 up = normalize(cross(right, forward));
  return {origin, forward, right, up, kFovScale};
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[(values.size() - 1) / 2];
}

__device__ __forceinline__ std::uint32_t extract_byte(std::uint32_t value,
                                                      std::uint32_t index) {
  return (value >> (index * 8)) & 0xff;
}

// Packed operations used by the official Nexus renderer CWBVH8 traversal.
__device__ __forceinline__ std::uint32_t sign_extend_s8x4(
    std::uint32_t value) {
  std::uint32_t result;
  asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;"
      : "=r"(result)
      : "r"(value));
  return result;
}

__device__ __forceinline__ std::uint32_t octant(const float3 direction) {
  return ((direction.x < 0 ? 1u : 0u) << 2) |
         ((direction.y < 0 ? 1u : 0u) << 1) |
         (direction.z < 0 ? 1u : 0u);
}

__device__ __forceinline__ void trace_children(
    const NXB::BVH8::Node* nodes,
    std::uint32_t node_index,
    float3 origin,
    float3 direction,
    float3 inverse_direction,
    std::uint32_t inverse_octant4,
    float hit_distance,
    uint2& internal_entry,
    uint2& triangle_entry) {
  const float4 p_e_imask = __ldg(&nodes[node_index].p_e_imask);
  const float4 childidx_tridx_meta =
      __ldg(&nodes[node_index].childidx_tridx_meta);
  const float4 qlox_qloy = __ldg(&nodes[node_index].qlox_qloy);
  const float4 qloz_qhix = __ldg(&nodes[node_index].qloz_qhix);
  const float4 qhiy_qhiz = __ldg(&nodes[node_index].qhiy_qhiz);

  const float3 p = make_float3(p_e_imask);
  const std::uint32_t e_imask = __float_as_uint(p_e_imask.w);
  const float3 transformed_direction = make_float3(
      __uint_as_float(extract_byte(e_imask, 0) << 23) * inverse_direction.x,
      __uint_as_float(extract_byte(e_imask, 1) << 23) * inverse_direction.y,
      __uint_as_float(extract_byte(e_imask, 2) << 23) * inverse_direction.z);
  const float3 transformed_origin = (p - origin) * inverse_direction;
  std::uint32_t hit_mask = 0;

#pragma unroll
  for (int group = 0; group < 2; ++group) {
    const std::uint32_t meta4 = __float_as_uint(
        group == 0 ? childidx_tridx_meta.z : childidx_tridx_meta.w);
    const std::uint32_t is_inner4 =
        (meta4 & (meta4 << 1)) & 0x10101010;
    const std::uint32_t inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
    const std::uint32_t bit_index4 =
        (meta4 ^ (inverse_octant4 & inner_mask4)) & 0x1f1f1f1f;
    const std::uint32_t child_bits4 = (meta4 >> 5) & 0x07070707;

    const std::uint32_t qlox = __float_as_uint(
        group == 0 ? qlox_qloy.x : qlox_qloy.y);
    const std::uint32_t qhix = __float_as_uint(
        group == 0 ? qloz_qhix.z : qloz_qhix.w);
    const std::uint32_t qloy = __float_as_uint(
        group == 0 ? qlox_qloy.z : qlox_qloy.w);
    const std::uint32_t qhiy = __float_as_uint(
        group == 0 ? qhiy_qhiz.x : qhiy_qhiz.y);
    const std::uint32_t qloz = __float_as_uint(
        group == 0 ? qloz_qhix.x : qloz_qhix.y);
    const std::uint32_t qhiz = __float_as_uint(
        group == 0 ? qhiy_qhiz.z : qhiy_qhiz.w);

    const std::uint32_t x_min = direction.x < 0.0f ? qhix : qlox;
    const std::uint32_t x_max = direction.x < 0.0f ? qlox : qhix;
    const std::uint32_t y_min = direction.y < 0.0f ? qhiy : qloy;
    const std::uint32_t y_max = direction.y < 0.0f ? qloy : qhiy;
    const std::uint32_t z_min = direction.z < 0.0f ? qhiz : qloz;
    const std::uint32_t z_max = direction.z < 0.0f ? qloz : qhiz;

#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      float3 t_min3 = make_float3(
          float(extract_byte(x_min, lane)),
          float(extract_byte(y_min, lane)),
          float(extract_byte(z_min, lane)));
      float3 t_max3 = make_float3(
          float(extract_byte(x_max, lane)),
          float(extract_byte(y_max, lane)),
          float(extract_byte(z_max, lane)));
      t_min3 = t_min3 * transformed_direction + transformed_origin;
      t_max3 = t_max3 * transformed_direction + transformed_origin;
      const float t_min = fmaxf(
          fmaxf(t_min3.x, t_min3.y), fmaxf(t_min3.z, 0.0f));
      const float t_max = fminf(
          fminf(t_max3.x, t_max3.y), fminf(t_max3.z, hit_distance));
      if (t_min <= t_max) {
        hit_mask |= extract_byte(child_bits4, lane)
                    << extract_byte(bit_index4, lane);
      }
    }
  }

  internal_entry.x = __float_as_uint(childidx_tridx_meta.x);
  internal_entry.y =
      (hit_mask & 0xff000000) | extract_byte(e_imask, 3);
  triangle_entry.x = __float_as_uint(childidx_tridx_meta.y);
  triangle_entry.y = hit_mask & 0x00ffffff;
}

__device__ __forceinline__ void trace_triangle(
    const NXB::Triangle& triangle,
    float3 origin,
    float3 direction,
    std::uint32_t primitive,
    HitRecord& hit) {
  const float3 edge1 = triangle.v1 - triangle.v0;
  const float3 edge2 = triangle.v2 - triangle.v0;
  const float3 p = cross(direction, edge2);
  const float determinant = dot(edge1, p);
  if (fabsf(determinant) <= 1.0e-8f) {
    return;
  }
  const float inverse_determinant = 1.0f / determinant;
  const float3 translated = origin - triangle.v0;
  const float u = dot(translated, p) * inverse_determinant;
  if (u < 0.0f || u > 1.0f) {
    return;
  }
  const float3 q = cross(translated, edge1);
  const float v = dot(direction, q) * inverse_determinant;
  if (v < 0.0f || u + v > 1.0f) {
    return;
  }
  const float t = dot(edge2, q) * inverse_determinant;
  if (t <= 0.0f || t >= hit.t) {
    return;
  }

  const float3 normal = normalize(cross(edge1, edge2));
  hit = {u, v, primitive, UINT32_MAX, normal.x, normal.y, normal.z, t};
}

__global__ void trace_camera_kernel(
    NXB::BVH8 bvh,
    const NXB::Triangle* triangles,
    Camera camera,
    HitRecord* hits,
    std::uint32_t ray_count,
    int ray_width,
    int ray_height) {
  const std::uint32_t ray_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray_index >= ray_count) {
    return;
  }

  const int pixel_x = ray_index % ray_width;
  const int pixel_y = ray_index / ray_width;
  const float inverse_height = 1.0f / float(ray_height);
  const float screen_x =
      (2.0f * (float(pixel_x) + 0.5f) - float(ray_width)) * inverse_height;
  const float screen_y =
      1.0f - 2.0f * (float(pixel_y) + 0.5f) * inverse_height;
  const float3 direction = normalize(
      camera.forward + camera.right * (screen_x * camera.fov_scale) +
      camera.up * (screen_y * camera.fov_scale));
  const float3 inverse_direction = 1.0f / direction;
  const std::uint32_t inverse_octant = 7 - octant(direction);
  const std::uint32_t inverse_octant4 =
      inverse_octant * 0x01010101;

  HitRecord hit = {
      0.0f,
      0.0f,
      UINT32_MAX,
      UINT32_MAX,
      0.0f,
      0.0f,
      0.0f,
      FLT_MAX,
  };
  uint2 stack[kTraversalStackSize];
  int stack_size = 0;
  uint2 node_entry = make_uint2(0, 0x80000000);
  uint2 triangle_entry = make_uint2(0, 0);

  while (true) {
    if (node_entry.y & 0xff000000) {
      const int node_offset = 31 - __clz(node_entry.y);
      node_entry.y &= ~(1u << node_offset);
      if (node_entry.y & 0xff000000) {
        if (stack_size >= kTraversalStackSize) {
          break;
        }
        stack[stack_size++] = node_entry;
      }
      const int node_slot = (node_offset - 24) ^ inverse_octant;
      const int relative_node =
          __popc(node_entry.y & ~(0xffffffffu << node_slot));
      trace_children(
          bvh.nodes,
          node_entry.x + relative_node,
          camera.origin,
          direction,
          inverse_direction,
          inverse_octant4,
          hit.t,
          node_entry,
          triangle_entry);
    } else {
      triangle_entry = node_entry;
      node_entry = make_uint2(0, 0);
    }

    while (triangle_entry.y) {
      const int triangle_offset = 31 - __clz(triangle_entry.y);
      triangle_entry.y &= ~(1u << triangle_offset);
      const std::uint32_t primitive =
          bvh.primIdx[triangle_entry.x + triangle_offset];
      trace_triangle(
          triangles[primitive], camera.origin, direction, primitive, hit);
    }

    if ((node_entry.y & 0xff000000) == 0) {
      if (stack_size == 0) {
        break;
      }
      node_entry = stack[--stack_size];
    }
  }

  hits[ray_index] = hit;
}

struct TraceResult {
  double median_ms;
  double minimum_ms;
  double maximum_ms;
  std::uint64_t hits;
  double checksum;
};

TraceResult benchmark_traversal(
    const NXB::BVH8& bvh,
    const NXB::Triangle* device_triangles,
    Camera camera) {
  HitRecord* device_hits = nullptr;
  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&device_hits),
                 kRayCount * sizeof(HitRecord)),
      "cudaMalloc hits");
  const int blocks = (kRayCount + kBlockSize - 1) / kBlockSize;

  trace_camera_kernel<<<blocks, kBlockSize>>>(
      bvh,
      device_triangles,
      camera,
      device_hits,
      kRayCount,
      kRayWidth,
      kRayHeight);
  cuda_check(cudaGetLastError(), "launch warm traversal");
  cuda_check(cudaDeviceSynchronize(), "warm traversal");

  std::vector<double> timings;
  timings.reserve(kMeasuredTraces);
  for (int i = 0; i < kMeasuredTraces; ++i) {
    const auto start = std::chrono::steady_clock::now();
    trace_camera_kernel<<<blocks, kBlockSize>>>(
        bvh,
        device_triangles,
        camera,
        device_hits,
        kRayCount,
        kRayWidth,
        kRayHeight);
    cuda_check(cudaGetLastError(), "launch traversal");
    cuda_check(cudaDeviceSynchronize(), "traversal");
    const auto stop = std::chrono::steady_clock::now();
    timings.push_back(
        std::chrono::duration<double, std::milli>(stop - start).count());
  }

  std::vector<HitRecord> hits(kRayCount);
  cuda_check(
      cudaMemcpy(hits.data(), device_hits, hits.size() * sizeof(HitRecord),
                 cudaMemcpyDeviceToHost),
      "cudaMemcpy hits");
  cuda_check(cudaFree(device_hits), "cudaFree hits");

  std::uint64_t hit_count = 0;
  double checksum = 0.0;
  for (const HitRecord& hit : hits) {
    if (hit.t < std::numeric_limits<float>::max()) {
      ++hit_count;
      checksum += hit.t;
    }
  }
  const auto [minimum, maximum] =
      std::minmax_element(timings.begin(), timings.end());
  return {median(timings), *minimum, *maximum, hit_count, checksum};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("usage: bajo_bench_nexusbvh <mesh.obj>");
    }

    const PackedMesh mesh = load_obj_triangles(argv[1]);
    const std::uint32_t triangle_count =
        static_cast<std::uint32_t>(mesh.triangles.size());
    NXB::Triangle* device_triangles = nullptr;
    cuda_check(
        cudaMalloc(reinterpret_cast<void**>(&device_triangles),
                   mesh.triangles.size() * sizeof(NXB::Triangle)),
        "cudaMalloc triangles");
    cuda_check(
        cudaMemcpy(device_triangles, mesh.triangles.data(),
                   mesh.triangles.size() * sizeof(NXB::Triangle),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy triangles");

    NXB::BuildConfig config;
    config.prioritizeSpeed = true;
    for (int i = 0; i < kWarmupBuilds; ++i) {
      NXB::BVH8 bvh =
          NXB::BuildBVH8(device_triangles, triangle_count, config, nullptr);
      NXB::FreeDeviceBVH(bvh);
    }

    std::vector<double> timings;
    timings.reserve(kMeasuredBuilds);
    for (int i = 0; i < kMeasuredBuilds; ++i) {
      const auto start = std::chrono::steady_clock::now();
      NXB::BVH8 bvh =
          NXB::BuildBVH8(device_triangles, triangle_count, config, nullptr);
      cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
      const auto stop = std::chrono::steady_clock::now();
      timings.push_back(
          std::chrono::duration<double, std::milli>(stop - start).count());
      NXB::FreeDeviceBVH(bvh);
    }

    NXB::BVH8 traversal_bvh =
        NXB::BuildBVH8(device_triangles, triangle_count, config, nullptr);
    const TraceResult trace = benchmark_traversal(
        traversal_bvh, device_triangles, make_camera(mesh.lower, mesh.upper));
    NXB::FreeDeviceBVH(traversal_bvh);
    cuda_check(cudaFree(device_triangles), "cudaFree triangles");

    const double build_median_ms = median(timings);
    const auto [build_minimum, build_maximum] =
        std::minmax_element(timings.begin(), timings.end());
    const double mrays = double(kRayCount) / (trace.median_ms * 1000.0);

    std::cout << "NexusBVH GPU BVH build and traversal benchmark\n"
              << "Revision: " << NEXUSBVH_REVISION << '\n'
              << "Triangles: " << triangle_count << '\n'
              << "Builder: H-PLOC, 32-bit Morton code\n"
              << "Output: CWBVH8\n"
              << "Measured builds: " << kMeasuredBuilds << '\n'
              << std::fixed << std::setprecision(3)
              << "Build median: " << build_median_ms << " ms ("
              << *build_minimum << ".." << *build_maximum << " ms)\n"
              << "Camera rays: " << kRayWidth << 'x' << kRayHeight << " ("
              << kRayCount << ")\n"
              << "Trace median: " << trace.median_ms << " ms ("
              << trace.minimum_ms << ".." << trace.maximum_ms << " ms), "
              << mrays << " MRay/s\n"
              << "Hits: " << trace.hits
              << ", t checksum: " << std::setprecision(9) << trace.checksum
              << '\n'
              << "RESULT\tnexusbvh\tNexusBVH-H-PLOC-CWBVH8\thploc\t"
                 "cwbvh8\t8\t1\t1\t"
              << triangle_count << '\t' << build_median_ms << '\t'
              << *build_minimum << '\t' << *build_maximum << '\t'
              << kRayCount << '\t' << trace.median_ms << '\t'
              << trace.minimum_ms << '\t' << trace.maximum_ms << '\t'
              << trace.hits << '\t' << trace.checksum << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
