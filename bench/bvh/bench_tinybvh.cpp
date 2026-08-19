#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if !defined(BENCH_THREADED_BUILDS)
#define NO_THREADED_BUILDS
#endif
#define NO_CUSTOM_GEOMETRY
#define NO_INDEXED_GEOMETRY
#define NO_VOXEL_SUPPORT
#define NO_DOUBLE_PRECISION_SUPPORT
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

namespace {

constexpr std::size_t kGridSide = 256;
constexpr std::size_t kPrimitiveCount = kGridSide * kGridSide;
constexpr std::size_t kRayRepeatsPerPrimitive = 4;
constexpr std::size_t kRayCount = kPrimitiveCount * kRayRepeatsPerPrimitive;
constexpr int kTraversalRepeats = 8;
constexpr std::size_t kRepresentativeRayWidth = 1024;
constexpr std::size_t kRepresentativeRayHeight = 576;
constexpr float kRepresentativeFovScale = 0.2f;

struct Vertex {
  float x, y, z;
};

struct InputRay {
  float ox, oy, oz;
  float dx, dy, dz;
};

struct Vec3 {
  float x, y, z;
};

struct PackedMesh {
  std::vector<Vertex> vertices;
};

struct TraceResult {
  double ms = 0.0;
  double checksum = 0.0;
  std::uint64_t hits = 0;
};

struct TraceSummary {
  double checksum = 0.0;
  std::uint64_t hits = 0;
};

enum class BuildQuality {
  Sah,
  High,
};

float grid_x(std::size_t i) {
  return (static_cast<float>(i % kGridSide) -
          static_cast<float>(kGridSide) * 0.5f) *
         3.0f;
}

float grid_y(std::size_t i) {
  return (static_cast<float>(i / kGridSide) -
          static_cast<float>(kGridSide) * 0.5f) *
         3.0f;
}

std::vector<Vertex> make_grid_vertices() {
  std::vector<Vertex> vertices;
  vertices.reserve(kPrimitiveCount * 3);

  for (std::size_t i = 0; i < kPrimitiveCount; ++i) {
    const float cx = grid_x(i);
    const float cy = grid_y(i);
    vertices.push_back({cx - 0.75f, cy - 0.75f, 2.0f});
    vertices.push_back({cx + 0.75f, cy - 0.75f, 2.0f});
    vertices.push_back({cx, cy + 0.75f, 2.0f});
  }

  return vertices;
}

std::vector<InputRay> make_hit_and_miss_rays() {
  std::vector<InputRay> rays;
  rays.reserve(kRayCount);

  for (std::size_t i = 0; i < kRayCount; ++i) {
    const std::size_t primitive_index = i % kPrimitiveCount;

    if (i % kRayRepeatsPerPrimitive == 0) {
      rays.push_back(
          {10000.0f + static_cast<float>(i), 10000.0f, 0.0f,
           0.0f, 0.0f, 1.0f});
    } else {
      rays.push_back({grid_x(primitive_index), grid_y(primitive_index), 0.0f,
                      0.0f, 0.0f, 1.0f});
    }
  }

  return rays;
}

Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(Vec3 value, float scale) {
  return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3 cross(Vec3 a, Vec3 b) {
  return {std::fma(a.y, b.z, -(a.z * b.y)),
          std::fma(a.z, b.x, -(a.x * b.z)),
          std::fma(a.x, b.y, -(a.y * b.x))};
}

Vec3 normalize(Vec3 value) {
  const float length_squared = std::fma(
      value.x, value.x,
      std::fma(value.y, value.y, value.z * value.z));
  const float inverse_length = 1.0f / std::sqrt(length_squared);
  return value * inverse_length;
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

PackedMesh load_packed_obj(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Could not open OBJ: " + path);
  }

  std::vector<Vertex> source_vertices;
  std::vector<std::array<std::uint32_t, 3>> source_triangles;
  std::string line;

  while (std::getline(input, line)) {
    if (line.starts_with("v ")) {
      std::istringstream fields(line.substr(2));
      Vertex vertex;
      if (!(fields >> vertex.x >> vertex.y >> vertex.z)) {
        throw std::runtime_error("Could not parse OBJ vertex");
      }
      source_vertices.push_back(vertex);
    } else if (line.starts_with("f ")) {
      std::istringstream fields(line.substr(2));
      std::vector<std::uint32_t> face;
      std::string token;

      while (fields >> token) {
        face.push_back(parse_obj_vertex_index(token, source_vertices.size()));
      }

      for (std::size_t i = 1; i + 1 < face.size(); ++i) {
        source_triangles.push_back({face[0], face[i], face[i + 1]});
      }
    }
  }

  PackedMesh mesh;
  mesh.vertices.reserve(source_triangles.size() * 3);

  for (const auto& triangle : source_triangles) {
    for (std::uint32_t index : triangle) {
      mesh.vertices.push_back(source_vertices[index]);
    }
  }

  return mesh;
}

std::vector<InputRay> make_representative_camera_rays(
    const std::vector<Vertex>& vertices) {
  Vec3 lower = {std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max()};
  Vec3 upper = {-std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max()};

  for (const Vertex& vertex : vertices) {
    lower.x = std::min(lower.x, vertex.x);
    lower.y = std::min(lower.y, vertex.y);
    lower.z = std::min(lower.z, vertex.z);
    upper.x = std::max(upper.x, vertex.x);
    upper.y = std::max(upper.y, vertex.y);
    upper.z = std::max(upper.z, vertex.z);
  }

  const Vec3 center = (lower + upper) * 0.5f;
  const Vec3 extent = upper - lower;
  const float scene_width =
      std::max(1.0f, std::max(extent.x, std::max(extent.y, extent.z)));
  const Vec3 origin =
      center + Vec3{0.0f, extent.y * 0.2f, -scene_width * 2.5f};
  const Vec3 forward = normalize(center - origin);
  const Vec3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
  const Vec3 up = normalize(cross(right, forward));
  const float aspect = static_cast<float>(kRepresentativeRayWidth) /
                       static_cast<float>(kRepresentativeRayHeight);

  std::vector<InputRay> rays;
  rays.reserve(kRepresentativeRayWidth * kRepresentativeRayHeight);

  for (std::size_t py = 0; py < kRepresentativeRayHeight; ++py) {
    for (std::size_t px = 0; px < kRepresentativeRayWidth; ++px) {
      const float sx =
          ((static_cast<float>(px) + 0.5f) /
               static_cast<float>(kRepresentativeRayWidth)) *
              2.0f -
          1.0f;
      const float sy =
          1.0f - ((static_cast<float>(py) + 0.5f) /
                      static_cast<float>(kRepresentativeRayHeight)) *
                     2.0f;
      const Vec3 direction = normalize(
          forward + right * (sx * aspect * kRepresentativeFovScale) +
          up * (sy * kRepresentativeFovScale));

      rays.push_back(
          {origin.x, origin.y, origin.z,
           direction.x, direction.y, direction.z});
    }
  }

  return rays;
}

std::vector<tinybvh::bvhvec4> pack_vertices(
    const std::vector<Vertex>& vertices) {
  std::vector<tinybvh::bvhvec4> packed;
  packed.reserve(vertices.size());

  for (const Vertex& vertex : vertices) {
    packed.emplace_back(vertex.x, vertex.y, vertex.z, 0.0f);
  }

  return packed;
}

// TinyBVH's optimized wide traversers evaluate bounds as
// bound * rD - origin * rD. FLT_MAX can overflow for exactly-zero
// direction components, so use BVH_FAR as the finite infinity value.
float benchmark_safe_rcp(float direction) {
  if (direction == 0.0f) {
    return std::copysign(BVH_FAR, direction);
  }

  const float reciprocal = 1.0f / direction;
  return std::clamp(reciprocal, -BVH_FAR, BVH_FAR);
}

tinybvh::Ray make_tinybvh_ray(const InputRay& input) {
  tinybvh::Ray ray{};

  ray.O = tinybvh::bvhvec3(input.ox, input.oy, input.oz);
  ray.D = tinybvh::bvhvec3(input.dx, input.dy, input.dz);

  ray.rD = tinybvh::bvhvec3(
      benchmark_safe_rcp(input.dx),
      benchmark_safe_rcp(input.dy),
      benchmark_safe_rcp(input.dz));

  ray.mask = RAY_MASK_INTERSECT_ALL;
  ray.instIdx = 0;
  ray.hit.t = BVH_FAR;

  return ray;
}

std::vector<tinybvh::Ray> make_tinybvh_rays(
    const std::vector<InputRay>& input_rays) {
  std::vector<tinybvh::Ray> rays;
  rays.reserve(input_rays.size());

  for (const InputRay& input : input_rays) {
    rays.push_back(make_tinybvh_ray(input));
  }

  return rays;
}

double hit_checksum(const tinybvh::Intersection& hit,
                    const std::vector<Vertex>& vertices) {
  const std::size_t base = static_cast<std::size_t>(hit.prim) * 3;
  if (base + 2 >= vertices.size()) {
    throw std::runtime_error("TinyBVH returned an invalid primitive index");
  }

  const Vertex& a = vertices[base];
  const Vertex& b = vertices[base + 1];
  const Vertex& c = vertices[base + 2];
  const Vec3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
  const Vec3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
  const Vec3 normal = normalize(cross(e1, e2));

  return static_cast<double>(hit.t) +
         static_cast<double>(hit.u) +
         static_cast<double>(hit.v) +
         static_cast<double>(normal.x) +
         static_cast<double>(normal.y) +
         static_cast<double>(normal.z) +
         static_cast<double>(hit.prim);
}

template <typename BVHType>
TraceSummary trace_once(const BVHType& bvh,
                        std::vector<tinybvh::Ray>& rays,
                        const std::vector<Vertex>& vertices) {
  TraceSummary result;

  for (tinybvh::Ray& ray : rays) {
    ray.hit.t = BVH_FAR;
    bvh.Intersect(ray);

    if (ray.hit.t < BVH_FAR) {
      result.checksum += hit_checksum(ray.hit, vertices);
      ++result.hits;
    }
  }

  return result;
}

template <typename TraceFunction>
TraceResult benchmark_trace(TraceFunction&& trace) {
  TraceSummary summary = trace();
  double best_ms = std::numeric_limits<double>::max();

  for (int repeat = 0; repeat < kTraversalRepeats; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    summary = trace();
    const auto stop = std::chrono::steady_clock::now();

    best_ms = std::min(
        best_ms,
        std::chrono::duration<double, std::milli>(stop - start).count());
  }

  return {best_ms, summary.checksum, summary.hits};
}

void print_result(std::string_view quality,
                  std::string_view layout,
                  double build_ms,
                  std::size_t ray_count,
                  const TraceResult& result) {
  const double mrays_per_second =
      static_cast<double>(ray_count) / (result.ms * 1000.0);

  std::cout << std::left << std::setw(9) << quality
            << std::setw(10) << layout
            << std::right << std::fixed << std::setprecision(3)
            << std::setw(11) << build_ms
            << std::setw(12) << result.ms
            << std::setw(12) << mrays_per_second
            << std::setw(11) << result.hits
            << std::setw(15) << result.checksum << '\n';
}

template <typename BVHType>
void benchmark_layout(std::string_view quality_name,
                      BuildQuality quality,
                      std::string_view layout_name,
                      const std::vector<tinybvh::bvhvec4>& packed_vertices,
                      const std::vector<Vertex>& vertices,
                      std::vector<tinybvh::Ray>& rays) {
  const std::uint32_t triangle_count =
      static_cast<std::uint32_t>(packed_vertices.size() / 3);

  BVHType bvh;
  const auto build_start = std::chrono::steady_clock::now();

  if (quality == BuildQuality::Sah) {
    bvh.Build(packed_vertices.data(), triangle_count);
  } else {
    bvh.BuildHQ(packed_vertices.data(), triangle_count);
  }

  const auto build_stop = std::chrono::steady_clock::now();
  const double build_ms =
      std::chrono::duration<double, std::milli>(build_stop - build_start)
          .count();

  const TraceResult trace = benchmark_trace(
      [&] { return trace_once(bvh, rays, vertices); });

  print_result(
      quality_name,
      layout_name,
      build_ms,
      rays.size(),
      trace);
}

void benchmark_case(std::string_view name,
                    const std::vector<Vertex>& vertices,
                    const std::vector<InputRay>& input_rays) {
  if (vertices.size() % 3 != 0) {
    throw std::runtime_error("Vertex count must be divisible by three");
  }

  const std::vector<tinybvh::bvhvec4> packed_vertices =
      pack_vertices(vertices);
  std::vector<tinybvh::Ray> rays = make_tinybvh_rays(input_rays);

  std::cout << "\n" << name << "\n"
            << "Triangles: " << vertices.size() / 3 << "\n"
            << "Rays: " << rays.size() << "\n"
            << std::left << std::setw(9) << "quality"
            << std::setw(10) << "layout"
            << std::right << std::setw(11) << "build ms"
            << std::setw(12) << "trace ms"
            << std::setw(12) << "MRay/s"
            << std::setw(11) << "hits"
            << std::setw(15) << "checksum" << '\n';

  benchmark_layout<tinybvh::BVH>(
      "sah", BuildQuality::Sah, "bvh2", packed_vertices, vertices, rays);
  benchmark_layout<tinybvh::BVH4_CPU>(
      "sah", BuildQuality::Sah, "bvh4", packed_vertices, vertices, rays);
  benchmark_layout<tinybvh::BVH8_CPU>(
      "sah", BuildQuality::Sah, "bvh8", packed_vertices, vertices, rays);

  benchmark_layout<tinybvh::BVH>(
      "high", BuildQuality::High, "bvh2", packed_vertices, vertices, rays);
  benchmark_layout<tinybvh::BVH4_CPU>(
      "high", BuildQuality::High, "bvh4", packed_vertices, vertices, rays);
  benchmark_layout<tinybvh::BVH8_CPU>(
      "high", BuildQuality::High, "bvh8", packed_vertices, vertices, rays);
}

}  // namespace

int main(int argc, char** argv) {
  try {
#if !defined(__AVX2__) || !defined(__FMA__)
    throw std::runtime_error(
        "This benchmark must be compiled with AVX2 and FMA enabled");
#endif

    const std::vector<Vertex> grid_vertices = make_grid_vertices();
    const std::vector<InputRay> grid_rays = make_hit_and_miss_rays();

    std::cout << "TinyBVH "
              << TINY_BVH_VERSION_MAJOR << "."
              << TINY_BVH_VERSION_MINOR << "."
              << TINY_BVH_VERSION_SUB
              << " CPU triangle benchmark\n"
#if defined(BENCH_THREADED_BUILDS)
              << "Build threads: all\n"
#else
              << "Build threads: 1\n"
#endif
              << "AVX2/FMA: yes\n"
              << "Traversal repeats: " << kTraversalRepeats << "\n";

    benchmark_case(
        "Regular-grid microbenchmark",
        grid_vertices,
        grid_rays);

    if (argc > 1) {
      const PackedMesh mesh = load_packed_obj(argv[1]);
      const std::vector<InputRay> representative_rays =
          make_representative_camera_rays(mesh.vertices);

      benchmark_case(
          "Representative Dragon camera-ray benchmark",
          mesh.vertices,
          representative_rays);
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
