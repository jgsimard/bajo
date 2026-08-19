#include <embree4/rtcore.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

struct Triangle {
  std::uint32_t v0, v1, v2;
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
  std::vector<Triangle> triangles;
};

struct Scene {
  RTCScene handle = nullptr;
  double build_ms = 0.0;
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

std::string requested_build_threads() {
  const char* environment = std::getenv("BVH_BUILD_THREADS");
  const std::string value = environment == nullptr ? "1" : environment;
  const bool is_positive_integer =
      !value.empty() && value != "0" &&
      std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isdigit(character);
      });
  if (!is_positive_integer) {
    throw std::runtime_error("BVH_BUILD_THREADS must be a positive integer");
  }
  return value;
}

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

std::vector<Triangle> make_triangles() {
  std::vector<Triangle> triangles(kPrimitiveCount);
  for (std::size_t i = 0; i < kPrimitiveCount; ++i) {
    const auto base = static_cast<std::uint32_t>(i * 3);
    triangles[i] = {base, base + 1, base + 2};
  }
  return triangles;
}

std::vector<InputRay> make_hit_and_miss_rays() {
  std::vector<InputRay> rays;
  rays.reserve(kRayCount);
  for (std::size_t i = 0; i < kRayCount; ++i) {
    const std::size_t primitive_index = i % kPrimitiveCount;
    if (i % 4 == 0) {
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
  mesh.triangles.resize(source_triangles.size());
  for (std::size_t i = 0; i < source_triangles.size(); ++i) {
    const std::uint32_t base = static_cast<std::uint32_t>(i * 3);
    for (std::uint32_t index : source_triangles[i]) {
      mesh.vertices.push_back(source_vertices[index]);
    }
    mesh.triangles[i] = {base, base + 1, base + 2};
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
          {origin.x, origin.y, origin.z, direction.x, direction.y, direction.z});
    }
  }
  return rays;
}

Scene build_scene(RTCDevice device, RTCBuildQuality quality,
                  const std::vector<Vertex>& vertices,
                  const std::vector<Triangle>& triangles) {
  const auto start = std::chrono::steady_clock::now();

  RTCScene scene = rtcNewScene(device);
  rtcSetSceneBuildQuality(scene, quality);

  RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
  rtcSetGeometryBuildQuality(geometry, quality);

  void* vertex_buffer = rtcSetNewGeometryBuffer(
      geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex),
      vertices.size());
  void* index_buffer = rtcSetNewGeometryBuffer(
      geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle),
      triangles.size());
  std::memcpy(vertex_buffer, vertices.data(), vertices.size() * sizeof(Vertex));
  std::memcpy(index_buffer, triangles.data(),
              triangles.size() * sizeof(Triangle));

  rtcCommitGeometry(geometry);
  rtcAttachGeometry(scene, geometry);
  rtcReleaseGeometry(geometry);
  rtcCommitScene(scene);

  const auto stop = std::chrono::steady_clock::now();
  const double build_ms =
      std::chrono::duration<double, std::milli>(stop - start).count();

  if (rtcGetDeviceError(device) != RTC_ERROR_NONE) {
    rtcReleaseScene(scene);
    throw std::runtime_error("Embree failed while building the scene");
  }
  return {scene, build_ms};
}

RTCRayHit make_ray_hit(const InputRay& input) {
  RTCRayHit ray_hit;
  ray_hit.ray.org_x = input.ox;
  ray_hit.ray.org_y = input.oy;
  ray_hit.ray.org_z = input.oz;
  ray_hit.ray.tnear = 0.0f;
  ray_hit.ray.dir_x = input.dx;
  ray_hit.ray.dir_y = input.dy;
  ray_hit.ray.dir_z = input.dz;
  ray_hit.ray.time = 0.0f;
  ray_hit.ray.tfar = std::numeric_limits<float>::max();
  ray_hit.ray.mask = 0xffffffffu;
  ray_hit.ray.id = 0;
  ray_hit.ray.flags = 0;
  ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
  for (unsigned level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT; ++level) {
    ray_hit.hit.instID[level] = RTC_INVALID_GEOMETRY_ID;
  }
  return ray_hit;
}

double hit_checksum(float t, float u, float v, float ng_x, float ng_y,
                    float ng_z, std::uint32_t primitive_id) {
  const float length_squared =
      std::fma(ng_x, ng_x, std::fma(ng_y, ng_y, ng_z * ng_z));
  const float length = std::sqrt(length_squared);
  const float inverse_length = length > 1.0e-20f ? 1.0f / length : 0.0f;
  return static_cast<double>(t) + static_cast<double>(u) +
         static_cast<double>(v) + static_cast<double>(ng_x * inverse_length) +
         static_cast<double>(ng_y * inverse_length) +
         static_cast<double>(ng_z * inverse_length) +
         static_cast<double>(primitive_id);
}

TraceSummary trace_scalar_once(RTCScene scene,
                               const std::vector<InputRay>& rays) {
  RTCIntersectArguments arguments;
  rtcInitIntersectArguments(&arguments);

  TraceSummary result;
  for (const InputRay& input : rays) {
    RTCRayHit ray_hit = make_ray_hit(input);
    rtcIntersect1(scene, &ray_hit, &arguments);
    if (ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
      result.checksum +=
          hit_checksum(ray_hit.ray.tfar, ray_hit.hit.u, ray_hit.hit.v,
                       ray_hit.hit.Ng_x, ray_hit.hit.Ng_y, ray_hit.hit.Ng_z,
                       ray_hit.hit.primID);
      ++result.hits;
    }
  }
  return result;
}

void init_packet_arguments(RTCIntersectArguments& arguments, bool coherent) {
  rtcInitIntersectArguments(&arguments);
  arguments.flags = coherent ? RTC_RAY_QUERY_FLAG_COHERENT
                             : RTC_RAY_QUERY_FLAG_INCOHERENT;
}

TraceSummary trace_packet4_once(RTCScene scene,
                                const std::vector<InputRay>& rays,
                                bool coherent) {
  RTCIntersectArguments arguments;
  init_packet_arguments(arguments, coherent);

  alignas(16) int valid[4] = {-1, -1, -1, -1};

  TraceSummary result;
  for (std::size_t base = 0; base < rays.size(); base += 4) {
    alignas(16) RTCRayHit4 packet;

    for (std::size_t lane = 0; lane < 4; ++lane) {
      const InputRay& input = rays[base + lane];

      packet.ray.org_x[lane] = input.ox;
      packet.ray.org_y[lane] = input.oy;
      packet.ray.org_z[lane] = input.oz;
      packet.ray.tnear[lane] = 0.0f;

      packet.ray.dir_x[lane] = input.dx;
      packet.ray.dir_y[lane] = input.dy;
      packet.ray.dir_z[lane] = input.dz;
      packet.ray.time[lane] = 0.0f;

      packet.ray.tfar[lane] = std::numeric_limits<float>::max();
      packet.ray.mask[lane] = 0xffffffffu;
      packet.ray.id[lane] = 0;
      packet.ray.flags[lane] = 0;

      packet.hit.geomID[lane] = RTC_INVALID_GEOMETRY_ID;
      packet.hit.primID[lane] = RTC_INVALID_GEOMETRY_ID;

      for (unsigned level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT; ++level) {
        packet.hit.instID[level][lane] = RTC_INVALID_GEOMETRY_ID;
      }
    }

    rtcIntersect4(valid, scene, &packet, &arguments);

    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (packet.hit.geomID[lane] != RTC_INVALID_GEOMETRY_ID) {
        result.checksum += hit_checksum(
            packet.ray.tfar[lane],
            packet.hit.u[lane],
            packet.hit.v[lane],
            packet.hit.Ng_x[lane],
            packet.hit.Ng_y[lane],
            packet.hit.Ng_z[lane],
            packet.hit.primID[lane]);
        ++result.hits;
      }
    }
  }

  return result;
}

TraceSummary trace_packet8_once(RTCScene scene,
                                const std::vector<InputRay>& rays,
                                bool coherent) {
  RTCIntersectArguments arguments;
  init_packet_arguments(arguments, coherent);

  alignas(32) int valid[8] = {
      -1, -1, -1, -1, -1, -1, -1, -1,
  };

  TraceSummary result;
  for (std::size_t base = 0; base < rays.size(); base += 8) {
    alignas(32) RTCRayHit8 packet;

    for (std::size_t lane = 0; lane < 8; ++lane) {
      const InputRay& input = rays[base + lane];

      packet.ray.org_x[lane] = input.ox;
      packet.ray.org_y[lane] = input.oy;
      packet.ray.org_z[lane] = input.oz;
      packet.ray.tnear[lane] = 0.0f;

      packet.ray.dir_x[lane] = input.dx;
      packet.ray.dir_y[lane] = input.dy;
      packet.ray.dir_z[lane] = input.dz;
      packet.ray.time[lane] = 0.0f;

      packet.ray.tfar[lane] = std::numeric_limits<float>::max();
      packet.ray.mask[lane] = 0xffffffffu;
      packet.ray.id[lane] = 0;
      packet.ray.flags[lane] = 0;

      packet.hit.geomID[lane] = RTC_INVALID_GEOMETRY_ID;
      packet.hit.primID[lane] = RTC_INVALID_GEOMETRY_ID;

      for (unsigned level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT; ++level) {
        packet.hit.instID[level][lane] = RTC_INVALID_GEOMETRY_ID;
      }
    }

    rtcIntersect8(valid, scene, &packet, &arguments);

    for (std::size_t lane = 0; lane < 8; ++lane) {
      if (packet.hit.geomID[lane] != RTC_INVALID_GEOMETRY_ID) {
        result.checksum += hit_checksum(
            packet.ray.tfar[lane],
            packet.hit.u[lane],
            packet.hit.v[lane],
            packet.hit.Ng_x[lane],
            packet.hit.Ng_y[lane],
            packet.hit.Ng_z[lane],
            packet.hit.primID[lane]);
        ++result.hits;
      }
    }
  }

  return result;
}

TraceSummary trace_packet16_once(RTCScene scene,
                                 const std::vector<InputRay>& rays,
                                 bool coherent) {
  RTCIntersectArguments arguments;
  init_packet_arguments(arguments, coherent);

  alignas(64) int valid[16] = {
      -1, -1, -1, -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
  };

  TraceSummary result;
  for (std::size_t base = 0; base < rays.size(); base += 16) {
    alignas(64) RTCRayHit16 packet;

    for (std::size_t lane = 0; lane < 16; ++lane) {
      const InputRay& input = rays[base + lane];

      packet.ray.org_x[lane] = input.ox;
      packet.ray.org_y[lane] = input.oy;
      packet.ray.org_z[lane] = input.oz;
      packet.ray.tnear[lane] = 0.0f;

      packet.ray.dir_x[lane] = input.dx;
      packet.ray.dir_y[lane] = input.dy;
      packet.ray.dir_z[lane] = input.dz;
      packet.ray.time[lane] = 0.0f;

      packet.ray.tfar[lane] = std::numeric_limits<float>::max();
      packet.ray.mask[lane] = 0xffffffffu;
      packet.ray.id[lane] = 0;
      packet.ray.flags[lane] = 0;

      packet.hit.geomID[lane] = RTC_INVALID_GEOMETRY_ID;
      packet.hit.primID[lane] = RTC_INVALID_GEOMETRY_ID;

      for (unsigned level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT; ++level) {
        packet.hit.instID[level][lane] = RTC_INVALID_GEOMETRY_ID;
      }
    }

    rtcIntersect16(valid, scene, &packet, &arguments);

    for (std::size_t lane = 0; lane < 16; ++lane) {
      if (packet.hit.geomID[lane] != RTC_INVALID_GEOMETRY_ID) {
        result.checksum += hit_checksum(
            packet.ray.tfar[lane],
            packet.hit.u[lane],
            packet.hit.v[lane],
            packet.hit.Ng_x[lane],
            packet.hit.Ng_y[lane],
            packet.hit.Ng_z[lane],
            packet.hit.primID[lane]);
        ++result.hits;
      }
    }
  }

  return result;
}


template <typename TraceFunction>
TraceResult benchmark_trace(TraceFunction trace) {
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

void print_result(std::string_view quality, std::string_view traversal,
                  double build_ms, std::size_t ray_count,
                  const TraceResult& result) {
  const double mrays_per_second =
      static_cast<double>(ray_count) / (result.ms * 1000.0);
  std::cout << std::left << std::setw(9) << quality << std::setw(14) << traversal
            << std::right << std::fixed << std::setprecision(3)
            << std::setw(11) << build_ms << std::setw(12) << result.ms
            << std::setw(12) << mrays_per_second << std::setw(11) << result.hits
            << std::setw(15) << result.checksum << '\n';
}

void benchmark_quality(RTCDevice device, RTCBuildQuality quality,
                       std::string_view quality_name,
                       const std::vector<Vertex>& vertices,
                       const std::vector<Triangle>& triangles,
                       const std::vector<InputRay>& rays) {
  Scene scene = build_scene(device, quality, vertices, triangles);

  const TraceResult scalar = benchmark_trace(
      [&] { return trace_scalar_once(scene.handle, rays); });

  const TraceResult packet4_incoherent = benchmark_trace(
      [&] { return trace_packet4_once(scene.handle, rays, false); });
  const TraceResult packet4_coherent = benchmark_trace(
      [&] { return trace_packet4_once(scene.handle, rays, true); });

  const TraceResult packet8_incoherent = benchmark_trace(
      [&] { return trace_packet8_once(scene.handle, rays, false); });
  const TraceResult packet8_coherent = benchmark_trace(
      [&] { return trace_packet8_once(scene.handle, rays, true); });

  const TraceResult packet16_incoherent = benchmark_trace(
      [&] { return trace_packet16_once(scene.handle, rays, false); });
  const TraceResult packet16_coherent = benchmark_trace(
      [&] { return trace_packet16_once(scene.handle, rays, true); });

  print_result(quality_name, "scalar1", scene.build_ms, rays.size(), scalar);

  print_result(quality_name, "inc-packet4", scene.build_ms, rays.size(),
               packet4_incoherent);
  print_result(quality_name, "coh-packet4", scene.build_ms, rays.size(),
               packet4_coherent);

  print_result(quality_name, "inc-packet8", scene.build_ms, rays.size(),
               packet8_incoherent);
  print_result(quality_name, "coh-packet8", scene.build_ms, rays.size(),
               packet8_coherent);

  print_result(quality_name, "inc-packet16", scene.build_ms, rays.size(),
               packet16_incoherent);
  print_result(quality_name, "coh-packet16", scene.build_ms, rays.size(),
               packet16_coherent);

  rtcReleaseScene(scene.handle);
}

void benchmark_case(RTCDevice device, std::string_view name,
                    const std::vector<Vertex>& vertices,
                    const std::vector<Triangle>& triangles,
                    const std::vector<InputRay>& rays) {
  if (rays.size() % 16 != 0) {
    throw std::runtime_error("Ray count must be divisible by sixteen");
  }

  std::cout << "\n" << name << "\n"
            << "Triangles: " << triangles.size() << "\n"
            << "Rays: " << rays.size() << "\n"
            << std::left << std::setw(9) << "quality" << std::setw(14)
            << "traversal" << std::right << std::setw(11) << "build ms"
            << std::setw(12) << "trace ms" << std::setw(12) << "MRay/s"
            << std::setw(11) << "hits" << std::setw(15) << "checksum" << '\n';

  benchmark_quality(device, RTC_BUILD_QUALITY_MEDIUM, "medium", vertices,
                    triangles, rays);
  benchmark_quality(device, RTC_BUILD_QUALITY_HIGH, "high", vertices,
                    triangles, rays);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::vector<Vertex> vertices = make_grid_vertices();
    const std::vector<Triangle> triangles = make_triangles();
    const std::vector<InputRay> rays = make_hit_and_miss_rays();

    const std::string build_threads = requested_build_threads();
    const std::string device_configuration = "threads=" + build_threads;
    RTCDevice device = rtcNewDevice(device_configuration.c_str());
    if (!device) {
      throw std::runtime_error("Could not create the Embree device");
    }

    std::cout << "Embree " << RTC_VERSION_STRING
              << " CPU triangle benchmark\n"
              << "Build threads: " << build_threads << "\n"
              << "Native ray4: "
              << (rtcGetDeviceProperty(
                      device, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED)
                      ? "yes"
                      : "no")
              << "\n"
              << "Native ray8: "
              << (rtcGetDeviceProperty(
                      device, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED)
                      ? "yes"
                      : "no")
              << "\n"
              << "Native ray16 (AVX-512): "
              << (rtcGetDeviceProperty(
                      device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED)
                      ? "yes"
                      : "no")
              << "\n"
              << "Traversal repeats: " << kTraversalRepeats << "\n";

    benchmark_case(device, "Regular-grid microbenchmark", vertices, triangles,
                   rays);

    if (argc > 1) {
      const PackedMesh mesh = load_packed_obj(argv[1]);
      const std::vector<InputRay> representative_rays =
          make_representative_camera_rays(mesh.vertices);
      benchmark_case(device, "Representative Dragon camera-ray benchmark",
                     mesh.vertices, mesh.triangles, representative_rays);
    }

    rtcReleaseDevice(device);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
