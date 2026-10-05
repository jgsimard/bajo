#include "params.h"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kWarmups = 3;
constexpr int kRepeats = 11;
constexpr int kBatch = 8;
using Clock = std::chrono::steady_clock;

void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void check(OptixResult code) {
  if (code != OPTIX_SUCCESS)
    throw std::runtime_error(std::string(optixGetErrorName(code)) + ": " +
                             optixGetErrorString(code));
}
struct Buffer {
  CUdeviceptr ptr = 0;
  explicit Buffer(size_t bytes) {
    check(cudaMalloc(reinterpret_cast<void**>(&ptr), bytes));
  }
  ~Buffer() { if (ptr) cudaFree(reinterpret_cast<void*>(ptr)); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void upload(const void* data, size_t bytes) {
    check(cudaMemcpy(reinterpret_cast<void*>(ptr), data, bytes,
                     cudaMemcpyHostToDevice));
  }
};

template <class T> std::vector<T> read_binary(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot read " + path.string());
  const auto size = file.tellg();
  if (size <= 0 || static_cast<size_t>(size) % sizeof(T))
    throw std::runtime_error("Invalid binary size: " + path.string());
  std::vector<T> data(static_cast<size_t>(size) / sizeof(T));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(data.data()), size))
    throw std::runtime_error("Short read: " + path.string());
  return data;
}

struct Stats { double median, minimum, maximum; };
Stats summarize(std::vector<double> times) {
  std::sort(times.begin(), times.end());
  return {times[times.size() / 2], times.front(), times.back()};
}
double elapsed(Clock::time_point start, int batch = 1) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count()
         / batch;
}

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

class Pipeline {
 public:
  OptixModule module = nullptr;
  OptixProgramGroup groups[3] = {};
  OptixPipeline pipeline = nullptr;
  Buffer records{3 * sizeof(Record)};
  OptixShaderBindingTable sbt = {};

  Pipeline(OptixDeviceContext ctx, const std::string& ptx, bool occlusion) {
    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions options = {};
    options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    options.numPayloadValues = 1;
    options.numAttributeValues = 2;
    options.pipelineLaunchParamsVariableName = "params";
    options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    char log[8192];
    size_t log_size = sizeof(log);
    auto result = optixModuleCreate(ctx, &module_options, &options,
                                   ptx.data(), ptx.size(), log, &log_size, &module);
    if (result != OPTIX_SUCCESS) std::cerr << log << '\n';
    check(result);
    OptixProgramGroupDesc descs[3] = {};
    descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descs[0].raygen.module = module;
    descs[0].raygen.entryFunctionName = occlusion ? "__raygen__occlusion" : "__raygen__closest";
    descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descs[1].miss.module = module;
    descs[1].miss.entryFunctionName = occlusion ? "__miss__occlusion" : "__miss__closest";
    descs[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    if (!occlusion) {
      descs[2].hitgroup.moduleCH = module;
      descs[2].hitgroup.entryFunctionNameCH = "__closesthit__closest";
    }
    OptixProgramGroupOptions group_options = {};
    log_size = sizeof(log);
    result = optixProgramGroupCreate(ctx, descs, 3, &group_options, log, &log_size, groups);
    if (result != OPTIX_SUCCESS) std::cerr << log << '\n';
    check(result);
    OptixPipelineLinkOptions link = {};
    link.maxTraceDepth = 1;
    log_size = sizeof(log);
    result = optixPipelineCreate(ctx, &options, &link, groups, 3, log, &log_size, &pipeline);
    if (result != OPTIX_SUCCESS) std::cerr << log << '\n';
    check(result);
    OptixStackSizes sizes = {};
    for (auto group : groups) check(optixUtilAccumulateStackSizes(group, &sizes, pipeline));
    unsigned int traversal, state, continuation;
    check(optixUtilComputeStackSizes(&sizes, 1, 0, 0, &traversal, &state, &continuation));
    check(optixPipelineSetStackSize(pipeline, traversal, state, continuation, 1));
    Record host[3] = {};
    for (int i = 0; i < 3; ++i) check(optixSbtRecordPackHeader(groups[i], &host[i]));
    records.upload(host, sizeof(host));
    sbt.raygenRecord = records.ptr;
    sbt.missRecordBase = records.ptr + sizeof(Record);
    sbt.missRecordStrideInBytes = sizeof(Record);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = records.ptr + 2 * sizeof(Record);
    sbt.hitgroupRecordStrideInBytes = sizeof(Record);
    sbt.hitgroupRecordCount = 1;
  }
  ~Pipeline() {
    if (pipeline) optixPipelineDestroy(pipeline);
    for (auto group : groups) if (group) optixProgramGroupDestroy(group);
    if (module) optixModuleDestroy(module);
  }
};

void run(OptixDeviceContext ctx, const std::string& ptx,
         const std::filesystem::path& fixture, const std::filesystem::path& output,
         bool fast_build) {
  const std::string label = fast_build ? "optix-fast-build" : "optix-fast-trace";
  auto vertices = read_binary<float>(fixture / "vertices.f32");
  if (vertices.size() % 9 || vertices.size() / 3 > std::numeric_limits<unsigned int>::max())
    throw std::runtime_error("Invalid triangle vertices");
  const unsigned int triangles = vertices.size() / 9;
  Buffer d_vertices(vertices.size() * sizeof(float));
  d_vertices.upload(vertices.data(), vertices.size() * sizeof(float));
  unsigned int flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  OptixBuildInput input = {};
  input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  input.triangleArray.vertexBuffers = &d_vertices.ptr;
  input.triangleArray.numVertices = vertices.size() / 3;
  input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  input.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
  input.triangleArray.flags = &flags;
  input.triangleArray.numSbtRecords = 1;
  OptixAccelBuildOptions options = {};
  options.buildFlags = fast_build ? OPTIX_BUILD_FLAG_PREFER_FAST_BUILD : OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  options.operation = OPTIX_BUILD_OPERATION_BUILD;
  check(cudaDeviceSynchronize());
  std::vector<double> times;
  for (int i = 0; i < kWarmups + kRepeats; ++i) {
    const auto start = Clock::now();
    OptixAccelBufferSizes sizes;
    check(optixAccelComputeMemoryUsage(ctx, &options, &input, 1, &sizes));
    Buffer scratch(sizes.tempSizeInBytes), gas(sizes.outputSizeInBytes);
    OptixTraversableHandle handle;
    check(optixAccelBuild(ctx, nullptr, &options, &input, 1,
                         scratch.ptr, sizes.tempSizeInBytes, gas.ptr,
                         sizes.outputSizeInBytes, &handle, nullptr, 0));
    check(cudaDeviceSynchronize());
    const double ms = elapsed(start);
    if (i >= kWarmups) times.push_back(ms);
  }
  const Stats build = summarize(times);
  std::cout << "BUILD\toptix\t" << label << '\t' << triangles << '\t'
            << build.median << '\t' << build.minimum << '\t' << build.maximum << '\n';
  OptixAccelBufferSizes sizes;
  check(optixAccelComputeMemoryUsage(ctx, &options, &input, 1, &sizes));
  Buffer scratch(sizes.tempSizeInBytes), gas(sizes.outputSizeInBytes);
  OptixTraversableHandle handle;
  check(optixAccelBuild(ctx, nullptr, &options, &input, 1, scratch.ptr,
                       sizes.tempSizeInBytes, gas.ptr, sizes.outputSizeInBytes,
                       &handle, nullptr, 0));
  check(cudaDeviceSynchronize());
  std::cout << "MEMORY\t" << label << '\t' << sizes.outputSizeInBytes << '\t'
            << sizes.tempSizeInBytes << '\n';
  Pipeline closest(ctx, ptx, false), occlusion(ctx, ptx, true);
  for (const char* workload : {"camera", "shuffled", "random"}) {
    auto rays = read_binary<float>(fixture / (std::string(workload) + ".f32"));
    if (rays.size() % 8 || rays.size() / 8 > std::numeric_limits<unsigned int>::max())
      throw std::runtime_error("Invalid rays");
    const unsigned int count = rays.size() / 8;
    Buffer d_rays(rays.size() * sizeof(float)), results(count * sizeof(float)), d_params(sizeof(Params));
    d_rays.upload(rays.data(), rays.size() * sizeof(float));
    Params params{handle, reinterpret_cast<const float*>(d_rays.ptr),
                  reinterpret_cast<float*>(results.ptr), count};
    d_params.upload(&params, sizeof(params));
    check(cudaDeviceSynchronize());
    for (bool any : {false, true}) {
      Pipeline& pipeline = any ? occlusion : closest;
      const char* mode = any ? "occlusion" : "closest";
      times.clear();
      for (int i = 0; i < kWarmups + kRepeats; ++i) {
        const auto start = Clock::now();
        for (int j = 0; j < kBatch; ++j)
          check(optixLaunch(pipeline.pipeline, nullptr, d_params.ptr, sizeof(Params),
                            &pipeline.sbt, count, 1, 1));
        check(cudaDeviceSynchronize());
        const double ms = elapsed(start, kBatch);
        if (i >= kWarmups) times.push_back(ms);
      }
      const Stats trace = summarize(times);
      std::vector<float> host(count);
      check(cudaMemcpy(host.data(), reinterpret_cast<void*>(results.ptr),
                       count * sizeof(float), cudaMemcpyDeviceToHost));
      const auto path = output / (label + "-" + workload + "-" + mode + ".f32");
      std::ofstream file(path, std::ios::binary);
      if (!file.write(reinterpret_cast<char*>(host.data()), count * sizeof(float)))
        throw std::runtime_error("Cannot write " + path.string());
      std::cout << "TRACE\toptix\t" << label << '\t' << workload << '\t' << mode
                << '\t' << count << '\t' << trace.median << '\t'
                << trace.minimum << '\t' << trace.maximum << '\n';
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("usage: bench_optix programs.ptx fixture_dir output_dir");
    check(cudaSetDevice(0));
    check(cudaFree(nullptr));
    check(optixInit());
    OptixDeviceContextOptions options = {};
    OptixDeviceContext ctx;
    check(optixDeviceContextCreate(nullptr, &options, &ctx));
    const auto bytes = read_binary<char>(argv[1]);
    const std::string ptx(bytes.begin(), bytes.end());
    std::filesystem::create_directories(argv[3]);
    std::cout << std::setprecision(12) << "OPTIX_VERSION\t" << OPTIX_VERSION << '\n';
    run(ctx, ptx, argv[2], argv[3], false);
    run(ctx, ptx, argv[2], argv[3], true);
    check(optixDeviceContextDestroy(ctx));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
