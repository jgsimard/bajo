cmake_minimum_required(VERSION 3.24)
project(BajoNexusBvhBenchmark LANGUAGES C CXX CUDA)

if (NOT DEFINED NEXUSBVH_DIR)
  message(FATAL_ERROR "Set NEXUSBVH_DIR to a NexusBVH source checkout")
endif()

set(NEXUSBVH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${NEXUSBVH_DIR}" nexusbvh)

find_package(CUDAToolkit REQUIRED)
find_package(Git QUIET)
set(NEXUSBVH_REVISION "unknown")
if (Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${NEXUSBVH_DIR}" rev-parse --short=12 HEAD
    OUTPUT_VARIABLE NEXUSBVH_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

add_executable(bajo_bench_nexusbvh ../bench_nexusbvh.cu)
set_target_properties(
  bajo_bench_nexusbvh PROPERTIES CUDA_ARCHITECTURES native
)
target_compile_features(bajo_bench_nexusbvh PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(
  bajo_bench_nexusbvh PRIVATE NEXUSBVH_REVISION="${NEXUSBVH_REVISION}"
)
target_link_libraries(bajo_bench_nexusbvh PRIVATE NexusBVH CUDA::cudart)
