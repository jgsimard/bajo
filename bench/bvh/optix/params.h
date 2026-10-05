#pragma once

#include <optix.h>
#include <cuda_runtime.h>

// Rays use Bajo's field-major ABI: ox, oy, oz, tmin, dx, dy, dz, tmax.
struct Params {
  OptixTraversableHandle traversable;
  const float* rays;
  float* results;
  unsigned int ray_count;
};
