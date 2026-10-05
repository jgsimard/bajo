#include "params.h"
#include <optix_device.h>

extern "C" __constant__ Params params;

template <bool Occlusion>
static __forceinline__ __device__ void trace_ray() {
  const unsigned int i = optixGetLaunchIndex().x;
  const unsigned int n = params.ray_count;
  const float* r = params.rays;
  unsigned int payload = Occlusion ? 1u : __float_as_uint(-1.0f);
  unsigned int flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
  if (Occlusion)
    flags |= OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
             OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
  optixTrace(params.traversable,
             make_float3(r[i], r[n + i], r[2 * n + i]),
             make_float3(r[4 * n + i], r[5 * n + i], r[6 * n + i]),
             r[3 * n + i], r[7 * n + i], 0.0f, 255, flags,
             0, 1, 0, payload);
  params.results[i] = Occlusion ? float(payload) : __uint_as_float(payload);
}

extern "C" __global__ void __raygen__closest() { trace_ray<false>(); }
extern "C" __global__ void __raygen__occlusion() { trace_ray<true>(); }
extern "C" __global__ void __closesthit__closest() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
}
extern "C" __global__ void __miss__closest() {}
extern "C" __global__ void __miss__occlusion() { optixSetPayload_0(0); }
