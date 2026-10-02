// tessel's Metal prelude: the helpers the generated kernels call, in the
// Metal Shading Language, for Apple GPUs. Kernels are generated as for
// CUDA without tensor-core instructions; a SIMD-group is 32 threads, as a
// CUDA warp, and threads join SIMD-groups in order of their index.
#include <metal_stdlib>
using namespace metal;
#define KDEV inline
#define TSP threadgroup
#define KI64 long
#define SMEM
#define KSYNC() threadgroup_barrier(mem_flags::mem_threadgroup)
#define KWSYNC() simdgroup_barrier(mem_flags::mem_threadgroup)
#define KINF as_type<float>(0x7f800000u)
#define KNAN as_type<float>(0x7fc00000u)
#define kexp(x) exp(x)
#define logf(x) log(x)
#define sqrtf(x) sqrt(x)
#define rsqrtf(x) rsqrt(x)
#define fabsf(x) fabs(x)
#define fmaxf(a, b) fmax(a, b)
#define fminf(a, b) fmin(a, b)
#define fmaf(a, b, c) fma(a, b, c)
// fp16 as raw bits, as in the CUDA prelude.
typedef ushort khalf;
typedef float4 kf4;
typedef uint4 kh8;
KDEV float kshfl_xor(float v, int m) { return simd_shuffle_xor(v, ushort(m)); }
KDEV int kshfl_xor_i(int v, int m) { return simd_shuffle_xor(v, ushort(m)); }
KDEV khalf kf2h(float f) { return as_type<ushort>(half(f)); }
KDEV float kh2f(khalf h) { return float(as_type<half>(h)); }
KDEV float kr16(float x) { return kh2f(kf2h(x)); }
KDEV int kfdiv(int a, int b) {
  int q = a / b;
  return (q * b != a && ((a < 0) != (b < 0))) ? q - 1 : q;
}
KDEV int kfmod(int a, int b) {
  int r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}
KDEV int kmaxi(int a, int b) { return a > b ? a : b; }
KDEV int kmini(int a, int b) { return a < b ? a : b; }
KDEV float klo(uint u) { return kh2f(khalf(u & 0xffffu)); }
KDEV float khi(uint u) { return kh2f(khalf(u >> 16)); }
KDEV uint kpack(float a, float b) { return uint(kf2h(a)) | (uint(kf2h(b)) << 16); }
KDEV kf4 kz4() { return float4(0.0f); }
KDEV kh8 kz8h() { return uint4(0u); }
// Vector loads and stores, for device and threadgroup memory alike.
#define KMEM(AS)                                                                                    \
  KDEV kf4 kld4(const AS float *p) { return *(const AS float4 *)p; }                                \
  KDEV float2 kld2f(const AS float *p) { return *(const AS float2 *)p; }                            \
  KDEV uint2 kld4h(const AS khalf *p) { return *(const AS uint2 *)p; }                              \
  KDEV kh8 kld8h(const AS khalf *p) { return *(const AS uint4 *)p; }                                \
  KDEV uint kld2(const AS khalf *p) { return *(const AS uint *)p; }                                 \
  KDEV void kst2f(AS float *p, float a, float b) { *(AS float2 *)p = float2(a, b); }                \
  KDEV void kst4f(AS float *p, float a, float b, float c, float d) { *(AS float4 *)p = float4(a, b, c, d); } \
  KDEV void kst8h(AS khalf *p, kh8 v) { *(AS uint4 *)p = v; }                                       \
  KDEV void kst4h(AS khalf *p, kf4 v) { *(AS uint2 *)p = uint2(kpack(v.x, v.y), kpack(v.z, v.w)); } \
  KDEV void kst2h(AS khalf *p, float a, float b) { *(AS uint *)p = kpack(a, b); }                   \
  KDEV void kst4hf(AS khalf *p, float a, float b, float c, float d) {                               \
    *(AS uint2 *)p = uint2(kpack(a, b), kpack(c, d));                                               \
  }                                                                                                 \
  KDEV void kst8hf(AS khalf *p, float a, float b, float c, float d, float e, float f, float g, float h) { \
    *(AS uint4 *)p = uint4(kpack(a, b), kpack(c, d), kpack(e, f), kpack(g, h));                     \
  }                                                                                                 \
  KDEV void kstu(AS khalf *p, uint u) { *(AS uint *)p = u; }                                        \
  KDEV void kst16(AS khalf *p, uint a, uint b, uint c, uint d) { *(AS uint4 *)p = uint4(a, b, c, d); }
KMEM(device)
KMEM(threadgroup)
