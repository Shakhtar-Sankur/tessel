// tessel's CUDA prelude. Under NVRTC (__CUDACC_RTC__) this is plain CUDA.
// Everywhere else the same kernel source compiles as C++ against an
// emulator of the CUDA execution model: a block's threads are OS threads,
// __syncthreads is a barrier over them, and warp operations (shuffles,
// tensor-core mma) exchange values through a per-warp buffer between two
// warp barriers. Blocks run one after another, so shared memory is one
// buffer reused by every block. Kernels are written so that every thread
// reaches every barrier and every warp operation, as CUDA requires.
#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define TSL_CUDA 1
#define KDEV __device__ __forceinline__
#define KGLOBAL(n) extern "C" __global__ void __launch_bounds__(n, 1)
#define SMEM extern __shared__ __align__(16) unsigned char tsmem[]
#define KSYNC() __syncthreads()
#define KWSYNC() __syncwarp()
#define KINF __int_as_float(0x7f800000)
#define KNAN __int_as_float(0x7fffffff)
KDEV float kshfl_xor(float v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }
// fp16 as raw bits (NVRTC has no cuda_fp16.h without the toolkit's headers).
typedef unsigned short khalf;
KDEV khalf kf2h(float f) {
  khalf h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
KDEV float kh2f(khalf h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
// Two consecutive halves as one 32-bit register (p 4-byte aligned).
KDEV unsigned kld2(const khalf *p) { return *(const unsigned *)p; }
// D = A B + D on tensor cores: A 16x8 (row), B 8x8 (col), f16 in, f32
// accumulate. Fragments, with g = lane / 4 and t = lane % 4:
//   a0 = A[g][2t..2t+1], a1 = A[g+8][2t..2t+1], b0 = B[2t..2t+1][g],
//   c[0..1] = C[g][2t..2t+1], c[2..3] = C[g+8][2t..2t+1].
KDEV void kmma(float *c, unsigned a0, unsigned a1, unsigned b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}
// Vector memory access: 4 floats, 8 halves (16 bytes, aligned).
typedef float4 kf4;
typedef uint4 kh8;
KDEV kf4 kld4(const float *p) { return *(const kf4 *)p; }
KDEV kf4 kz4() { return make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
KDEV kh8 kld8h(const khalf *p) { return *(const kh8 *)p; }
KDEV kh8 kz8h() { return make_uint4(0, 0, 0, 0); }
KDEV void kst8h(khalf *p, kh8 v) { *(kh8 *)p = v; }
KDEV void kst4h(khalf *p, kf4 v) {
  uint2 u;
  u.x = (unsigned)kf2h(v.x) | ((unsigned)kf2h(v.y) << 16);
  u.y = (unsigned)kf2h(v.z) | ((unsigned)kf2h(v.w) << 16);
  *(uint2 *)p = u;
}
#else
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define KDEV static inline
#define KGLOBAL(n) static void
#define __restrict__ __restrict
#define KINF INFINITY
#define KNAN NAN
struct kdim3 {
  unsigned x, y, z;
};
#define KEMU_XCH 64
struct KTeam {
  pthread_barrier_t block;
  pthread_barrier_t warp[32];
  float xch[32][32 * KEMU_XCH];
  void (*fn)(float **);
  float **args;
  kdim3 grid, dim;
  unsigned char *smem;
};
static thread_local kdim3 threadIdx, blockIdx, blockDim, gridDim;
static thread_local KTeam *kteam;
static thread_local unsigned ktid;
#define SMEM unsigned char *tsmem = kteam->smem
static inline void KSYNC() { pthread_barrier_wait(&kteam->block); }
static inline void KWSYNC() { pthread_barrier_wait(&kteam->warp[ktid / 32]); }
static inline float *kxch() { return kteam->xch[ktid / 32]; }
static inline float kshfl_xor(float v, int m) {
  float *x = kxch();
  unsigned l = ktid % 32;
  x[l] = v;
  KWSYNC();
  float r = x[l ^ m];
  KWSYNC();
  return r;
}
typedef unsigned short khalf;
// IEEE binary16, round to nearest even (as cvt.rn.f16.f32).
static inline khalf kf2h(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
  int32_t e = (int32_t)((x >> 23) & 0xff);
  if (e == 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  int32_t exp = e - 127 + 15;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    uint32_t shift = 14 - exp, h = mant >> shift, rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return sign | h;
  }
  uint32_t h = ((uint32_t)exp << 10) | (mant >> 13), rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
  return sign | h;
}
static inline float kh2f(khalf h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {
      int32_t e = 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        e--;
      }
      x = sign | ((uint32_t)(e + 127 - 15) << 23) | ((mant & 0x3ff) << 13);
    }
  } else if (exp == 31) {
    x = sign | 0x7f800000 | (mant << 13);
  } else {
    x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &x, 4);
  return f;
}
static inline unsigned kld2(const khalf *p) {
  unsigned v;
  memcpy(&v, p, 4);
  return v;
}
// The warp's fragments meet in the exchange buffer; each lane computes its
// four outputs from them, in the fragment layout of mma.m16n8k8.
static inline void kmma(float *c, unsigned a0, unsigned a1, unsigned b0) {
  unsigned *x = (unsigned *)kxch();
  unsigned l = ktid % 32, g = l / 4, t = l % 4;
  x[l * 3] = a0;
  x[l * 3 + 1] = a1;
  x[l * 3 + 2] = b0;
  KWSYNC();
  for (int i = 0; i < 4; i++) {
    unsigned row = g + (i >= 2 ? 8 : 0), col = 2 * t + (i & 1);
    float s = 0.0f;
    for (unsigned k = 0; k < 8; k++) {
      unsigned ra = x[((row % 8) * 4 + k / 2) * 3 + (row >= 8 ? 1 : 0)];
      unsigned rb = x[(col * 4 + k / 2) * 3 + 2];
      khalf ha = (khalf)(ra >> (16 * (k & 1))), hb = (khalf)(rb >> (16 * (k & 1)));
      s += kh2f(ha) * kh2f(hb);
    }
    c[i] += s;
  }
  KWSYNC();
}
struct kf4 {
  float x, y, z, w;
};
struct kh8 {
  unsigned x, y, z, w;
};
static inline kf4 kld4(const float *p) {
  kf4 v;
  memcpy(&v, p, 16);
  return v;
}
static inline kf4 kz4() { return kf4{0.0f, 0.0f, 0.0f, 0.0f}; }
static inline kh8 kld8h(const khalf *p) {
  kh8 v;
  memcpy(&v, p, 16);
  return v;
}
static inline kh8 kz8h() { return kh8{0, 0, 0, 0}; }
static inline void kst8h(khalf *p, kh8 v) { memcpy(p, &v, 16); }
static inline void kst4h(khalf *p, kf4 v) {
  khalf h[4] = {kf2h(v.x), kf2h(v.y), kf2h(v.z), kf2h(v.w)};
  memcpy(p, h, 8);
}
struct KStart {
  KTeam *t;
  unsigned tid;
};
static void *kemu_run(void *p) {
  KStart *s = (KStart *)p;
  kteam = s->t;
  ktid = s->tid;
  blockDim = kteam->dim;
  gridDim = kteam->grid;
  threadIdx.x = ktid % blockDim.x;
  threadIdx.y = ktid / blockDim.x;
  threadIdx.z = 0;
  for (unsigned z = 0; z < gridDim.z; z++)
    for (unsigned y = 0; y < gridDim.y; y++)
      for (unsigned x = 0; x < gridDim.x; x++) {
        blockIdx.x = x;
        blockIdx.y = y;
        blockIdx.z = z;
        kteam->fn(kteam->args);
        // Every thread leaves the block before the next one reuses shared memory.
        pthread_barrier_wait(&kteam->block);
      }
  return 0;
}
static void kemu_launch(void (*fn)(float **), float **args, unsigned gx, unsigned gy, unsigned gz,
                        unsigned bx, unsigned by, unsigned smem) {
  unsigned n = bx * by;
  KTeam *t = (KTeam *)calloc(1, sizeof(KTeam));
  t->fn = fn;
  t->args = args;
  t->grid = {gx, gy, gz};
  t->dim = {bx, by, 1};
  t->smem = (unsigned char *)calloc(smem + 16, 1);
  // Uninitialized shared memory is NaN, so a read before a write shows up.
  for (unsigned i = 0; i < smem / 4; i++) ((float *)t->smem)[i] = NAN;
  pthread_barrier_init(&t->block, 0, n);
  for (unsigned w = 0; w < (n + 31) / 32; w++) pthread_barrier_init(&t->warp[w], 0, 32);
  pthread_t *th = (pthread_t *)malloc(n * sizeof(pthread_t));
  KStart *st = (KStart *)malloc(n * sizeof(KStart));
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 1 << 18);
  for (unsigned i = 0; i < n; i++) {
    st[i] = {t, i};
    pthread_create(&th[i], &attr, kemu_run, &st[i]);
  }
  for (unsigned i = 0; i < n; i++) pthread_join(th[i], 0);
  pthread_attr_destroy(&attr);
  pthread_barrier_destroy(&t->block);
  for (unsigned w = 0; w < (n + 31) / 32; w++) pthread_barrier_destroy(&t->warp[w]);
  free(th);
  free(st);
  free(t->smem);
  free(t);
}
#endif

// ---- shared by both: math and reductions ----
KDEV float kmax(float a, float b) { return a > b ? a : b; }
KDEV float ksig(float x) { return 1.0f / (1.0f + expf(-x)); }
KDEV float krelu(float x) { return x > 0.0f ? x : 0.0f; }
// Reduce over groups of g consecutive lanes (g a power of two, <= 32).
KDEV float kwarp_sum(float v, int g) {
  for (int m = g / 2; m > 0; m /= 2) v += kshfl_xor(v, m);
  return v;
}
KDEV float kwarp_max(float v, int g) {
  for (int m = g / 2; m > 0; m /= 2) v = kmax(v, kshfl_xor(v, m));
  return v;
}

// ---- tessel ----
#ifdef TSL_CUDA
KDEV int kshfl_xor_i(int v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }
#else
static inline int kshfl_xor_i(int v, int m) {
  float f, r;
  memcpy(&f, &v, 4);
  r = kshfl_xor(f, m);
  int o;
  memcpy(&o, &r, 4);
  return o;
}
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif
// exp with the hardware's fast path (ex2.approx) on the GPU.
#ifdef TSL_CUDA
#define kexp(x) __expf(x)
#else
#define kexp(x) expf(x)
#endif
// A value rounded to f16 (f16 arithmetic happens in f32, rounded per op).
KDEV float kr16(float x) { return kh2f(kf2h(x)); }
// Floor division and modulo, as in Python (the tile language's semantics).
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
// More vector accesses: 2 floats, 4 halves (8 bytes), stores from floats.
#ifdef TSL_CUDA
KDEV float2 kld2f(const float *p) { return *(const float2 *)p; }
KDEV uint2 kld4h(const khalf *p) { return *(const uint2 *)p; }
KDEV void kst2f(float *p, float a, float b) { *(float2 *)p = make_float2(a, b); }
KDEV void kst4f(float *p, float a, float b, float c, float d) { *(float4 *)p = make_float4(a, b, c, d); }
#else
struct float2 {
  float x, y;
};
struct uint2 {
  unsigned x, y;
};
static inline float2 kld2f(const float *p) {
  float2 v;
  memcpy(&v, p, 8);
  return v;
}
static inline uint2 kld4h(const khalf *p) {
  uint2 v;
  memcpy(&v, p, 8);
  return v;
}
static inline void kst2f(float *p, float a, float b) {
  p[0] = a;
  p[1] = b;
}
static inline void kst4f(float *p, float a, float b, float c, float d) {
  p[0] = a;
  p[1] = b;
  p[2] = c;
  p[3] = d;
}
#endif
KDEV float klo(unsigned u) { return kh2f((khalf)(u & 0xffffu)); }
KDEV float khi(unsigned u) { return kh2f((khalf)(u >> 16)); }
KDEV unsigned kpack(float a, float b) { return (unsigned)kf2h(a) | ((unsigned)kf2h(b) << 16); }
#ifdef TSL_CUDA
KDEV void kst2h(khalf *p, float a, float b) { *(unsigned *)p = kpack(a, b); }
#else
static inline void kst2h(khalf *p, float a, float b) {
  unsigned u = kpack(a, b);
  memcpy(p, &u, 4);
}
#endif
#ifdef TSL_CUDA
KDEV void kst4hf(khalf *p, float a, float b, float c, float d) { *(uint2 *)p = make_uint2(kpack(a, b), kpack(c, d)); }
KDEV void kst8hf(khalf *p, float a, float b, float c, float d, float e, float f, float g, float h) {
  *(uint4 *)p = make_uint4(kpack(a, b), kpack(c, d), kpack(e, f), kpack(g, h));
}
#else
static inline void kst4hf(khalf *p, float a, float b, float c, float d) {
  unsigned u[2] = {kpack(a, b), kpack(c, d)};
  memcpy(p, u, 8);
}
static inline void kst8hf(khalf *p, float a, float b, float c, float d, float e, float f, float g, float h) {
  unsigned u[4] = {kpack(a, b), kpack(c, d), kpack(e, f), kpack(g, h)};
  memcpy(p, u, 16);
}
#endif
// ldmatrix: a warp loads n (1, 2 or 4) 8x8 matrices of halves from shared
// memory; lanes 8i..8i+7 give the row addresses of matrix i. Lane l gets
// in r[i] the pair (row l/4, columns 2(l%4), 2(l%4)+1) of matrix i, or with
// trans the pair (rows 2(l%4), 2(l%4)+1, column l/4).
#ifdef TSL_CUDA
KDEV unsigned ksaddr(const void *p) {
  unsigned a;
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
  return a;
}
KDEV void kldm4(unsigned *r, const khalf *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(ksaddr(p)));
}
KDEV void kldm4t(unsigned *r, const khalf *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(ksaddr(p)));
}
KDEV void kldm2(unsigned *r, const khalf *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(ksaddr(p)));
}
KDEV void kldm2t(unsigned *r, const khalf *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(ksaddr(p)));
}
KDEV void kst16(khalf *p, unsigned a, unsigned b, unsigned c, unsigned d) { *(uint4 *)p = make_uint4(a, b, c, d); }
#else
static inline void kldm(unsigned *r, int n, bool trans, const khalf *p) {
  const khalf **x = (const khalf **)kxch();
  unsigned l = ktid % 32;
  x[l] = p;
  KWSYNC();
  for (int i = 0; i < n; i++) {
    khalf lo, hi;
    if (!trans) {
      const khalf *row = x[8 * i + l / 4];
      lo = row[2 * (l % 4)];
      hi = row[2 * (l % 4) + 1];
    } else {
      lo = x[8 * i + 2 * (l % 4)][l / 4];
      hi = x[8 * i + 2 * (l % 4) + 1][l / 4];
    }
    r[i] = (unsigned)lo | ((unsigned)hi << 16);
  }
  KWSYNC();
}
static inline void kldm4(unsigned *r, const khalf *p) { kldm(r, 4, false, p); }
static inline void kldm4t(unsigned *r, const khalf *p) { kldm(r, 4, true, p); }
static inline void kldm2(unsigned *r, const khalf *p) { kldm(r, 2, false, p); }
static inline void kldm2t(unsigned *r, const khalf *p) { kldm(r, 2, true, p); }
static inline void kst16(khalf *p, unsigned a, unsigned b, unsigned c, unsigned d) {
  unsigned u[4] = {a, b, c, d};
  memcpy(p, u, 16);
}
#endif
// A 32-bit store of two packed halves.
#ifdef TSL_CUDA
KDEV void kstu(khalf *p, unsigned u) { *(unsigned *)p = u; }
#else
static inline void kstu(khalf *p, unsigned u) { memcpy(p, &u, 4); }
#endif

// A matmul operand's place in shared memory: row r, column c (in halves)
// of a tile with ld halves per row, its 16-byte chunks permuted by XOR with
// the row (>> sh, masked by m) so 8-row ldmatrix reads and 16-byte stores
// touch distinct banks without padding.
KDEV int kswz(int r, int c, int ld, int sh, int m) { return r * ld + ((((c >> 3) ^ ((r >> sh) & m))) << 3) + (c & 7); }
