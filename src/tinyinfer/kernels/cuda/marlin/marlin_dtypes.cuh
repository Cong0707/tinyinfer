
#ifndef _data_types_cuh
#define _data_types_cuh
#include "marlin.cuh"
#include "scalar_type.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

template <long scalar_type_id>
class MarlinScalarType {};

template <>
class MarlinScalarType<vllm::kFloat16.id()> {
 public:
  using scalar_t = half;
  using scalar_t2 = half2;
  using scalar_t4 = half2;
  using scalar_32bit_t = half2;

  // Matrix fragments for tensor core instructions; their precise layout is
  // documented here:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  using FragA = Vec<half2, 4>;
  using FragB = Vec<half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<half2, 1>;
  using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
  using FragZP = Vec<half2, 4>;

  static __device__ float inline num2float(const half x) {
    return __half2float(x);
  }

  static __device__ half2 inline num2num2(const half x) {
    return __half2half2(x);
  }

  static __device__ half2 inline nums2num2(const half x1, const half x2) {
    return __halves2half2(x1, x2);
  }

  static __host__ __device__ half inline float2num(const float x) {
    return __float2half(x);
  }

  static __host__ __device__ float2 inline num22float2(const half2 x) {
    return __half22float2(x);
  }
};

template <>
class MarlinScalarType<vllm::kBFloat16.id()> {
 public:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;
  using scalar_t4 = nv_bfloat162;
  using scalar_32bit_t = nv_bfloat162;

  using FragA = Vec<nv_bfloat162, 4>;
  using FragB = Vec<nv_bfloat162, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<nv_bfloat162, 1>;
  using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
  using FragZP = Vec<nv_bfloat162, 4>;

  // bf16 arithmetic is native from SM 80. Before that the CUDA headers only emulate
  // float<->bfloat16 (via __bfloat162float and friends), and those helpers are NOT
  // defined at all below SM 80 in some toolkit versions -- which is why this class used
  // to compile to an empty shell on Turing and every call site failed with "has no member
  // num2float". Upstream vLLM sidesteps it by running Turing in fp16; this engine runs
  // the checkpoint's bf16 activations everywhere, so the conversions have to exist.
  //
  // The emulation is exact and cheap: bf16 IS the top 16 bits of fp32 (same exponent
  // field, 7 explicit mantissa bits), so widening is a shift and narrowing is a shift
  // plus round-to-nearest-even. No arithmetic is performed in bf16 here -- the kernels
  // convert to float, accumulate in float, and convert back -- so the missing bf16 ALU
  // on Turing costs nothing on this path.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  static __device__ float inline num2float(const nv_bfloat16 x) {
    return __bfloat162float(x);
  }

  static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
  }

  static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1,
                                                  const nv_bfloat16 x2) {
    return __halves2bfloat162(x1, x2);
  }

  static __host__ __device__ nv_bfloat16 inline float2num(const float x) {
    return __float2bfloat16(x);
  }

  static __host__ __device__ float2 inline num22float2(const nv_bfloat162 x) {
    return __bfloat1622float2(x);
  }
#else
  // SM 75 and below: widen/narrow by hand.
  static __device__ float inline num2float(const nv_bfloat16 x) {
    uint32_t bits = 0u;
    memcpy(&bits, &x, sizeof(uint16_t));
    bits <<= 16;
    float out;
    memcpy(&out, &bits, sizeof(float));
    return out;
  }

  static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) {
    nv_bfloat162 out;
    uint32_t bits = 0u;
    memcpy(&bits, &x, sizeof(uint16_t));
    uint32_t packed = bits | (bits << 16);
    memcpy(&out, &packed, sizeof(uint32_t));
    return out;
  }

  static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1,
                                                  const nv_bfloat16 x2) {
    uint32_t lo = 0u, hi = 0u;
    memcpy(&lo, &x1, sizeof(uint16_t));
    memcpy(&hi, &x2, sizeof(uint16_t));
    uint32_t packed = lo | (hi << 16);
    nv_bfloat162 out;
    memcpy(&out, &packed, sizeof(uint32_t));
    return out;
  }

  static __host__ __device__ nv_bfloat16 inline float2num(const float x) {
    // round-to-nearest-even on the discarded low 16 bits
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    uint16_t top = (uint16_t)(bits >> 16);
    nv_bfloat16 out;
    memcpy(&out, &top, sizeof(uint16_t));
    return out;
  }

  static __host__ __device__ float2 inline num22float2(const nv_bfloat162 x) {
    float2 out;
    uint32_t packed;
    memcpy(&packed, &x, sizeof(uint32_t));
    nv_bfloat16 lo, hi;
    uint16_t l = (uint16_t)(packed & 0xFFFFu), h = (uint16_t)(packed >> 16);
    memcpy(&lo, &l, sizeof(uint16_t));
    memcpy(&hi, &h, sizeof(uint16_t));
    out.x = num2float(lo);
    out.y = num2float(hi);
    return out;
  }
#endif
};

template <>
class MarlinScalarType<vllm::kFE4M3fn.id()> {
 public:
  using scalar_t = __nv_fp8_e4m3;
  using scalar_t2 = __nv_fp8x2_e4m3;
  using scalar_t4 = __nv_fp8x4_e4m3;
  using scalar_32bit_t = __nv_fp8x4_e4m3;

  using FragA = Vec<__nv_fp8x4_e4m3, 4>;
  using FragB = Vec<__nv_fp8x4_e4m3, 2>;
  using FragC = Vec<float, 4>;
  using FragZP = Vec<__nv_fp8x2_e4m3, 4>;

  static __host__ __device__
      float2 inline num22float2(const __nv_fp8x2_e4m3 x) {
    return (float2)x;
  }
};

template <>
class MarlinScalarType<vllm::kS8.id()> {
 public:
  using scalar_t = int8_t;
  using scalar_t2 = int16_t;
  using scalar_t4 = int32_t;
  using scalar_32bit_t = int32_t;

  using FragA = Vec<int32_t, 4>;
  using FragB = Vec<int32_t, 2>;
  using FragC = Vec<float, 4>;
  using FragZP = Vec<int16_t, 4>;
};

template <typename scalar_t>
class MarlinScalarType2 {};

template <>
class MarlinScalarType2<half> : public MarlinScalarType<vllm::kFloat16.id()> {};

template <>
class MarlinScalarType2<nv_bfloat16>
    : public MarlinScalarType<vllm::kBFloat16.id()> {};

template <>
class MarlinScalarType2<__nv_fp8_e4m3>
    : public MarlinScalarType<vllm::kFE4M3fn.id()> {};

template <>
class MarlinScalarType2<int8_t> : public MarlinScalarType<vllm::kS8.id()> {};

}  // namespace MARLIN_NAMESPACE_NAME

#endif
