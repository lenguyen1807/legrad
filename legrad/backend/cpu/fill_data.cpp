#include <cstdint>

#include "fill_data.h"

// clang-format off
#if defined(LEGRAD_USE_ARM) && __ARM_NEON
  #define ARM_SIMD
  #include <arm_neon.h>
#elif defined(LNDL_USE_X86) && __AVX__
  #define AVX_SIMD
  #include <x86intrin.h>
#endif

// clang-format on
#define SIMD_FILL(type, vec_define, vec_size, fill_method) \
  type vec = vec_define(value); \
  size_t i = 0; \
  for (; i + vec_size <= n; i += vec_size) \
    fill_method(data + i, vec); \
  for (; i < n; ++i) \
    data[i] = value;

void fill_cpu_float64(double* data, size_t n, double value)
{
#if defined(ARM_SIMD) && defined(__aarch64__)
  SIMD_FILL(float64x2_t, vdupq_n_f64, 2, vst1q_f64)
#elif defined(X86_SIMD)
  SIMD_FILL(__m256d, _mm256_set1_pd, 4, _mm256_storeu_pd)
#else
  fill_cpu<double>(data, n, value);
#endif
}

void fill_cpu_float32(float* data, size_t n, float value)
{
#ifdef ARM_SIMD
  SIMD_FILL(float32x4_t, vdupq_n_f32, 4, vst1q_f32)
#elif defined(X86_SIMD)
  SIMD_FILL(__m256, _mm256_set1_ps, 8, _mm256_storeu_ps)
#else  // fallback
  fill_cpu<float>(data, n, value);
#endif
}

void fill_cpu_int64(int64_t* data, size_t n, int64_t value)
{
#if defined(ARM_SIMD) && defined(__aarch64__)
  SIMD_FILL(int64x2_t, vdupq_n_s64, 2, vst1q_s64)
#else
  fill_cpu<int64_t>(data, n, value);
#endif
}

void fill_cpu_int32(int32_t* data, size_t n, int32_t value)
{
#ifdef ARM_SIMD
  SIMD_FILL(int32x4_t, vdupq_n_s32, 4, vst1q_s32);
#else
  fill_cpu<int32_t>(data, n, value);
#endif
}

void fill_cpu_int16(int16_t* data, size_t n, int16_t value)
{
#ifdef ARM_SIMD
  SIMD_FILL(int16x8_t, vdupq_n_s16, 8, vst1q_s16)
#else
  fill_cpu<int16_t>(data, n, value);
#endif
}

void fill_cpu_int8(int8_t* data, size_t n, int8_t value)
{
#ifdef ARM_SIMD
  SIMD_FILL(int8x16_t, vdupq_n_s8, 16, vst1q_s8)
#else
  fill_cpu<int8_t>(data, n, value);
#endif
}

void fill_cpu_uint64(uint64_t* data, size_t n, uint64_t value)
{
#if defined(ARM_SIMD) && defined(__aarch64__)
  SIMD_FILL(uint64x2_t, vdupq_n_u64, 2, vst1q_u64)
#else
  fill_cpu<uint64_t>(data, n, value);
#endif
}

void fill_cpu_uint32(uint32_t* data, size_t n, uint32_t value)
{
#ifdef ARM_SIMD
  SIMD_FILL(uint32x4_t, vdupq_n_u32, 4, vst1q_u32);
#else
  fill_cpu<uint32_t>(data, n, value);
#endif
}

void fill_cpu_uint16(uint16_t* data, size_t n, uint16_t value)
{
#ifdef ARM_SIMD
  SIMD_FILL(uint16x8_t, vdupq_n_u16, 8, vst1q_u16)
#else
  fill_cpu<uint16_t>(data, n, value);
#endif
}

void fill_cpu_uint8(uint8_t* data, size_t n, uint8_t value)
{
#ifdef ARM_SIMD
  SIMD_FILL(uint8x16_t, vdupq_n_u8, 16, vst1q_u8)
#else
  fill_cpu<uint8_t>(data, n, value);
#endif
}

void fill_cpu_bool(bool* data, size_t n, bool value)
{
  fill_cpu<bool>(data, n, value);
}