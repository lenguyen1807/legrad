#ifndef CPU_FILL_DATA_H
#define CPU_FILL_DATA_H

#include <cstddef>
#include <cstdint>

void fill_cpu_float32(float* data, size_t n, float value);  // done
void fill_cpu_float64(double* data, size_t n, double value);  // done
void fill_cpu_int64(int64_t* data, size_t n, int64_t vlaue);  // done
void fill_cpu_int32(int32_t* data, size_t n, int32_t value);  // done
void fill_cpu_int16(int16_t* data, size_t n, int16_t value);  // done
void fill_cpu_int8(int8_t* data, size_t n, int8_t value);  // done
void fill_cpu_uint64(uint64_t* data, size_t n, uint64_t value);  // done
void fill_cpu_uint32(uint32_t* data, size_t n, uint32_t value);  // done
void fill_cpu_uint16(uint16_t* data, size_t n, uint16_t value);  // done
void fill_cpu_uint8(uint8_t* data, size_t n, uint8_t value);  // done
void fill_cpu_bool(bool* data, size_t n, bool value);

template <typename T>
inline void fill_cpu(T* data, size_t n, T value)
{
  size_t i = 0;
  for (; i < n; ++i) {
    data[i] = value;
  }
}

#endif  // FILL_DATA_H