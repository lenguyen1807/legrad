#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

#include "macros/expr.h"
#include "macros/log.h"

namespace legrad::util
{
LEGRAD_INLINE std::string to_lower(const std::string& str)
{
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  return result;
}

template <typename T>
LEGRAD_INLINE T _wrap_dim(T dim, T rank)
{
  if (LEGRAD_LIKELY(rank * T{-1} <= dim && dim < rank)) {
    if (dim <= 0) {
      return dim + rank;
    }
    return dim;
  } else {  // If cannot hit the first case, should throw error
    LEGRAD_CHECK_AND_THROW(rank >= 0, std::runtime_error,
                           "Rank cannot be negative", 0);

    LEGRAD_CHECK_AND_THROW(
        rank != 0, std::runtime_error,
        "Dimension {} is specified but the Tensor is empty (has 0 rank)", dim);

    T min = rank * T{-1};
    T max = rank - T{1};

    LEGRAD_CHECK_AND_THROW(min <= dim && dim <= max, std::runtime_error,
                           "Dimension {} is out of range [{}, {}]", dim, min,
                           max);

    LEGRAD_THROW_ERROR(std::logic_error, "All path above should been hit", 0);
  }
}

LEGRAD_INLINE int64_t maybe_wrap_dim(int64_t dim, int64_t rank)
{
  return _wrap_dim(dim, rank);
}
}  // namespace legrad::util