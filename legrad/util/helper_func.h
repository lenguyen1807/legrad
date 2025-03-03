#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "macros/expr.h"

namespace legrad::util
{
template <typename T>
LEGRAD_INLINE std::string vec2str(const std::vector<T>& vec)
{
  if (vec.empty()) {
    return "()";
  }

  // HACK
  std::string result = "(";
  for (size_t i = 0; i < vec.size() - 1; ++i) {
    result += std::to_string(vec[i]) + ",";
  }
  result += std::to_string(vec[vec.size() - 1]) + ")";

  return result;
}

template <typename T>
LEGRAD_INLINE std::string vec2strAcc(const std::vector<std::vector<T>>& args)
{
  std::string result = "";
  for (const auto& arg : args) {
    result += vec2str(arg);
  }
  return result;
}

LEGRAD_INLINE std::string to_lower(const std::string& str)
{
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  return result;
}
}  // namespace legrad::util