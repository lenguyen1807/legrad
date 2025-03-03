#pragma once

#include <cstddef>
#include <tuple>

#include "boost/functional/hash.hpp"

namespace legrad::util
{
/*
 * We cannot use std::pair as std::unordered_map's key directly, we need to have
 * a hash function for the key
 * https://stackoverflow.com/questions/20590656/how-to-solve-error-for-hash-function-of-pair-of-ints-in-unordered-map
 */
struct HashPairKey
{
  template <typename T, typename U>
  size_t operator()(const std::pair<T, U>& x) const
  {
    size_t seed = 0;
    boost::hash_combine(seed, x.first);
    boost::hash_combine(seed, x.second);
    return seed;
  }
};

/*
 * HACK: Ugly hack to iterate in tuple
 * https://stackoverflow.com/questions/63485835/how-can-you-iterate-over-elements-of-a-stdtuple-with-a-shared-base-class
 */
template <typename Tuple, typename Callable>
void iterate_tuple(Tuple&& t, Callable c)
{
  std::apply([&](auto&&... args) { (c(args), ...); }, t);
}

struct HashTupleKey
{
  template <typename... T>
  size_t operator()(const std::tuple<T...>& x) const
  {
    size_t seed = 0;
    iterate_tuple(x, [&seed](auto& arg) { boost::hash_combine(seed, arg); });
    return seed;
  }
};
}  // namespace legrad::util