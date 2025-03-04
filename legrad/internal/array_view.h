#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <string>
#include <type_traits>

#include "macros/log.h"

namespace legrad
{
namespace internal
{
template <typename T>
class array_view
{
  using iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;

public:
  array_view(const T* ptr, std::size_t len) noexcept
      : ptr_(ptr)
      , len_(len)
  {
  }

  array_view(const std::initializer_list<T>& list)
      : ptr_(std::begin(list) == std::end(list) ? static_cast<T*>(nullptr)
                                                : std::begin(list))
      , len_(list.size())
  {
  }

  template <typename A>
  array_view(const std::vector<T, A>& vec)
      : ptr_(vec.data())
      , len_(vec.size())
  {
    static_assert(!std::is_same_v<T, bool>,
                  "Cannot initialize array_view with std::vector<bool>");
  }

  /*
   * What is the point of these two template ?
   * We need to delete the assignment of array_view
   * so array_view = ... (will be error).
   * But why we don't just use array_view<T>& operator(V&&) = delete ?
   * Because we want array_view = {} valid.
   */
  template <typename U, typename = std::enable_if_t<std::is_same_v<U, T>>>
  array_view<T>& operator=(U&&) = delete;
  template <typename U, typename = std::enable_if_t<std::is_same_v<U, T>>>
  array_view<T>& operator=(std::initializer_list<U>) = delete;

  constexpr const T& operator[](size_t idx) const { return ptr_[idx]; }

  const T& at(size_t idx) const
  {
    LEGRAD_ASSERT(idx < len_ && idx >= 0,
                  "Index {} is not valid for array_view with length {}", idx,
                  len_);
    return ptr_[idx];
  }

  const T* data() const noexcept { return ptr_; }
  size_t size() const noexcept { return len_; }
  bool empty() const { return len_ == 0; }

  constexpr iterator begin() const noexcept { return ptr_; }
  constexpr iterator end() const noexcept { return ptr_ + len_; }
  constexpr const iterator cbegin() const noexcept { return ptr_; }
  constexpr const iterator cend() const noexcept { return ptr_ + len_; }
  constexpr reverse_iterator rbegin() const noexcept
  {
    return std::make_reverse_iterator(end());
  }
  constexpr reverse_iterator rend() const noexcept
  {
    return std::make_reverse_iterator(begin());
  }

  const T& front() const
  {
    LEGRAD_ASSERT(!empty(), "Attempt to access empty array view", 0);
    return at(0);
  }

  const T& back() const
  {
    LEGRAD_ASSERT(!empty(), "Attempt to access empty array view", 0);
    return at(size() - 1);
  }

  array_view<T> slice(size_t start, size_t end)
  {
    LEGRAD_ASSERT(end - start <= size(),
                  "Invalid slice with start {} and end {}", start, end);
    return {data() + start, end - start};
  }

  std::vector<T> to_vec() const
  {
    // Note that this is really expensive
    // Because vector will copy the data (to make it ownership)
    return std::vector<T>(ptr_, ptr_ + len_);
  }

  friend std::ostream& operator<<(std::ostream& os, array_view view)
  {
    os << view_to_str(view);
    return os;
  }

  static std::string view_to_str(internal::array_view<T> view)
  {
    LEGRAD_DEFAULT_ASSERT(std::is_arithmetic_v<T>);

    if (view.size() == 0) {
      return "()";
    }

    std::string result = "(";
    for (size_t i = 0; i < view.size() - 1; ++i) {
      result += std::to_string(view[i]) + ",";
    }
    result += std::to_string(view[view.size() - 1]) + ")";

    return result;
  }

  bool equals(array_view other)
  {
    return other.len_ == len_ && std::equal(begin(), end(), other.begin());
  }

  bool equals(const std::initializer_list<T>& other)
  {
    return equals(array_view<T>(other));
  }

  friend bool operator==(internal::array_view<T> lhs,
                         internal::array_view<T> rhs)
  {
    return lhs.equals(rhs);
  }

  friend bool operator!=(internal::array_view<T> lhs,
                         internal::array_view<T> rhs)
  {
    return !(lhs == rhs);
  }

  friend bool operator==(const std::vector<T>& lhs, internal::array_view<T> rhs)
  {
    return internal::array_view<T>(lhs).equals(rhs);
  }

  friend bool operator==(internal::array_view<T> lhs, const std::vector<T>& rhs)
  {
    return rhs == lhs;
  }

  friend bool operator!=(const std::vector<T>& lhs, internal::array_view<T> rhs)
  {
    return !(lhs == rhs);
  }

  friend bool operator!=(internal::array_view<T> lhs, const std::vector<T>& rhs)
  {
    return rhs != lhs;
  }

private:
  const T* ptr_;
  size_t len_;
};
}  // namespace internal

using IntArrayView = internal::array_view<int64_t>;
using Int2DArrayView = internal::array_view<internal::array_view<int64_t>>;
};  // namespace legrad