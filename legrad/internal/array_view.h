/* From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

// legrad: modified from c10::ArrayRef
// - simplified implementation

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
  array_view()
      : ptr_(nullptr)
      , len_(0)
  {
  }

  array_view(const T& elem)
      : ptr_(&elem)
      , len_(1)
  {
  }

  array_view(const T* ptr, std::size_t len) noexcept
      : ptr_(ptr)
      , len_(len)
  {
  }

  array_view(const T* start, const T* end)
      : ptr_(start)
      , len_(end - start)
  {
  }

  template <typename Container,
            typename = std::enable_if_t<std::is_same<
                std::remove_const_t<decltype(std::declval<Container>().data())>,
                T*>::value>>
  array_view(const Container& container)
      : ptr_(container.data())
      , len_(container.size())
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
                  "Index {} is out of bounds with range [0, {})", idx, len_);
    return ptr_[idx];
  }

  const T* data() const noexcept { return ptr_; }
  size_t size() const noexcept { return len_; }
  bool empty() const noexcept { return len_ == 0; }

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
    // Because vector will copy the data (to make its ownership)
    return std::vector<T>(ptr_, ptr_ + len_);
  }

  friend std::ostream& operator<<(std::ostream& os, array_view view)
  {
    os << array_view::view_to_str(view);
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