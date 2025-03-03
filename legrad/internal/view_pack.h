#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "macros/expr.h"
#include "macros/log.h"

namespace legrad
{
using Size = size_t;
using Int = int64_t;
using IntIterator = Int*;
using ConstIntIterator = const Int*;
using IntList = std::initializer_list<Int>;
template <size_t N>
using IntArray = std::array<Int, N>;

namespace internal
{
constexpr Int LEGRAD_MAX_PACK_VIEW_SIZE = 5;

/*
 * view_pack is a class inspired by SizesAndStrides in Pytorch C10
 * Instead of using std::vector to store stride and shape
 * seperately, we can pack it in one array, the format is blow
 * [shape[0], ..., shape[4], stride[0], ..., stride[4]]
 * but for larger tensor with larger dimension (> 5)
 * then we can use out_range_storage (dynamically arrayor vector)
 */
class view_pack
{
public:
  view_pack()
      : ndim_(1)
  {
    unsafe_shape_at(0) = 0;
    unsafe_stride_at(0) = 1;
  }

  view_pack(Size size)
      : ndim_(size)
  {
    if (is_in_range()) {
      in_range_storage_.fill(0);
    } else {
      out_of_range_storage_ = new Int[ndim_ * 2]{0};
    }
  }

  view_pack(const IntList& shape)
      : ndim_(shape.size())
  {
    set_shape(shape);
  }

  ~view_pack()
  {
    if (!is_in_range()) {
      delete[] out_of_range_storage_;
    }
  }

  IntIterator shape_data()
  {
    if (LEGRAD_LIKELY(is_in_range())) {
      return &in_range_storage_[0];
    } else {
      return &out_of_range_storage_[0];
    }
  }

  ConstIntIterator shape_data() const
  {
    if (LEGRAD_LIKELY(is_in_range())) {
      return &in_range_storage_[0];
    } else {
      return &out_of_range_storage_[0];
    }
  }

  IntIterator stride_data()
  {
    if (LEGRAD_LIKELY(is_in_range())) {
      return &in_range_storage_[LEGRAD_MAX_PACK_VIEW_SIZE];
    } else {
      return &out_of_range_storage_[dim()];
    }
  }

  ConstIntIterator stride_data() const
  {
    if (LEGRAD_LIKELY(is_in_range())) {
      return &in_range_storage_[LEGRAD_MAX_PACK_VIEW_SIZE];
    } else {
      return &out_of_range_storage_[dim()];
    }
  }

  Int shape_at(Size idx) const
  {
    LEGRAD_ASSERT(idx < ndim_, "Index {} is out of range [0:{}) for shape", idx,
                  ndim_);
    return unsafe_shape_at(idx);
  }

  Int stride_at(Size idx) const
  {
    LEGRAD_ASSERT(idx < ndim_, "Index {} is out of range [0:{}) for stride",
                  idx, ndim_);
    return unsafe_stride_at(idx);
  }

  IntIterator shape_begin() { return shape_data(); }
  ConstIntIterator shape_begin() const { return shape_data(); }
  IntIterator shape_end() { return shape_data() + dim(); }
  ConstIntIterator shape_end() const { return shape_data() + dim(); }
  IntIterator stride_begin() { return stride_data(); }
  ConstIntIterator stride_begin() const { return stride_data(); }
  IntIterator stride_end() { return stride_data() + dim(); }
  ConstIntIterator stride_end() const { return stride_data() + dim(); }

  Size dim() const { return ndim_; }

  Size numel();

  void set_shape(const IntList& shape)
  {
    resize_storage(shape.size());
    std::copy(shape.begin(), shape.end(), shape_begin());
  }

  void set_stride(const IntList& stride)
  {
    resize_storage(stride.size());
    std::copy(stride.begin(), stride.end(), stride_begin());
  }

  void resize_storage(Size new_ndim);

private:
  Int& unsafe_stride_at(Size idx) { return stride_data()[idx]; }
  Int& unsafe_shape_at(Size idx) { return shape_data()[idx]; }
  Int unsafe_stride_at(Size idx) const { return stride_data()[idx]; }
  Int unsafe_shape_at(Size idx) const { return shape_data()[idx]; }

  bool is_in_range() const { return ndim_ <= LEGRAD_MAX_PACK_VIEW_SIZE; }

  void slower_resize(Size new_ndim, Size old_ndim);

private:
  Int numel_ = -1;
  Size ndim_;
  union
  {
    IntArray<LEGRAD_MAX_PACK_VIEW_SIZE * 2> in_range_storage_;
    IntIterator out_of_range_storage_;
  };
};
}  // namespace internal
};  // namespace legrad