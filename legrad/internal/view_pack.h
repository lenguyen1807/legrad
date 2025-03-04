#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "internal/array_view.h"
#include "macros/log.h"

namespace legrad
{
using Size = size_t;
using Int = int64_t;

namespace internal
{
static constexpr Int LEGRAD_VIEW_PACK_MAX_DIM = 5;

/*
 * view_pack is a class inspired (a lot) by PyTorch's `SizesAndStrides`.
 * Memory Layout:
 * - For tensors with dimension <= LEGRAD_VIEW_PACK_MAX_DIM (e.g., 5):
 *   [shape[0], ..., shape[4], stride[0], ..., stride[4]] - Stored inline
 * - For tensors with dimension > LEGRAD_VIEW_PACK_MAX_DIM:
 *   Out-of-line storage (dynamically allocated array) is used to store
 *   shape and stride data contiguously.
 */
class view_pack
{
public:
  ~view_pack()
  {
    if (!is_inline()) {
      std::free(out_of_line_storage_);
    }
  }

  view_pack()
      : dim_(1)
  {
    inline_storage_.fill(0);
  }

  view_pack(Size size)
      : dim_(size)
  {
    if (is_inline()) {
      inline_storage_.fill(0);
    } else {
      out_of_line_storage_ = allocate_new_storage(dim_);
      std::fill_n(out_of_line_storage_, dim_ * 2, 0);
    }
  }

  view_pack(const view_pack&);
  view_pack& operator=(const view_pack&);
  view_pack(view_pack&&) noexcept;
  view_pack& operator=(view_pack&&) noexcept;

  IntArrayView shape_view() const noexcept { return {shape_data(), dim()}; }
  IntArrayView stride_view() const noexcept { return {stride_data(), dim()}; }

  const Int* shape_data() const noexcept
  {
    return is_inline() ? &inline_storage_[0] : &out_of_line_storage_[0];
  }

  Int* shape_data() noexcept
  {
    return is_inline() ? &inline_storage_[0] : &out_of_line_storage_[0];
  }

  const Int* stride_data() const noexcept
  {
    return is_inline() ? &inline_storage_[LEGRAD_VIEW_PACK_MAX_DIM]
                       : &out_of_line_storage_[dim()];
  }

  Int* stride_data() noexcept
  {
    return is_inline() ? &inline_storage_[LEGRAD_VIEW_PACK_MAX_DIM]
                       : &out_of_line_storage_[dim()];
  }

  Int* shape_begin() { return shape_data(); }
  const Int* shape_begin() const { return shape_data(); }
  Int* shape_end() { return shape_data() + dim(); }
  const Int* shape_end() const { return shape_data() + dim(); }
  Int* stride_begin() { return stride_data(); }
  const Int* stride_begin() const { return stride_data(); }
  Int* stride_end() { return stride_data() + dim(); }
  const Int* stride_end() const { return stride_data() + dim(); }

  Int shape_at(Size idx) const
  {
    LEGRAD_ASSERT(idx < dim_, "Index {} is out of range [0:{}) for shape", idx,
                  dim_);
    return unsafe_shape_at(idx);
  }

  Int stride_at(Size idx) const
  {
    LEGRAD_ASSERT(idx < dim_, "Index {} is out of range [0:{}) for stride", idx,
                  dim_);
    return unsafe_stride_at(idx);
  }

  void set_shape(IntArrayView shape)
  {
    resize_storage(shape.size());
    std::copy(shape.begin(), shape.end(), shape_begin());
  }

  void set_stride(IntArrayView stride)
  {
    if (stride.size() != dim_) {
      LEGRAD_THROW_ERROR(std::invalid_argument,
                         "New stride is not match with current shape size", 0);
    }
    std::copy(stride.begin(), stride.end(), stride_begin());
  }

  void resize_storage(Size new_dim);
  bool is_inline() const { return dim_ <= LEGRAD_VIEW_PACK_MAX_DIM; }
  Size dim() const noexcept { return dim_; }

private:
  Int& unsafe_stride_at(Size idx) { return stride_data()[idx]; }
  Int& unsafe_shape_at(Size idx) { return shape_data()[idx]; }
  Int unsafe_stride_at(Size idx) const { return stride_data()[idx]; }
  Int unsafe_shape_at(Size idx) const { return shape_data()[idx]; }

  void resize_out_of_line_storage(Size new_dim, Size old_dim);
  void move_out_to_inline_storage(Size new_dim, Size old_dim);
  void move_inline_to_out_storage(Size new_dim, Size old_dim);

  Int* allocate_new_storage(Size n);
  void reallocate_out_of_line_storage(Size n);
  Int storage_bytes(Size n) noexcept { return n * 2 * sizeof(Int); }

  void copy_inline_storage(const view_pack& other)
  {
    std::copy(other.inline_storage_.begin(), other.inline_storage_.end(),
              inline_storage_.data());
  }

  void copy_outline_storage(const view_pack& other)
  {
    std::copy(other.out_of_line_storage_,
              other.out_of_line_storage_ + other.dim() * 2,
              out_of_line_storage_);
  }

private:
  Size dim_;
  union
  {
    std::array<Int, LEGRAD_VIEW_PACK_MAX_DIM * 2> inline_storage_;
    Int* out_of_line_storage_;
  };
};
}  // namespace internal
}  // namespace legrad