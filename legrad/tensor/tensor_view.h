#pragma once

#include <numeric>

#include "internal/array_view.h"
#include "internal/view_pack.h"

namespace legrad
{
using ShapeAndStride = internal::view_pack;

class TensorView
{
public:
  TensorView(IntArrayView shape, IntArrayView stride, Int offset = 0);

  /**
   * Return a reference to the shape of this tensor.  This reference remains
   * valid as long as the tensor is live and not resized.
   */
  IntArrayView shape() const { return view_.shape_view(); }

  /**
   * Return a reference to the strides of this tensor.  This reference remains
   * valid as long as the tensor is live and not restrided.
   */
  IntArrayView stride() const { return view_.stride_view(); }

  Int numel()
  {
    if (numel_ == -1) {
      numel_ = std::accumulate(view_.shape_begin(), view_.shape_end(), 1,
                               std::multiplies<Int>());
    }
    return numel_;
  }

  Int dim() { return view_.dim(); }

  Int shape_at() const
  {
    // TODO:
  }

  Int stride_at() const
  {
    // TODO:
  }

  /**
   * A tensor is contiguous if its elements are stored sequentially in memory
   * without gaps. This is true if the strides match the computed strides
   * for the current shape. Empty tensors are considered contiguous.
   */
  bool is_contiguous() const;

  /**
   * Converts N-dimensional indices to a linear memory offset using the tensor's
   * strides and base offset. Handles negative indices by wrapping around
   * (e.g., -1 refers to the last element in that dimension).
   */
  Size get_indices_offset(IntArrayView indices) const;

  /*
   * Creates a new view of the tensor with dimensions reordered according
   * to the permutation specified in new_axis. For example, permuting a
   * 3D tensor with new_axis={2,0,1} would transform shape (A,B,C) to (C,A,B).
   */
  TensorView permute(IntArrayView new_axis) const;

  /*
   * Implements numpy-style broadcasting, where a dimension can be expanded
   * if it's either the same size or has size 1. When broadcasting a dimension
   * of size 1, the stride for that dimension becomes 0 to reuse the same value.
   */
  TensorView expand(IntArrayView new_shape) const;

  /*
   * Shrink shape of Tensor
   * Note that we want to support negative slice too
   * E.g: tensor[2:-1] is valid
   */
  TensorView shrink(Int2DArrayView args) const;

  /*
   * TODO: Add description here
   */
  TensorView strided(Int2DArrayView args) const;

  /*
   * Adds padding before and after each dimension of the tensor.
   */
  TensorView pad(Int2DArrayView args) const;

  /*
   * Creates a new view of the tensor with a different shape but the same total
   * number of elements. The tensor must be contiguous in memory (or have only 1
   * element).
   */
  TensorView reshape(IntArrayView new_shape) const;

private:
  bool is_contiguous_;
  Size offset_{0};
  Int numel_{-1};

  /*
   * Note that in view_ we store shape (and stride) left to right.
   * But for the correct Tensor shape (or stride),
   * we must store from right to left.
   * E.g:
   * - Shape = (2, 3, 1) => 2 depth, 3 row, 1 height
   * - Stride = (6, 3, 1)
   * - ShapeAndStride = [1, 3, 2, 0, 0, 1, 3, 6, 0, 0]
   */
  ShapeAndStride view_;
};
};  // namespace legrad