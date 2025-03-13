#include <stdexcept>

#include "macros/log.h"
#include "tensor_view.h"

using namespace legrad;

bool TensorView::is_contiguous() const
{
  /*
   * A Tensor is contiguous when:
   * - It has no shape (means empty)
   * - It shape has zero in it (means empty too)
   * - The stride is contiguous
   */
}

Size TensorView::get_indices_offset(IntArrayView indices) const
{
  // TODO:
}

TensorView TensorView::permute(IntArrayView new_axis) const
{
  // TODO:
}

TensorView TensorView::expand(IntArrayView new_shape) const
{
  // TODO:
}

TensorView TensorView::shrink(Int2DArrayView args) const
{
  // TODO:
}

TensorView TensorView::strided(Int2DArrayView args) const
{
  // TODO:
}

TensorView TensorView::pad(Int2DArrayView args) const
{
  // TODO:
}

TensorView TensorView::reshape(IntArrayView new_shape) const
{
  // TODO:
}