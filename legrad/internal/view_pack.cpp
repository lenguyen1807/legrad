#include <functional>
#include <numeric>

#include "view_pack.h"

namespace legrad::internal
{
Size view_pack::numel()
{
  // calculate number of elements if it's not available
  if (numel_ == -1) {
    numel_ =
        std::accumulate(shape_begin(), shape_end(), 1, std::multiplies<Int>());
  }
  return numel_;
}

void view_pack::resize_storage(Size new_ndim)
{
  if (new_ndim == dim()) {
    return;
  }

  /*
   * In this resize function, we will have 2 path:
   * - Path 1: We currently using in range storage, new dim is still in
   * range and new dim is larger than old dim (fastest path)
   * - Path 2: Other cases (which is much more expensive and slower)
   */
  const Size old_ndim = ndim_;
  if (LEGRAD_LIKELY(new_ndim <= LEGRAD_MAX_PACK_VIEW_SIZE && is_in_range())) {
    if (ndim_ < new_ndim) {
      Size range = new_ndim - old_ndim;
      std::memset(&in_range_storage_[old_ndim], 0, range * sizeof(Int));
      std::memset(&in_range_storage_[old_ndim + LEGRAD_MAX_PACK_VIEW_SIZE], 0,
                  range * sizeof(Int));
    }
    /*
     * for the case new dim < old dim
     * we just skip it, but why ?
     * let see we have an array [1, 2, 3]
     * and we have new size which is 2
     * then we never need to access the number 3
     */
  } else {
    slower_resize(new_ndim, old_ndim);
  }
}

void view_pack::slower_resize(Size new_ndim, Size old_ndim)
{
  // Case 1: Transitioning from out-of-range to in-range storage
  if (new_ndim <= LEGRAD_MAX_PACK_VIEW_SIZE) {
    // we dont need out of range storage anymore
    delete out_of_range_storage_;
    return;
  }

  // Case 2: Transitioning from in-range to out-of-range
  if (is_in_range()) {
  }
  // Case 3: Already out-of-range, just resizing out-of-range storage
  else
  {
  }
}
}  // namespace legrad::internal