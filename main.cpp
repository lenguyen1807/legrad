#include <iostream>
#include <stdexcept>

#include "fmt/core.h"
#include "fmt/format.h"
#include "legrad/core/device.h"
#include "legrad/internal/array_view.h"
#include "legrad/internal/view_pack.h"
#include "legrad/tensor/tensor_view.h"
#include "legrad/util/helper_func.h"
#include "macros/log.h"

int main()
{
  legrad::ShapeAndStride vp(5);
  std::cout << vp.shape_view() << "\n";
  std::cout << vp.stride_view() << "\n";

  legrad::internal::array_view<int> view = {1, 2, 3};
  std::cout << view.at(0) << "\n";
  std::cout << view.at(1) << "\n";
  std::cout << view.at(2) << "\n";
  std::cout << view << "\n";

  return 0;
}