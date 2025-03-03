#include <iostream>

#include "legrad/core/device.h"
#include "legrad/internal/view_pack.h"
#include "legrad/tensor/tensor_view.h"
#include "legrad/util/helper_func.h"

int main()
{
  legrad::ShapeAndStride view1(3);
  legrad::ShapeAndStride view2(5);
  legrad::ShapeAndStride view3(7);
  view1.set_shape({1, 2, 3});
  view2.set_shape({1, 2, 3, 4, 5});
  std::cout << view1.numel() << "\n";
  std::cout << view2.numel() << "\n";
  return 0;
}