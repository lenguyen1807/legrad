#include <catch2/catch_test_macros.hpp>
#include <internal/view_pack.h>

using namespace legrad;
using namespace legrad::internal;

/*
 * I don't know what to say
 * But thanks Gemini for generating test cases for me
 */

TEST_CASE("view_pack Construction and Basic Access", "[view_pack]")
{
  SECTION("Default constructor")
  {
    view_pack vp;
    REQUIRE(vp.dim() == 1);
    REQUIRE(vp.shape_at(0) == 0);
    REQUIRE(vp.stride_at(0) == 0);
  }

  SECTION("Size constructor - inline")
  {
    view_pack vp(3);
    REQUIRE(vp.dim() == 3);
    for (size_t i = 0; i < 3; ++i) {
      REQUIRE(vp.shape_at(i) == 0);  // Default initialized to 0
      REQUIRE(vp.stride_at(i) == 0);  // Default initialized to 0
    }
  }

  SECTION("Size constructor - out-of-line")
  {
    view_pack vp(LEGRAD_VIEW_PACK_MAX_DIM + 2);
    REQUIRE(vp.dim() == LEGRAD_VIEW_PACK_MAX_DIM + 2);
    for (size_t i = 0; i < LEGRAD_VIEW_PACK_MAX_DIM + 2; ++i) {
      REQUIRE(vp.shape_at(i) == 0);  // Default initialized to 0
      REQUIRE(vp.stride_at(i) == 0);  // Default initialized to 0
    }
  }

  SECTION("Copy constructor")
  {
    view_pack vp(LEGRAD_VIEW_PACK_MAX_DIM + 2);
    vp.set_shape({1, 2, 3, 4, 5, 6, 7});
    REQUIRE(vp.dim() == LEGRAD_VIEW_PACK_MAX_DIM + 2);

    view_pack vp2 = vp;
    REQUIRE(!vp2.is_inline());
    REQUIRE(vp2.dim() == LEGRAD_VIEW_PACK_MAX_DIM + 2);
    REQUIRE(vp2.shape_view().equals({1, 2, 3, 4, 5, 6, 7}));
    REQUIRE(vp2.shape_view() == vp.shape_view());
  }

  SECTION("Copy assignment")
  {
    view_pack vp(LEGRAD_VIEW_PACK_MAX_DIM + 1);
    vp.set_shape({1, 2, 3, 4, 5, 6});
    REQUIRE(vp.dim() == LEGRAD_VIEW_PACK_MAX_DIM + 1);
    REQUIRE(!vp.is_inline());

    view_pack vp2(3);
    vp2.set_shape({2, 3, 4});
    vp = vp2;
    REQUIRE(vp.dim() == 3);
    REQUIRE(vp.is_inline());
    REQUIRE(vp.shape_view().equals({2, 3, 4}));
  }

  SECTION("Move constructor - Inline Storage")
  {
    view_pack original_inline;
    original_inline.resize_storage(3);
    view_pack moved_inline(std::move(original_inline));
    REQUIRE(original_inline.dim() == 0);
  }

  SECTION("Move constructor - Out-of-line Storage")
  {
    view_pack original_inline;
    original_inline.resize_storage(8);
    view_pack moved_inline(std::move(original_inline));
    REQUIRE(original_inline.dim() == 0);
  }

  SECTION("Move assignment - Inline Storage")
  {
    view_pack obj1_inline;
    obj1_inline.resize_storage(3);
    view_pack obj2_inline = std::move(obj1_inline);
    REQUIRE(obj1_inline.dim() == 0);
  }

  SECTION("Move assignment - Outline Storage")
  {
    view_pack obj1_inline, obj2_outline;
    obj1_inline.resize_storage(8);
    obj2_outline = std::move(obj1_inline);
    REQUIRE(obj1_inline.dim() == 0);
  }
}

TEST_CASE("view_pack Set Shape and Stride", "[view_pack]")
{
  SECTION("Set shape - inline")
  {
    view_pack vp(3);
    vp.set_shape({2, 3, 4});
    REQUIRE(vp.dim() == 3);
    REQUIRE(vp.shape_at(0) == 2);
    REQUIRE(vp.shape_at(1) == 3);
    REQUIRE(vp.shape_at(2) == 4);
  }

  SECTION("Set shape - out-of-line")
  {
    view_pack vp(LEGRAD_VIEW_PACK_MAX_DIM + 2);
    vp.set_shape({2, 3, 4, 5, 6, 7});
    REQUIRE(vp.dim() == 6);
    REQUIRE(vp.shape_at(0) == 2);
    REQUIRE(vp.shape_at(5) == 7);
  }

  SECTION("Set stride - inline")
  {
    view_pack vp(3);
    vp.set_stride({12, 4, 1});
    REQUIRE(vp.dim() == 3);
    REQUIRE(vp.stride_at(0) == 12);
    REQUIRE(vp.stride_at(1) == 4);
    REQUIRE(vp.stride_at(2) == 1);
  }

  SECTION("Set stride - out-of-line")
  {
    view_pack vp(LEGRAD_VIEW_PACK_MAX_DIM + 1);
    vp.set_stride({720, 120, 24, 6, 1, 0});
    REQUIRE(vp.dim() == 6);
    REQUIRE(vp.stride_at(0) == 720);
    REQUIRE(vp.stride_at(5) == 0);
  }

  SECTION("Set shape then stride - inline")
  {
    view_pack vp(3);
    vp.set_shape({2, 3, 4});
    vp.set_stride({12, 4, 1});
    REQUIRE(vp.shape_at(0) == 2);
    REQUIRE(vp.stride_at(0) == 12);
  }

  SECTION("Set shape then stride - out-of-line")
  {
    view_pack vp(LEGRAD_VIEW_PACK_MAX_DIM + 2);
    vp.set_shape({2, 3, 4, 5, 6, 7});
    vp.set_stride({720, 120, 24, 6, 1, 0});
    REQUIRE(vp.dim() == 6);
    REQUIRE(vp.shape_at(0) == 2);
    REQUIRE(vp.stride_at(5) == 0);
  }
}

TEST_CASE("view_pack Resize Storage", "[view_pack]")
{
  SECTION("Resize - inline to inline (smaller)")
  {
    view_pack vp(5);
    vp.set_shape({1, 2, 3, 4, 5});
    vp.resize_storage(3);
    REQUIRE(vp.dim() == 3);
    REQUIRE(vp.shape_at(0) == 1);
    REQUIRE(vp.shape_at(1) == 2);
    REQUIRE(vp.shape_at(2) == 3);
  }

  SECTION("Resize - inline to inline (larger)")
  {
    view_pack vp(2);
    vp.set_shape({2, 3});
    vp.resize_storage(4);
    REQUIRE(vp.dim() == 4);
    REQUIRE(vp.shape_at(0) == 2);
    REQUIRE(vp.shape_at(1) == 3);
    REQUIRE(vp.shape_at(2) == 0);  // New dims are zero-initialized
    REQUIRE(vp.shape_at(3) == 0);
  }

  SECTION("Resize - inline to out-of-line")
  {
    view_pack vp(LEGRAD_VIEW_PACK_MAX_DIM);
    vp.set_shape({1, 2, 3, 4, 5});
    vp.resize_storage(LEGRAD_VIEW_PACK_MAX_DIM + 2);
    REQUIRE(vp.dim() == LEGRAD_VIEW_PACK_MAX_DIM + 2);
    REQUIRE(vp.shape_at(0) == 1);
    REQUIRE(vp.shape_at(LEGRAD_VIEW_PACK_MAX_DIM - 1) == 5);
    REQUIRE(vp.shape_at(LEGRAD_VIEW_PACK_MAX_DIM)
            == 0);  // New dims are zero-initialized
    REQUIRE(vp.shape_at(LEGRAD_VIEW_PACK_MAX_DIM + 1) == 0);
  }

  SECTION("Resize - out-of-line to out-of-line (smaller)")
  {
    view_pack vp(LEGRAD_VIEW_PACK_MAX_DIM + 3);
    vp.set_shape({1, 2, 3, 4, 5, 6, 7, 8});
    vp.resize_storage(LEGRAD_VIEW_PACK_MAX_DIM + 1);
    REQUIRE(vp.dim() == LEGRAD_VIEW_PACK_MAX_DIM + 1);
    REQUIRE(vp.shape_at(0) == 1);
    REQUIRE(vp.shape_at(LEGRAD_VIEW_PACK_MAX_DIM) == 6);
  }

  SECTION("Resize - out-of-line to out-of-line (larger)")
  {
    view_pack vp(LEGRAD_VIEW_PACK_MAX_DIM + 1);
    vp.set_shape({1, 2, 3, 4, 5, 6});
    vp.resize_storage(LEGRAD_VIEW_PACK_MAX_DIM + 3);
    REQUIRE(vp.dim() == LEGRAD_VIEW_PACK_MAX_DIM + 3);
    REQUIRE(vp.shape_at(0) == 1);
    REQUIRE(vp.shape_at(LEGRAD_VIEW_PACK_MAX_DIM) == 6);
    REQUIRE(vp.shape_at(LEGRAD_VIEW_PACK_MAX_DIM + 1)
            == 0);  // New dims are zero-initialized
    REQUIRE(vp.shape_at(LEGRAD_VIEW_PACK_MAX_DIM + 2) == 0);
  }

  SECTION("Resize - out-of-line to inline")
  {
    view_pack vp(LEGRAD_VIEW_PACK_MAX_DIM + 1);
    vp.set_shape({1, 2, 3, 4, 5, 6});
    vp.resize_storage(LEGRAD_VIEW_PACK_MAX_DIM - 1);
    REQUIRE(vp.dim() == LEGRAD_VIEW_PACK_MAX_DIM - 1);
    REQUIRE(vp.is_inline());  // Should transition to inline
    REQUIRE(vp.shape_at(0) == 1);
    REQUIRE(vp.shape_at(LEGRAD_VIEW_PACK_MAX_DIM - 2) == 4);
  }
}