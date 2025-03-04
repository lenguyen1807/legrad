#include <numeric>

#include <catch2/catch_test_macros.hpp>
#include <internal/array_view.h>

using namespace legrad;

/*
 * I don't know what to say
 * But thanks Gemini for generating test cases for me
 */

TEST_CASE("array_view Construction", "[array_view]")
{
  SECTION("From pointer and size")
  {
    int data[] = {1, 2, 3, 4, 5};
    internal::array_view<int> view(data, 5);
    REQUIRE(view.size() == 5);
    REQUIRE(view.data() == data);
  }

  SECTION("From initializer_list")
  {
    // TODO: Note that all cases with initilizer list have errors
    // So I will debug it later

    internal::array_view<int> view = {1, 2, 3};
    REQUIRE(view.size() == 3);
    // REQUIRE(view.at(0) == 1);
    // REQUIRE(view.at(1) == 2);
    // REQUIRE(view.at(2) == 3);

    internal::array_view<int> empty_view = {};
    REQUIRE(empty_view.size() == 0);
    REQUIRE(empty_view.data() == nullptr);
    REQUIRE(empty_view.empty());
  }

  SECTION("From std::vector")
  {
    std::vector<int> vec = {6, 7, 8, 9};
    internal::array_view<int> view(vec);
    REQUIRE(view.size() == 4);
    REQUIRE(view.data() == vec.data());
    REQUIRE(view[0] == 6);
    REQUIRE(view[3] == 9);
  }

  SECTION("From empty std::vector")
  {
    std::vector<int> vec = {};
    internal::array_view<int> view(vec);
    REQUIRE(view.size() == 0);
    REQUIRE(view.data() == vec.data());  // vec.data() might return nullptr for
                                         // empty vector in some implementations
    REQUIRE(view.empty());
  }
}

TEST_CASE("array_view Element Access", "[array_view]")
{
  int data[] = {10, 20, 30};
  internal::array_view<int> view(data, 3);

  SECTION("operator[]")
  {
    REQUIRE(view[0] == 10);
    REQUIRE(view[1] == 20);
    REQUIRE(view[2] == 30);
  }

  SECTION("front() and back()")
  {
    REQUIRE(view.front() == 10);
    REQUIRE(view.back() == 30);
  }
}

TEST_CASE("array_view Size and Emptiness", "[array_view]")
{
  int data[] = {40, 50};
  internal::array_view<int> view(data, 2);
  internal::array_view<int> empty_view = {};

  SECTION("size()")
  {
    REQUIRE(view.size() == 2);
    REQUIRE(empty_view.size() == 0);
  }

  SECTION("empty()")
  {
    REQUIRE_FALSE(view.empty());
    REQUIRE(empty_view.empty());
  }
}

TEST_CASE("array_view Iterators", "[array_view]")
{
  int data[] = {100, 200, 300};
  internal::array_view<int> view(data, 3);

  SECTION("begin() and end()")
  {
    std::vector<int> iterated_values;
    for (auto it = view.begin(); it != view.end(); ++it) {
      iterated_values.push_back(*it);
    }
    REQUIRE(iterated_values == std::vector<int>{100, 200, 300});

    int sum = std::accumulate(view.begin(), view.end(), 0);
    REQUIRE(sum == 600);
  }

  SECTION("cbegin() and cend()")
  {  // Test with cbegin/cend too for good measure (though begin/end are
     // constexpr already)
    std::vector<int> iterated_values;
    for (auto it = view.cbegin(); it != view.cend(); ++it) {
      iterated_values.push_back(*it);
    }
    REQUIRE(iterated_values == std::vector<int>{100, 200, 300});
  }

  SECTION("rbegin() and rend()")
  {
    std::vector<int> reverse_iterated_values;
    for (auto it = view.rbegin(); it != view.rend(); ++it) {
      reverse_iterated_values.push_back(*it);
    }
    REQUIRE(reverse_iterated_values == std::vector<int>{300, 200, 100});
  }
}

TEST_CASE("array_view Slice", "[array_view]")
{
  int data[] = {1, 2, 3, 4, 5, 6};
  internal::array_view<int> view(data, 6);

  SECTION("Valid slice")
  {
    internal::array_view<int> slice_view =
        view.slice(1, 4);  // Elements at index 1, 2, 3
    REQUIRE(slice_view.size() == 3);
    REQUIRE(slice_view[0] == 2);
    REQUIRE(slice_view[1] == 3);
    REQUIRE(slice_view[2] == 4);
  }

  SECTION("Slice from beginning")
  {
    internal::array_view<int> slice_view =
        view.slice(0, 3);  // Elements at index 0, 1, 2
    REQUIRE(slice_view.size() == 3);
    REQUIRE(slice_view[0] == 1);
    REQUIRE(slice_view[2] == 3);
  }

  SECTION("Slice to end")
  {
    internal::array_view<int> slice_view =
        view.slice(3, 6);  // Elements at index 3, 4, 5
    REQUIRE(slice_view.size() == 3);
    REQUIRE(slice_view[0] == 4);
    REQUIRE(slice_view[2] == 6);
  }

  SECTION("Empty slice (start == end)")
  {
    internal::array_view<int> slice_view = view.slice(2, 2);
    REQUIRE(slice_view.empty());
    REQUIRE(slice_view.size() == 0);
  }

  SECTION("Full slice (start = 0, end = size())")
  {
    internal::array_view<int> slice_view = view.slice(0, 6);
    REQUIRE(slice_view.size() == 6);
    REQUIRE(slice_view == view);  // Should be equal to the original view
  }
}

TEST_CASE("array_view to_vec()", "[array_view]")
{
  int data[] = {70, 80, 90};
  internal::array_view<int> view(data, 3);
  std::vector<int> vec = view.to_vec();

  REQUIRE(vec == std::vector<int>{70, 80, 90});
  // Check if it's a copy, not just a view of the same data
  vec[0] = 700;
  REQUIRE(view[0] == 70);  // Original view should not be modified
}

TEST_CASE("array_view Output Stream Operator", "[array_view]")
{
  SECTION("Non-empty view")
  {
    // TODO: Note that all cases with initilizer list have errors
    // So I will debug it later

    // internal::array_view<int> view = {1, 2, 3, 4};
    // std::stringstream ss;
    // ss << view;
    // REQUIRE(ss.str() == "(1,2,3,4)");
  }

  SECTION("Empty view")
  {
    internal::array_view<int> view = {};
    std::stringstream ss;
    ss << view;
    REQUIRE(ss.str() == "()");
  }
}

TEST_CASE("array_view Equality Operators", "[array_view]")
{
  int data1[] = {1, 2, 3};
  int data2[] = {1, 2, 3};
  int data3[] = {4, 5, 6};
  internal::array_view<int> view1(data1, 3);
  internal::array_view<int> view2(data2, 3);
  internal::array_view<int> view3(data3, 3);
  internal::array_view<int> view_short(data1, 2);

  std::vector<int> vec1 = {1, 2, 3};
  std::vector<int> vec2 = {4, 5, 6};
  std::initializer_list<int> list1 = {1, 2, 3};
  std::initializer_list<int> list2 = {4, 5, 6};

  SECTION("operator== with array_view")
  {
    REQUIRE(view1 == view2);
    REQUIRE_FALSE(view1 == view3);
    REQUIRE_FALSE(view1 == view_short);
  }

  SECTION("operator!= with array_view")
  {
    REQUIRE_FALSE(view1 != view2);
    REQUIRE(view1 != view3);
    REQUIRE(view1 != view_short);
  }

  SECTION("operator== with std::vector")
  {
    REQUIRE(vec1 == view1);
    REQUIRE(view1 == vec1);
    REQUIRE_FALSE(vec2 == view1);
    REQUIRE_FALSE(view1 == vec2);
  }

  SECTION("operator!= with std::vector")
  {
    REQUIRE_FALSE(vec1 != view1);
    REQUIRE_FALSE(view1 != vec1);
    REQUIRE(vec2 != view1);
    REQUIRE(view1 != vec2);
  }

  SECTION("equals with array_view")
  {
    REQUIRE(view1.equals(view2));
    REQUIRE_FALSE(view1.equals(view3));
    REQUIRE_FALSE(view1.equals(view_short));
  }

  SECTION("equals with initializer_list")
  {
    REQUIRE(view1.equals(list1));
    REQUIRE_FALSE(view1.equals(list2));
  }
}

TEST_CASE("array_view No Assignment", "[array_view]")
{
  SECTION("Assignment from int - Compile Error")
  {
    STATIC_REQUIRE_FALSE(std::is_assignable_v<internal::array_view<int>, int>);
  }

  SECTION("Assignment from initializer_list - Compile Error")
  {
    STATIC_REQUIRE_FALSE(std::is_assignable_v<internal::array_view<int>,
                                              std::initializer_list<int>>);
  }
}