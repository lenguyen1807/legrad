set(TEST_NAME ${LIB_NAME}_test)

# Include Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(${TEST_NAME}
    tensor/tensor_view.cpp
    internal/array_view.cpp
    internal/view_pack.cpp
)

if (LEGRAD_BUILD_SHARED)
    target_link_libraries(${TEST_NAME}
        PRIVATE
        ${SHARED_LIB_NAME}
        Catch2::Catch2WithMain
    )
else()
    target_link_libraries(${TEST_NAME}
        PRIVATE
        ${STATIC_LIB_NAME}
        Catch2::Catch2WithMain
    )
endif()

# Enable testing
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(${TEST_NAME})