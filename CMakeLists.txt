cmake_minimum_required(VERSION 3.16)
project(legrad_main VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LEGRAD_BUILD_TESTS ON)
set(LEGRAD_SANITIZE_ADDRESS ON)
set(LEGRAD_BLAS ON)
set(LEGRAD_DEBUG_1 ON)
set(LEGRAD_DEBUG_2 ON)
set(LEGRAD_DEBUG_3 ON)
set(LEGRAD_BUILD_SHARED ON)
add_subdirectory(legrad)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} legrad_shared)
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})