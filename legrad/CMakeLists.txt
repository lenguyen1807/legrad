set(LIB_NAME legrad)
set(STATIC_LIB_NAME ${LIB_NAME}_static)
set(SHARED_LIB_NAME ${LIB_NAME}_shared)

file(GLOB_RECURSE SRC_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/macros/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tensor/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/*.cpp"
)
file(GLOB_RECURSE INCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/macros/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/tensor/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/*.h"
)
set(INCLUDE_DIR 
    "${CMAKE_CURRENT_SOURCE_DIR}/core" 
    "${CMAKE_CURRENT_SOURCE_DIR}/macros" 
    "${CMAKE_CURREN_SOURCE_DIR}/util"
    "${CMAKE_CURREN_SOURCE_DIR}/tensor"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal"
)

# Build static and shared
option(LEGRAD_BUILD_SHARED "legrad: Build shared library" OFF)

# Debug level
option(LEGRAD_DEBUG_1 "legrad: Set debug at level 1" OFF)
option(LEGRAD_DEBUG_2 "legrad: Set debug at level 2" OFF)
option(LEGRAD_DEBUG_3 "legrad: Set debug at level 3" OFF)
option(LEGRAD_WARNINGS "legrad: Enable Warnings" ON)

# Should turn on this option, for now I don't know how to handle image without OpenCV
option(LEGRAD_OPENCV "legrad: Use OpenCV" ON)
# Should turn on this option, for now I will use this library for format string
option(LEGRAD_FMT "legrad: Use fmt format library" ON)

# Sanitizer
option(LEGRAD_SANITIZE_ADDRESS "legrad: Enable Sanitize Address" OFF)
option(LEGRAD_SANITIZE_THREAD "legrad: Enable Sanitize Thread" OFF)
option(LEGRAD_SANITIZE_UNDEFINED "legrad: Enable Sanitize Undefined" OFF)

# Backend specific
option(LEGRAD_BLAS "legrad: Enable BLAS" OFF) # BLAS option should ON with other CPU option
option(LEGRAD_CUDA "legrad: Enable CUDA" OFF)
option(LEGRAD_METAL "legrad: Enable Metal" OFF) # Enable Metal by default

# Tests and examples
option(LEGRAD_BUILD_TESTS "legrad: Build tests" OFF)
option(LEGRAD_BUILD_EXAMPLES "legrad: Build examples" OFF)
option(LEGRAD_BUILD_BENCHMARK "legrad: Build benchmark" OFF) # Just for testing benchmark

if (LEGRAD_OPENCV)
    find_package(OpenCV REQUIRED)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV not found")
    endif()
    set(LEGRAD_EXTRA_LIBS ${LEGRAD_EXTRA_LIBS} ${OpenCV_LIBS})
endif()

if (LEGRAD_FMT)
    include(FetchContent)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281
    )
    FetchContent_MakeAvailable(fmt)
    set(LEGRAD_EXTRA_LIBS ${LEGRAD_EXTRA_LIBS} fmt::fmt)
    add_compile_definitions(LEGRAD_USE_FMT)
endif()

# Use threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Use Boost
find_package(Boost CONFIG COMPONENTS system log log_setup REQUIRED)

# Add backend for CPU
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu)

if (NOT MSVC)
    if (LEGRAD_SANITIZE_THREAD)
        message(STATUS "Sanitize Threads")
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (LEGRAD_SANITIZE_ADDRESS)
        message(STATUS "Sanitize Address")
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (LEGRAD_SANITIZE_UNDEFINED)
        message(STATUS "Sanitize Undefined")
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (LEGRAD_BLAS)
    add_compile_definitions(LEGRAD_USE_BLAS)
    if (APPLE)
        # Use BLAS Implementation from Accelerate Framework
        find_library(ACCELERATE_LIBRARY Accelerate)
        if (NOT ACCELERATE_LIBRARY)
            message(FATAL_ERROR "Accelerate not found")
        endif()

        find_library(FOUNDATION_LIBRARY Foundation)
        if (NOT FOUNDATION_LIBRARY)
            message(FATAL_ERROR "Foundation not found")
        endif()

        find_library(QUARTZCORE_LIBRARY QuartzCore)
        if (NOT QUARTZCORE_LIBRARY)
            message(FATAL_ERROR "QuartzCore not found")
        endif()

        message(STATUS "Apple (Accelerate) BLAS Backend")

        if (LEGRAD_METAL)
            set(LEGRAD_EXTRA_LIBS
            ${LEGRAD_EXTRA_LIBS}
            ${ACCELERATE_LIBRARY})
        else ()
            set(LEGRAD_EXTRA_LIBS
            ${LEGRAD_EXTRA_LIBS}
            ${ACCELERATE_LIBRARY} 
            ${FOUNDATION_LIBRARY} 
            ${QUARTZCORE_LIBRARY})
        endif()
    endif()
endif()

if (LEGRAD_CUDA)
    # TODO: Implement later
endif()

if (APPLE AND LEGRAD_METAL)
    message(STATUS "Apple Metal Backend")
    add_compile_definitions(LEGRAD_USE_METAL)

    # Copy shader to binary directory
    file(GLOB SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/backend/metal/shader/*")
    foreach(SHADER_FILE ${SHADER_FILES})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
        configure_file(${SHADER_FILE} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SHADER_NAME}" COPYONLY)
    endforeach()

    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/backend/metal")
    set(LEGRAD_EXTRA_LIBS ${LEGRAD_EXTRA_LIBS} METAL_CPP)
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    add_compile_definitions(LEGRAD_USE_ARM)
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64")
        # Apple M1, M2, etc.
        # Raspberry Pi 3, 4, Zero 2 (64-bit)
        message(STATUS "ARM64 detected")
        add_compile_options(-mcpu=native)
    endif()
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
        # Raspberry Pi 1, Zero
        message(STATUS "ARMv6 detected")
        add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access)
    endif()
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
        # Raspberry Pi 2
        message(STATUS "ARMv7 detected")
        add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations)
    endif()
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
        # Raspberry Pi 3, 4, Zero 2 (32-bit)
        message(STATUS "ARMv78 detected")
        add_compile_options(-mfp16-format=ieee -mno-unaligned-access)
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    add_compile_definitions(LEGRAD_USE_X86)
    # TODO: Implement later
endif()

if (LEGRAD_WARNINGS)
    set(cxx_flags
        -Wall
        -Wextra
        -Wpedantic
        -Wcast-qual
        -Wno-unused-function
        -Wno-multichar
    )

    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>")
endif()

if (LEGRAD_DEBUG_1)
    message(STATUS "Enable Debug Level 1: Information")
    add_compile_definitions(LEGRAD_DEBUG_LEVEL_1)
endif()

if (LEGRAD_DEBUG_2)
    message(STATUS "Enable Debug Level 2: Debug")
    add_compile_definitions(LEGRAD_DEBUG_LEVEL_2)
endif()

if (LEGRAD_DEBUG_3)
    message(STATUS "Enable Debug Level 3: Trace")
    add_compile_definitions(LEGRAD_DEBUG_LEVEL_3)
endif()

if (LEGRAD_BUILD_SHARED)
    add_compile_definitions(LEGRAD_SHARED_LIB)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_library(${SHARED_LIB_NAME} SHARED ${SRC_FILES})
    target_include_directories(${SHARED_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
        $<INSTALL_INTERFACE:${INCLUDE_DIR}>
    )
    target_link_libraries(${SHARED_LIB_NAME} PUBLIC 
                        Boost::system
                        Boost::log
                        Boost::log_setup
                        Threads::Threads
                        ${LEGRAD_EXTRA_LIBS}
                        CPU_BACKEND)
else ()
    set(${STATIC_LIB_NAME})
    add_library(${STATIC_LIB_NAME} STATIC ${SRC_FILES})
    target_include_directories(${STATIC_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
        $<INSTALL_INTERFACE:${INCLUDE_DIR}>
    )
    target_link_libraries(${STATIC_LIB_NAME} PUBLIC 
                        Boost::system
                        Boost::log
                        Boost::log_setup
                        Threads::Threads
                        ${LEGRAD_EXTRA_LIBS}
                        CPU_BACKEND)
endif()

if (LEGRAD_BUILD_TESTS)
    add_subdirectory(tests)
endif()