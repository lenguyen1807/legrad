set(LIB_NAME legrad)
set(STATIC_LIB_NAME ${LIB_NAME}_static)
set(SHARED_LIB_NAME ${LIB_NAME}_shared)

# --- Source and Include File Gathering ---
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
    "${CMAKE_CURRENT_SOURCE_DIR}/util" 
    "${CMAKE_CURRENT_SOURCE_DIR}/tensor"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal"
)

# --- Options ---
# General
option(LEGRAD_LTO "legrad: Enable link time optimization" OFF)
option(LEGRAD_GPROF "legrad: Enable profilling" OFF)

# Build static and shared
option(LEGRAD_BUILD_SHARED "legrad: Build shared library" OFF) # Default to build only static library if not specified

# Debug level
option(LEGRAD_DEBUG_1 "legrad: Set debug at level 1" OFF)
option(LEGRAD_DEBUG_2 "legrad: Set debug at level 2" OFF)
option(LEGRAD_DEBUG_3 "legrad: Set debug at level 3" OFF)
option(LEGRAD_WARNINGS "legrad: Enable Warnings" ON) # Enable compiler warnings by default.

# Feature options for external libraries.
option(LEGRAD_OPENCV "legrad: Use OpenCV" ON) # Enable OpenCV support by default.
option(LEGRAD_FMT "legrad: Use fmt format library" ON) # Enable fmt library support by default.

# Sanitizer options for debugging memory and thread related issues.
option(LEGRAD_SANITIZE_ADDRESS "legrad: Enable Sanitize Address" OFF)
option(LEGRAD_SANITIZE_THREAD "legrad: Enable Sanitize Thread" OFF)
option(LEGRAD_SANITIZE_UNDEFINED "legrad: Enable Sanitize Undefined" OFF)

# Backend specific options, likely for different hardware or computation backends.
# Note that CPU with SIMD will be enabled by default
option(LEGRAD_BLAS "legrad: Enable BLAS" OFF) # Basic Linear Algebra Subprograms (BLAS) support.
option(LEGRAD_CUDA "legrad: Enable CUDA" OFF) # CUDA backend support (NVIDIA GPUs).
option(LEGRAD_METAL "legrad: Enable Metal" OFF) # Metal backend support (Apple GPUs).

# Build options for tests, examples, and benchmarks.
option(LEGRAD_BUILD_TESTS "legrad: Build tests" OFF)
option(LEGRAD_BUILD_EXAMPLES "legrad: Build examples" OFF)
option(LEGRAD_BUILD_BENCHMARK "legrad: Build benchmark" OFF) # Option specifically for benchmark builds.

# --- External Library Handling ---
if (LEGRAD_OPENCV)
    find_package(OpenCV REQUIRED)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV not found")
    endif()
    message(STATUS "legrad: Use OpenCV")
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
    message(STATUS "legrad: Use fmt")
    set(LEGRAD_COMPILE_DEFINITIONS ${LEGRAD_COMPILE_DEFINITIONS} "LEGRAD_USE_FMT")
    set(LEGRAD_EXTRA_LIBS ${LEGRAD_EXTRA_LIBS} fmt::fmt)
endif()

# Use threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Use Boost
find_package(Boost CONFIG COMPONENTS system log log_setup REQUIRED)

# Add backend for CPU
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu)

# --- Sanitizer Options (Non-MSVC specific) ---
if (NOT MSVC)
    if (LEGRAD_SANITIZE_THREAD)
        message(STATUS "legrad: Sanitize Threads")
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (LEGRAD_SANITIZE_ADDRESS)
        message(STATUS "legrad: Sanitize Address")
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (LEGRAD_SANITIZE_UNDEFINED)
        message(STATUS "legrad: Sanitize Undefined")
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

# --- BLAS Backend Handling ---
if (LEGRAD_BLAS)
    set(LEGRAD_COMPILE_DEFINITIONS ${LEGRAD_COMPILE_DEFINITIONS} "LEGRAD_USE_BLAS")

    if (NOT LEGRAD_BUILD_SHARED)
        set(BLA_STATIC ON)
    endif()

    if ($(CMAKE_VERSION) VERSION_GREATER_EQUAL 3.22)
        set(BLA_SIZEOF_INTEGER 8)
    endif()

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

        message(STATUS "legrad: Apple (Accelerate) BLAS Backend")

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
    else()
        find_package(BLAS)
        if (BLAS_FOUND)
            message(STATUS "legrad: BLAS found, Libraries ${BLAS_LIBRARIES}")
        endif()

        if ("${BLAS_INCLUDE_DIRS}" STREQUAL "")
            # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
            # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
            find_package(PkgConfig REQUIRED)
            if (${LLAMA_BLAS_VENDOR} MATCHES "Generic")
                pkg_check_modules(DepBLAS REQUIRED blas)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "OpenBLAS")
                pkg_check_modules(DepBLAS REQUIRED openblas)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "FLAME")
                pkg_check_modules(DepBLAS REQUIRED blis)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "ATLAS")
                pkg_check_modules(DepBLAS REQUIRED blas-atlas)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "FlexiBLAS")
                pkg_check_modules(DepBLAS REQUIRED flexiblas_api)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "Intel")
                # all Intel* libraries share the same include path
                pkg_check_modules(DepBLAS REQUIRED mkl-sdl)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "NVHPC")
                # this doesn't provide pkg-config
                # suggest to assign BLAS_INCLUDE_DIRS on your own
                if ("${NVHPC_VERSION}" STREQUAL "")
                    message(WARNING "Better to set NVHPC_VERSION")
                else()
                    set(DepBLAS_FOUND ON)
                    set(DepBLAS_INCLUDE_DIRS "/opt/nvidia/hpc_sdk/${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}/${NVHPC_VERSION}/math_libs/include")
                endif()
            endif()
            
            if (DepBLAS_FOUND)
                set(BLAS_INCLUDE_DIRS ${DepBLAS_INCLUDE_DIRS})
            else()
                message(WARNING "BLAS_INCLUDE_DIRS neither been provided nor been automatically"
                " detected by pkgconfig, trying to find cblas.h from possible paths...")
                find_path(BLAS_INCLUDE_DIRS
                    NAMES cblas.h
                    HINTS
                        /usr/include
                        /usr/local/include
                        /usr/include/openblas
                        /opt/homebrew/opt/openblas/include
                        /usr/local/opt/openblas/include
                        /usr/include/x86_64-linux-gnu/openblas/include
                )
            endif()
        endif()

        message(STATUS "legrad: BLAS found, Includes ${BLAS_INCLUDE_DIRS}")
        add_compile_options(${BLAS_LINKER_FLAGS})
        set(LEGRAD_EXTRA_LIBS ${LEGRAD_EXTRA_LIBS} ${BLAS_LIBRARIES})
        set(LEGRAD_EXTRA_INCLUDES ${LLAMA_EXTRA_INCLUDES} ${BLAS_INCLUDE_DIRS})
    endif()
endif()

# --- CUDA Backend (TODO: Implement later) ---
if (LEGRAD_CUDA)
    # TODO: Implement later
endif()

# --- Vulkan Backend (TODO: Implement later) ---
if (LEGRAD_VULKAN)
    # TODO: Implement later
endif()

# --- Metal Backend (Apple Specific) ---
if (APPLE AND LEGRAD_METAL)
    message(STATUS "legrad: Apple Metal Backend")
    set(LEGRAD_COMPILE_DEFINITIONS ${LEGRAD_COMPILE_DEFINITIONS} "LEGRAD_USE_METAL")

    # Copy shader to binary directory
    file(GLOB SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/backend/metal/shader/*")
    foreach(SHADER_FILE ${SHADER_FILES})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
        configure_file(${SHADER_FILE} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${SHADER_NAME}" COPYONLY)
    endforeach()

    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/backend/metal")
    set(LEGRAD_EXTRA_LIBS ${LEGRAD_EXTRA_LIBS} METAL_CPP)
endif()

# --- CPU Architecture Specific Options ---
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(LEGRAD_COMPILE_DEFINITIONS ${LEGRAD_COMPILE_DEFINITIONS} "LEGRAD_USE_ARM")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64")
        # Apple M1, M2, etc.
        # Raspberry Pi 3, 4, Zero 2 (64-bit)
        message(STATUS "legrad: ARM64 detected")
        add_compile_options(-mcpu=native)
    endif()
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
        # Raspberry Pi 1, Zero
        message(STATUS "legrad: ARMv6 detected")
        add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access)
    endif()
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
        # Raspberry Pi 2
        message(STATUS "legrad: ARMv7 detected")
        add_compile_options(-mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations)
    endif()
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
        # Raspberry Pi 3, 4, Zero 2 (32-bit)
        message(STATUS "legrad: ARMv8 detected")
        add_compile_options(-mfp16-format=ieee -mno-unaligned-access)
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "legrad: x86 detected")
    # TODO: Implement later
endif()

# --- Debug Level Definitions ---
if (LEGRAD_DEBUG_1)
    message(STATUS "legrad: Enable Debug Level 1 - Information")
    set(LEGRAD_COMPILE_DEFINITIONS ${LEGRAD_COMPILE_DEFINITIONS} "LEGRAD_DEBUG_LEVEL_1")
endif()

if (LEGRAD_DEBUG_2)
    message(STATUS "legrad: Enable Debug Level 2 - Debug")
    set(LEGRAD_COMPILE_DEFINITIONS ${LEGRAD_COMPILE_DEFINITIONS} "LEGRAD_DEBUG_LEVEL_2")
endif()

if (LEGRAD_DEBUG_3)
    message(STATUS "legrad: Enable Debug Level 3 - Trace")
    set(LEGRAD_COMPILE_DEFINITIONS ${LEGRAD_COMPILE_DEFINITIONS} "LEGRAD_DEBUG_LEVEL_3")
endif()

# --- Compiler Warnings ---
if (LEGRAD_WARNINGS)
    set(cxx_flags # Common and useful compiler warning flags.
        -Wall # Enable all common warnings.
        -Wextra # Enable extra warnings.
        -Wpedantic # Enable pedantic warnings (stricter standard conformance).
        -Wcast-qual # Warn about casting away qualifiers (const, volatile).
        -Wno-unused-function # Suppress warnings about unused functions (can be noisy in header-only libraries or during development).
    ) 
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>")
endif()

# --- Link Time Optimization (LTO) ---
if (LEGRAD_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

# --- Static Linking (Non-MSVC specific) ---
if (NOT MSVC)
    if (NOT LEGRAD_BUILD_SHARED)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (LEGRAD_GPROF) # Enable profiler support.
        add_compile_options(-pg)
    endif()
endif()

# --- Release/Debug Build Type Specific Flags ---
if(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "legrad: Build in Release mode")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CXX_FLAGS_RELEASE}>")
else()
    message(STATUS "legrad: Build in Debug mode")
endif()

# --- Library Creation (Shared or Static) ---
if (LEGRAD_BUILD_SHARED)
    message(STATUS "legrad: Build shared library")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_library(${SHARED_LIB_NAME} SHARED ${SRC_FILES})
    target_include_directories(${SHARED_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
        $<INSTALL_INTERFACE:${INCLUDE_DIR}>
        ${LEGRAD_EXTRA_INCLUDES}
    )
    target_link_libraries(${SHARED_LIB_NAME} PUBLIC 
                        Boost::system
                        Boost::log
                        Boost::log_setup
                        Threads::Threads
                        ${LEGRAD_EXTRA_LIBS}
                        CPU_BACKEND)
    target_compile_definitions(${SHARED_LIB_NAME} PUBLIC ${LEGRAD_COMPILE_DEFINITIONS})
else ()
    add_library(${STATIC_LIB_NAME} STATIC ${SRC_FILES})
    target_include_directories(${STATIC_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
        $<INSTALL_INTERFACE:${INCLUDE_DIR}>
        ${LEGRAD_EXTRA_INCLUDES}
    )
    target_link_libraries(${STATIC_LIB_NAME} PUBLIC 
                        Boost::system
                        Boost::log
                        Boost::log_setup
                        Threads::Threads
                        ${LEGRAD_EXTRA_LIBS}
                        CPU_BACKEND)
    target_compile_definitions(${STATIC_LIB_NAME} PUBLIC ${LEGRAD_COMPILE_DEFINITIONS})
endif()

if (LEGRAD_BUILD_TESTS)
    message(STATUS "legrad: Build tests")
    add_subdirectory(tests)
endif()
