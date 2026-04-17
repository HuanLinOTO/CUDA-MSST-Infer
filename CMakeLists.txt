cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(CudaInfer LANGUAGES CXX CUDA)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures — use "-real" to embed SASS only (no PTX), keeping
# the last arch with PTX for forward-compatibility via JIT.
# Override with -DCMAKE_CUDA_ARCHITECTURES=... for local single-GPU builds.
set(CMAKE_CUDA_ARCHITECTURES "80-real;86-real;89")

# Require CUDA 12.0+
find_package(CUDAToolkit 12.0 REQUIRED)

# NVCC flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr --allow-unsupported-compiler")

# Platform-specific compiler and linker flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /DNDEBUG")
else()
    # Strip debug symbols in Release builds (saves 50-100 MB)
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -s")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# cuDNN paths — configurable via CMake cache variables
# Default: nvidia-cudnn-cu12 pip package location (Windows)
if(NOT DEFINED CUDNN_ROOT)
    set(CUDNN_ROOT "C:/Users/Administrator/miniconda3/Lib/site-packages/nvidia/cudnn" CACHE PATH "cuDNN root directory")
endif()
if(NOT DEFINED CUDNN_INCLUDE_DIR)
    set(CUDNN_INCLUDE_DIR "${CUDNN_ROOT}/include" CACHE PATH "cuDNN include directory")
endif()
if(NOT DEFINED CUDNN_LIB_DIR)
    set(CUDNN_LIB_DIR "${CMAKE_SOURCE_DIR}/cudnn_libs" CACHE PATH "cuDNN library directory")
endif()
# Ensure CUDNN_LIB_DIR is absolute (CI may pass relative path)
if(NOT IS_ABSOLUTE "${CUDNN_LIB_DIR}")
    get_filename_component(CUDNN_LIB_DIR "${CMAKE_SOURCE_DIR}/${CUDNN_LIB_DIR}" ABSOLUTE)
endif()
include_directories(${CUDNN_INCLUDE_DIR})

# Collect source files
file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
# Exclude Python bindings from main exe
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*python_bindings\\.cpp$")
set(ALL_SOURCES ${CUDA_SOURCES} ${CPP_SOURCES})

# Main executable
add_executable(cudasep_infer ${ALL_SOURCES})

# Platform-specific cuDNN linking
if(WIN32)
    set(CUDNN_LIBS
        ${CUDNN_LIB_DIR}/cudnn64_9.lib
        ${CUDNN_LIB_DIR}/cudnn_cnn64_9.lib
        ${CUDNN_LIB_DIR}/cudnn_ops64_9.lib
        ${CUDNN_LIB_DIR}/cudnn_adv64_9.lib
        ${CUDNN_LIB_DIR}/cudnn_graph64_9.lib
    )
else()
    # Linux: find .so libs in CUDNN_LIB_DIR
    find_library(CUDNN_LIB cudnn PATHS ${CUDNN_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CUDNN_CNN_LIB cudnn_cnn PATHS ${CUDNN_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CUDNN_OPS_LIB cudnn_ops PATHS ${CUDNN_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CUDNN_ADV_LIB cudnn_adv PATHS ${CUDNN_LIB_DIR} NO_DEFAULT_PATH)
    find_library(CUDNN_GRAPH_LIB cudnn_graph PATHS ${CUDNN_LIB_DIR} NO_DEFAULT_PATH)
    set(CUDNN_LIBS
        ${CUDNN_LIB}
        ${CUDNN_CNN_LIB}
        ${CUDNN_OPS_LIB}
        ${CUDNN_ADV_LIB}
        ${CUDNN_GRAPH_LIB}
    )
endif()

target_link_libraries(cudasep_infer PRIVATE
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cufft
    CUDA::cudart
    ${CUDNN_LIBS}
)

target_include_directories(cudasep_infer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDNN_INCLUDE_DIR}
)

# Optional Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)

if(BUILD_PYTHON_BINDINGS)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11
    )
    FetchContent_MakeAvailable(pybind11)

    # Shared library for Python bindings
    # Re-collect sources, exclude main.cpp, include python_bindings.cpp
    file(GLOB_RECURSE PY_CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")
    file(GLOB_RECURSE PY_CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
    list(FILTER PY_CPP_SOURCES EXCLUDE REGEX ".*main\\.cpp$")
    set(LIB_SOURCES ${PY_CUDA_SOURCES} ${PY_CPP_SOURCES})

    pybind11_add_module(cudasep ${LIB_SOURCES})

    target_link_libraries(cudasep PRIVATE
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cufft
        CUDA::cudart
        ${CUDNN_LIBS}
    )

    target_include_directories(cudasep PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CUDNN_INCLUDE_DIR}
    )
endif()
