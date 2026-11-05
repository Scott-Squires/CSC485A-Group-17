cmake_minimum_required(VERSION 3.24)

# Default to the local GPU's architecture; override with -DCMAKE_CUDA_ARCHITECTURES=75
# to match the notebook's -arch=sm_75.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(A2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_executable(a2
    src/main.cu
    src/cuda_common.h
    src/data_generator.h
    src/data_types.h
    src/dense_graph.h
    src/sparse_graph.h
)
target_include_directories(a2 PRIVATE src)

# Mirror the notebook's "-O3 -g" for optimized builds.
target_compile_options(a2 PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3>
)

# `cmake --build build --target run` builds and runs the program.
add_custom_target(run
    COMMAND a2
    DEPENDS a2
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL
)
