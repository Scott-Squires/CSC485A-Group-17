cmake_minimum_required(VERSION 3.20)

project(csc485b_a1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_executable(csc485b_a1
    main.cu
    cpu_baseline.cu
    gpu_solution.cu
    algorithm_choices.h
    cuda_common.h
    data_generator.h
    data_types.h
)

set_target_properties(csc485b_a1 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(csc485b_a1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
)
