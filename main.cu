/**
 * Driver for the benchmark comparison. Generates random data,
 * runs the CPU baseline, and then runs your code.
 */

#include <cstddef>  // std::size_t type
#include <iostream> // std::cout, std::endl
#include <vector>

#include "algorithm_choices.h"
#include "data_generator.h"
#include "data_types.h"
#include "cuda_common.h"

int main()
{
    std::size_t const n = 4;
    std::size_t const switch_at = 3 * ( n >> 2 ) ;

    auto data = csc485b::a1::generate_uniform< element_t >( n );
    csc485b::a1::cpu::run_cpu_baseline( data, switch_at, n );
    csc485b::a1::gpu::run_gpu_soln( data, switch_at, n );

    return EXIT_SUCCESS;
}