/**
 * Driver for the benchmark comparison. Generates random data,
 * runs the CPU baseline, and then runs your code.
 */

#include <algorithm> // std::mismatch
#include <cstddef>  // std::size_t type
#include <iostream> // std::cout, std::endl
#include <vector>

#include "algorithm_choices.h"
#include "data_generator.h"
#include "data_types.h"
#include "cuda_common.h"

int run_opposing_sort()
{
    std::size_t const n = 1024;
    std::size_t const switch_at = 3 * (n >> 2);

    auto const data = csc485b::a1::generate_uniform< element_t >(n);
    for (auto const x : data) std::cout << x << " "; std::cout << std::endl;

    // Each run receives its own copy of the input and returns its sorted result.
    auto const cpu_result = csc485b::a1::cpu::run_cpu_baseline( data, switch_at, n );
    auto const gpu_result = csc485b::a1::gpu::run_gpu_soln( data, switch_at, n );

    if (cpu_result.size() != gpu_result.size())
    {
        std::cout << "FAIL: CPU returned " << cpu_result.size()
                  << " elements but GPU returned " << gpu_result.size() << std::endl;
        return 1;
    }

    auto const mismatch = std::mismatch( cpu_result.begin(), cpu_result.end(), gpu_result.begin() );
    if (mismatch.first != cpu_result.end())
    {
        std::size_t const i = mismatch.first - cpu_result.begin();
        std::cout << "FAIL at index " << i
                  << ": CPU = " << *mismatch.first
                  << ", GPU = " << *mismatch.second << std::endl;
        return 1;
    }

    std::cout << "+++++++++ Matches CPU benchmark +++++++++" << std::endl;

    return EXIT_SUCCESS;
}

//courtesy of https://stackoverflow.com/questions/1489830/efficient-way-to-determine-number-of-digits-in-an-integer
unsigned GetNumberOfDigits(unsigned i)
{
    return i > 0 ? (int)log10((double)i) + 1 : 1;
}

int run_radix_sort()
{
    std::size_t const n = 64;

    auto const data = csc485b::a1::generate_uniform< element_t >(n);
    //std::vector<element_t> const data{ 123,654,1023,8,99,40,77,5 };
    for (auto const x : data) std::cout << x << " "; std::cout << std::endl;

    std::size_t max_digits = GetNumberOfDigits(n * 2);
    //std::size_t max_digits = 4;

    auto const gpu_result = csc485b::a1::gpu::run_gpu_radix(data, n, max_digits);

    return EXIT_SUCCESS;
}

int main()
{

    //return run_opposing_sort();
    return run_radix_sort();
    
}
