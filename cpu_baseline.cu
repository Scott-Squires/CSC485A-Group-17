/**
 * CPU methods that the GPU should outperform.
 */

#include "algorithm_choices.h"

#include <algorithm> // std::sort()
#include <chrono>    // for timing
#include <iostream>  // std::cout, std::endl

namespace csc485b {
namespace a1      {
namespace cpu     {

/**
 * Simple solution that just sorts the whole array with a built-in sort
 * function and then resorts the last portion in the opposing order with
 * a second call to that same built-in sort function.
 */
void opposing_sort( element_t * data, std::size_t invert_at_pos, std::size_t num_elements )
{
    std::sort( data, data + num_elements, std::less< element_t >{} );
    std::sort( data + invert_at_pos, data + num_elements, std::greater< element_t >{} );
}

/**
 * Run the single-threaded CPU baseline that students are supposed to outperform
 * in order to obtain higher grades on this assignment. Times the execution and
 * prints to the standard output (e.g., the screen) that "wall time." Note that
 * the functions takes the input by value so as to not perturb the original data
 * in place.
 */
void run_cpu_baseline( std::vector< element_t > data, std::size_t switch_at, std::size_t n )
{
    auto const cpu_start = std::chrono::high_resolution_clock::now();
    opposing_sort( data.data(), switch_at, n );
    auto const cpu_end = std::chrono::high_resolution_clock::now();

    std::cout << "CPU Baseline time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()
              << " ns" << std::endl;

    for( auto const x : data ) std::cout << x << " "; std::cout << std::endl;
}

} // namespace cpu
} // namespace a1
} // namespace csc485b