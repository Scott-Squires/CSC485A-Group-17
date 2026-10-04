#pragma once

#include <vector>

#include "data_types.h"

namespace csc485b {
namespace a1 {
namespace cpu {

void run_cpu_baseline( std::vector< element_t > data, std::size_t switch_at, std::size_t n );

} // namespace cpu


namespace gpu {

void run_gpu_soln( std::vector< element_t > data, std::size_t switch_at, std::size_t n );

} // namespace gpu
} // namespace a1
} // namespace csc485b