/**
 * A collection of commonly used data types throughout this project.
 */
#pragma once

#include <iostream> // for std::ostream
#include <vector>

namespace csc485b{
namespace a2{

using node_t = int;
using edge_t = int2;

using edge_list_t = std::vector< edge_t >;
using node_list_t = std::vector< node_t >;

} // namespace a2
} // namespace csc485b
