/**
 * The file in which you will implement your SparseGraph GPU solutions!
 */

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
namespace a2      {

/**
 * A SparseGraph is optimised for a graph in which the number of edges
 * is close to cn, for a small constanct c. It is represented in CSR format.
 */
struct SparseGraph
{
  std::size_t n; /**< Number of nodes in the graph. */
  std::size_t m; /**< Number of edges in the graph. */
  node_t * neighbours_start_at; /** Pointer to an n=|V| offset array */
  node_t * neighbours; /** Pointer to an m=|E| array of edge destinations */
};


namespace gpu {

/**
 * Constructs a SparseGraph from an input edge list of m edges.
 *
 * @pre The pointers in SparseGraph g have already been allocated.
 */
__global__
void build_graph( SparseGraph g, edge_t const * edge_list, std::size_t m )
{
    // IMPLEMENT ME!
    // TODO: Important to get a 60%
    return;
}

/**
  * Repopulates the adjacency lists as a new graph that represents
  * the two-hop neighbourhood of input graph g
  */
__global__
void two_hop_reachability( SparseGraph g )
{
    // IMPLEMENT ME!
    // algorithm unknown
    return;
}

inline void run_two_hop( SparseGraph g )
{
    two_hop_reachability<<< 1, 1 >>>( g );  // placeholder until you write sparse
}


} // namespace gpu
} // namespace a2
} // namespace csc485b
