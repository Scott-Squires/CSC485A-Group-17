/**
 * The file in which you will implement your DenseGraph GPU solutions!
 */
#pragma once

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
namespace a2      {

/**
 * A DenseGraph is optimised for a graph in which the number of edges
 * is close to n(n-1). It is represented using an adjacency matrix.
 */
struct DenseGraph
{
  std::size_t n; /**< Number of nodes in the graph. */
  node_t * adjacencyMatrix; /** Pointer to an n x n adj. matrix */

  /** Returns number of cells in the adjacency matrix. */
  __device__ __host__ __forceinline__
  std::size_t matrix_size() const { return n * n; }
};


namespace gpu {


/**
 * Constructs a DenseGraph from an input edge list of m edges.
 *
 * @pre The pointers in DenseGraph g have already been allocated.
 */
__global__
void build_graph_gpu( DenseGraph g, edge_t const * edge_list, std::size_t m )
{
    // IMPLEMENT ME! DONE!
    size_t th_id = threadIdx.x + blockIdx.x * blockDim.x; // maybe can get rid of block part.

    if ( th_id >= m) {
        return; // thread is larger than number of edge list entries and is not needed.
    }

    int2 edge = edge_list[th_id];
    size_t n = g.n; // number of rows/columns
    g.adjacencyMatrix[edge.x * n + edge.y] = 1;
    g.adjacencyMatrix[edge.y * n + edge.x] = 1;
    return;
}

void build_graph(DenseGraph g, edge_t const* edge_list, std::size_t m)
{
    csc485b::a2::gpu::build_graph_gpu << < (m + 255) / 256, 256 >> > (g, edge_list, m);
}

/**
  * Repopulates the adjacency matrix as a new graph that represents
  * the two-hop neighbourhood of input graph g
  */
__global__
void two_hop_reachability( DenseGraph g, node_t * out)
{
    // IMPLEMENT ME! Nieve DONE!
    // square adjacencyMatrix

    // TODO: write seprate functions and use this as a benchmark,

    size_t th_id = threadIdx.x + blockIdx.x * blockDim.x; // maybe can get rid of block part.

    size_t row = th_id / g.n;
    size_t col = th_id % g.n;

    if (th_id >= g.n * g.n) {
        return;
    }

    int result = 0;

    for (size_t i = 0; i < g.n; i++) {
        result += g.adjacencyMatrix[row * g.n + i] * g.adjacencyMatrix[i * g.n + col];
    }


    // then remove the diagonal and clamp values back to [0,1]
    if (row == col) {
        result = 0;
    } else if (result > 1) {
        result = 1;
    }

    out[row * g.n + col] = result;

    return;
}

/**
  * Repopulates the adjacency matrix as a new graph that represents
  * the two-hop neighbourhood of input graph g. Each output cell is
  * mapped to one warp.
  */
__global__
void two_hop_reachability_bminus(DenseGraph g, node_t* out)
{
    // IMPLEMENT ME! Nieve DONE!
    // square adjacencyMatrix

    // TODO: write seprate functions and use this as a benchmark,

    size_t th_id = threadIdx.x + blockIdx.x * blockDim.x; // maybe can get rid of block part.

    size_t cell_id = th_id / 32;
    size_t warp_th_id = th_id % 32;

    size_t row = cell_id / g.n;
    size_t col = cell_id % g.n;

    if (th_id >= g.n * g.n * 32) {
        return;
    }

    int result = 0;

    for (size_t i = 0; i < g.n; i += 32) {
        if (i + warp_th_id < g.n)
            result += g.adjacencyMatrix[row * g.n + i + warp_th_id] * g.adjacencyMatrix[(i + warp_th_id) * g.n + col];
    }


    // then remove the diagonal and clamp values back to [0,1]
    if (row == col) {
        result = 0;
    }
    else if (result > 1) {
        result = 1;
    }

    if (result > 0)
        out[row * g.n + col] = result;

    return;
}

inline void run_two_hop( DenseGraph g )
{
    node_t * d_out;
    cudaMalloc( (void**)&d_out, sizeof( node_t ) * g.matrix_size() );

    std::size_t const threads = 256;
    std::size_t const blocks  = ( g.matrix_size() + threads - 1 ) / threads * 32;
    two_hop_reachability_bminus <<< blocks, threads >>>( g, d_out );

    cudaMemcpy( g.adjacencyMatrix, d_out, sizeof( node_t ) * g.matrix_size(), cudaMemcpyDeviceToDevice );
    cudaFree( d_out );
}

} // namespace gpu
} // namespace a2
} // namespace csc485b
