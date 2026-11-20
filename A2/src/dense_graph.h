/**
 * The file in which you will implement your DenseGraph GPU solutions!
 */

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
void build_graph( DenseGraph g, edge_t const * edge_list, std::size_t m )
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


__global__
void two_hop_tiled( DenseGraph g, node_t * out)
{
    // IMPLEMENT ME! tiled! DO TILING look at slides and stuff


    // TODO: write seprate functions and use this as a benchmark,

    size_t th_id = threadIdx.x + blockIdx.x * blockDim.x; // maybe can get rid of block part.

    // Make Const expr and use same thing as block configuration size
    const size_t TILE_SIZE = 32; // FACT CHECK THIS


    size_t br = blockIdx.y; // block row
    size_t bc = blockIdx.x; // block col

    size_t tr = threadIdx.y; //  what warp number its in 0-32
    size_t tc = threadIdx.x; // 0-32

    size_t row = br * TILE_SIZE + tr;
    size_t col = bc * TILE_SIZE + tc;

    __shared__ node_t tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ node_t tile_b[TILE_SIZE][TILE_SIZE];


    int result = 0; // thread result gets acumulated across tiles.

    for (size_t i = 0; i < g.n / TILE_SIZE; i++) {

      // Load values to shared memory
        size_t a_row = row;
        size_t a_col = i * TILE_SIZE + tc;

       tile_a[tr][tc] = g.adjacencyMatrix[a_row * g.n + a_col];

        size_t b_row = i * TILE_SIZE + tr;
        size_t b_col = col;
       tile_b[tr][tc] = g.adjacencyMatrix[b_row * g.n + b_col];
        __syncthreads();
      // Compute partial result for tile.
        for (size_t k = 0; k < TILE_SIZE; k++){
            result += tile_a[tr][k] * tile_b[k][tc];
        }
        __syncthreads();
    }


    // then remove the diagonal and clamp values back to [0,1]
    if (row == col) {
        result = 0;
    } else if (result > 1) {
        result = 1;
    }

    // Only write output if in range
    if (th_id >= g.n * g.n) {
        return;
    }

    out[row * g.n + col] = result;

    return;
}


__global__
void two_hop_warp_optimizations( DenseGraph g, node_t * out)
{
    // IMPLEMENT ME! warp primitives + bit-packing! __popc and __any_sync

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



inline void run_two_hop( DenseGraph g )
{

    node_t * d_out;
    cudaMalloc( (void**)&d_out, sizeof( node_t ) * g.matrix_size() );

    // std::size_t const threads = 256;

    // use same const expr here
    std::size_t const threads = 1024;
    std::size_t const blocks  = ( g.matrix_size() + threads - 1 ) / threads;
    // two_hop_reachability<<< blocks, threads >>>( g, d_out );
    two_hop_tiled<<< blocks, threads >>>( g, d_out );
    // two_hop_warp_optimizations<<< blocks, threads >>>( g, d_out );

    cudaMemcpy( g.adjacencyMatrix, d_out, sizeof( node_t ) * g.matrix_size(), cudaMemcpyDeviceToDevice );
    cudaFree( d_out );
}

} // namespace gpu
} // namespace a2
} // namespace csc485b
