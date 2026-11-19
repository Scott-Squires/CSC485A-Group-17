/**
 * The file in which you will implement your SparseGraph GPU solutions!
 */
#pragma once

#include <algorithm> // std::min
#include <cstddef>  // std::size_t type
#include <cassert>

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


//Calculates the degree of each node with atomic adds into the g.neighbours_start_at list
__global__
void calculate_degrees( SparseGraph g, edge_t const * edge_list, std::size_t m )
{
    size_t th_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (th_id >= m) return;

    //calculate the degree of each node and store in nieghbours start at
    edge_t edge = edge_list[th_id];
    atomicAdd(&(g.neighbours_start_at[edge.x]), 1); //we are only doing edge.x because there will be a symmetric edge in the graph that will handle y.

    return;
}

//prefix sum where each block does its own prefix sum and writes the last element to output_sums. Modifies input to have the prefix sum
__global__
void prefix_sum(node_t* input, node_t* output_sums, int size)
{
    //assert(blockDim.x == 1024);

    int th_id = threadIdx.x + blockIdx.x * blockDim.x;
    int block_id = blockIdx.x;
    int local_warp_id = (th_id % 1024) / 32;

    __shared__ int warp_sums[32];

    //use warp primitives to calculate warp sums
    int val = th_id < size ? input[th_id] : 0;
    for (int stride = 1; stride < 32; stride <<= 1)
    {
        int result = __shfl_up_sync(__activemask(), val, stride);
        if (th_id % 32 >= stride) val += result;
    }

    //last thread in each warp writes to warp_sums
    if (th_id % 32 == 31) warp_sums[local_warp_id] = val;

    __syncthreads();

    //first warp in each block does a prefix sum on the warp_totals
    if (th_id % 1024 < 32)
    {
        int val2 = warp_sums[th_id % 1024];
        for (int stride = 1; stride < 32; stride <<= 1)
        {
            int result = __shfl_up_sync(__activemask(), val2, stride);
            if (th_id % 32 >= stride) val2 += result;
        }

        warp_sums[th_id % 1024] = val2;
    }

    __syncthreads();

    int warp_sum_result = local_warp_id > 0 ? warp_sums[local_warp_id - 1] : 0;
    int result = val + warp_sum_result;
    if (th_id < size) input[th_id] = result - input[th_id]; //each thread adds its sum to the total from previous warps in the thread (minus the current value so it is exclusive)
    if (th_id % 1024 == 1023) output_sums[block_id] = result; //last thread in block writes to output
}

__global__
void finalize_neighbour_start_at(SparseGraph g, node_t* block_prefix_sums)
{
    //add block totals to neighbours_start_at
    size_t th_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (th_id >= g.n) return;

    //after this, neighbour_start_at will be in the desired CSR representation
    g.neighbours_start_at[th_id] += block_prefix_sums[th_id / 1024];
}

__global__
void finalize_neighbours(SparseGraph g, edge_t const* edge_list, std::size_t m, int* counter_array)
{
    size_t th_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (th_id >= m) return;

    edge_t edge = edge_list[th_id];

    int index_to_write = g.neighbours_start_at[edge.x] + atomicAdd(&counter_array[edge.x], 1);

    g.neighbours[index_to_write] = edge.y;
}

/**
 * Constructs a SparseGraph from an input edge list of m edges.
 *
 * @pre The pointers in SparseGraph g have already been allocated.
 */
void build_graph(SparseGraph g, edge_t const* edge_list, std::size_t m)
{
    //because we don't do a recursive prefix sum on the block_totals below, we can't handle larger than 1024*1024. 
    //Can be implemented later if we want to handle that size of input by doing a recursive prefix sum on block_totals
    if (g.n > 1024 * 1024)
    {
        std::cout << "cannot process inputs with more than 1,048,576 nodes" << std::endl;
    }

    calculate_degrees << < (m + 1023) / 1024, 1024 >> >(g, edge_list, m);

    node_t* block_totals;
    node_t* block_sums;
    cudaMalloc((void**)&block_totals, sizeof(node_t) * g.n);
    cudaMalloc((void**)&block_sums, sizeof(node_t) * g.n);
    cudaMemset(block_totals, 0, sizeof(a2::node_t) * g.n);
    cudaMemset(block_sums, 0, sizeof(a2::node_t) * g.n);
    
    //prefix sum on neighbours_start_at and output each thread block total to block_totals
    prefix_sum << < (g.n + 1023) / 1024, 1024 >> > (g.neighbours_start_at, block_totals, g.n);

    //prefix sum on block_totals.
    prefix_sum << < 1, 1024 >> > (block_totals, block_sums, g.n);

    //add block_sums back to the neighbours_start_at list. Then calculate neighbours list.
    finalize_neighbour_start_at << < (g.n + 1023) / 1024, 1024 >> > (g, block_totals);

    cudaMemset(block_sums, 0, sizeof(a2::node_t) * g.n); //reusing block_sums here as a new empty array of size n for the next pass
    finalize_neighbours << <(m + 1023) / 1024, 1024 >> > (g, edge_list, m, block_sums);

    cudaFree(block_totals);
    cudaFree(block_sums);
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
