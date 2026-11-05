/**
 * Driver for the benchmark comparison. Generates random data,
 * runs the CPU baseline, and then runs your code.
 */

#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <iterator> // std::ostream_iterator
#include <vector>

#include "dense_graph.h"
#include "sparse_graph.h"

#include "data_generator.h"
#include "data_types.h"

/**
 * Runs timing tests on a CUDA graph implementation.
 * Consists of independently constructing the graph and then
 * modifying it to its two-hop neighbourhood.
 */
template < typename DeviceGraph >
void run( DeviceGraph g, csc485b::a2::edge_t const * d_edges, std::size_t m )
{
    cudaDeviceSynchronize();
    auto const build_start = std::chrono::high_resolution_clock::now();

    // this code doesn't work yet!
    csc485b::a2::gpu::build_graph<<< 1, 1 >>>( g, d_edges, m );

    cudaDeviceSynchronize();
    auto const reachability_start = std::chrono::high_resolution_clock::now();

    // neither does this!
    csc485b::a2::gpu::two_hop_reachability<<< 1, 1 >>>( g );

    cudaDeviceSynchronize();
    auto const end = std::chrono::high_resolution_clock::now();

    std::cout << "Build time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(reachability_start - build_start).count()
              << " us"
              << std::endl;

    std::cout << "Reachability time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - reachability_start).count()
              << " us"
              << std::endl;
}

/**
 * Allocates space for a dense graph and then runs the test code on it.
 */
void run_dense( csc485b::a2::edge_t const * d_edges, std::size_t n, std::size_t m )
{
    using namespace csc485b;

    // allocate device DenseGraph
    a2::node_t * d_matrix;
    cudaMalloc( (void**)&d_matrix, sizeof( a2::node_t ) * n * n );
    a2::DenseGraph d_dg{ n, d_matrix };

    run( d_dg, d_edges, m );

    // check output?
    std::vector< a2::node_t > host_matrix( d_dg.matrix_size() );
    a2::DenseGraph dg{ n, host_matrix.data() };
    cudaMemcpy( dg.adjacencyMatrix, d_dg.adjacencyMatrix, sizeof( a2::node_t ) * d_dg.matrix_size(), cudaMemcpyDeviceToHost );
    std::copy( host_matrix.cbegin(), host_matrix.cend(), std::ostream_iterator< a2::node_t >( std::cout, " " ) );

    // clean up
    cudaFree( d_matrix );
}

/**
 * Allocates space for a sparse graph and then runs the test code on it.
 */
void run_sparse( csc485b::a2::edge_t const * d_edges, std::size_t n, std::size_t m )
{
    using namespace csc485b;

    // allocate device SparseGraph
    a2::node_t * d_offsets, * d_neighbours;
    cudaMalloc( (void**)&d_offsets,    sizeof( a2::node_t ) * n );
    cudaMalloc( (void**)&d_neighbours, sizeof( a2::node_t ) * m );
    a2::SparseGraph d_sg{ n, m, d_offsets, d_neighbours };

    run( d_sg, d_edges, m );

    // clean up
    cudaFree( d_neighbours );
    cudaFree( d_offsets );
}

int main()
{
    using namespace csc485b;

    // Create input
    std::size_t constexpr n = 8;
    std::size_t constexpr expected_degree = n >> 1;

    a2::edge_list_t const graph = a2::generate_graph( n, n * expected_degree );
    std::size_t const m = graph.size();

    // lazily echo out input graph
    for( auto const& e : graph )
    {
        std::cout << "(" << e.x << "," << e.y << ") ";
    }

    // allocate and memcpy input to device
    a2::edge_t * d_edges;
    cudaMalloc( (void**)&d_edges, sizeof( a2::edge_t ) * m );
    cudaMemcpyAsync( d_edges, graph.data(), sizeof( a2::edge_t ) * m, cudaMemcpyHostToDevice );

    // run your code!
    run_dense ( d_edges, n, m );
    run_sparse( d_edges, n, m );

    return EXIT_SUCCESS;
}