/**
 * Functions for generating random input data with a fixed seed
 */
#pragma once

#include <cassert>  // for assert()
#include <cstddef>  // std::size_t type
#include <random>   // for std::mt19937, std::uniform_int_distribution
#include <vector>

#include "data_types.h"

namespace csc485b {
namespace a2 {

/**
 * Generates and returns a vector of random edges
 * for a graph `G=(V,E)` with `n=|V|=n` and expected `m=|E|`.
 * Referred to as an Erdős-Rényi graph.
 *
 * @see https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.fast_gnp_random_graph.html#networkx.generators.random_graphs.fast_gnp_random_graph
 */
edge_list_t generate_graph( std::size_t n, std::size_t m )
{
    assert( "At most n(n-1) edges in a simple graph" && m < n * ( n - 1 ) );

    int const probability = ( 100 * m ) / ( n * ( n - 1 ) );

    // for details of random number generation, see:
    // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
    std::size_t random_seed = 20241008;  // use magic seed
    std::mt19937 rng( random_seed );     // use mersenne twister generator
    std::uniform_int_distribution<> distrib(0, 100);

    edge_list_t random_edges;
    random_edges.reserve( 2 * m );

    for( node_t u = 0; u < n; ++u )
    {
        for( node_t v = u + 1; v < n; ++v )
        {
            auto const dice_roll = distrib( rng );
            if( dice_roll <= probability )
            {
                random_edges.push_back( make_int2( u, v ) );
                random_edges.push_back( make_int2( v, u ) );
            }
        }
    }

    random_edges.resize( random_edges.size() );


    return random_edges;
}

} // namespace a2
} // namespace csc485b
