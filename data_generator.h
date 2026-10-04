/**
 * Functions for generating random input data with a fixed seed
 */
#pragma once

#include <random>  // for std::mt19937, std::uniform_int_distribution
#include <vector>

#include "data_types.h"

namespace csc485b {
namespace a1 {

/**
 * Generates and returns a vector of random uniform data of a given length, n,
 * for any integral type. Input range will be [0, 2n].
 */
template < typename T >
std::vector< T > generate_uniform( std::size_t n )
{
    // for details of random number generation, see:
    // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
    std::size_t random_seed = 20240916;  // use magic seed
    std::mt19937 rng( random_seed );     // use mersenne twister generator
    std::uniform_int_distribution<> distrib(0, 2 * n);

    std::vector< T > random_data( n ); // init array
    std::generate( std::begin( random_data )
                 , std::end  ( random_data )
                 , [ &rng, &distrib ](){ return static_cast< T >( distrib( rng ) ); });

    return random_data;
}

} // namespace a1
} // namespace csc485b