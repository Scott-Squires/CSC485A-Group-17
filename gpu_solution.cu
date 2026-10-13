/**
 * The file in which you will implement your GPU solutions!
 */

#include "algorithm_choices.h"

#include <chrono>    // for timing
#include <iostream>  // std::cout, std::endl
#include <cmath>

#include "cuda_common.h"

namespace csc485b {
namespace a1      {
namespace gpu     {

/**
 * The CPU baseline benefits from warm caches because the data was generated on
 * the CPU. Run the data through the GPU once with some arbitrary logic to
 * ensure that the GPU cache is warm too and the comparison is more fair.
 */
__global__
void warm_the_gpu( element_t * data, std::size_t invert_at_pos, std::size_t num_elements )
{
    int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

    // We know this will never be true, because of the data generator logic,
    // but I doubt that the compiler will figure it out. Thus every element
    // should be read, but none of them should be modified.
    if( th_id < num_elements && data[ th_id ] > num_elements * 100 )
    {
        ++data[ th_id ]; // should not be possible.
    }
}

/**
 * Your solution. Should match the CPU output.
 */
__global__
void opposing_sort( element_t * data, std::size_t invert_at_pos, std::size_t num_elements )
{
    int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

    if( th_id >= num_elements )
        return;
    
    //MAIN FOR LOOP: RUN UP THE STRIDE
    for (std::size_t stride = 1; stride < num_elements; stride <<= 1)
    {
        //INNER FOR LOOP: WALK DOWN FROM THE STRIDE TO ZERO TO SWAP ACCORDINGLY
        for (std::size_t meta_stride = stride; meta_stride > 0; meta_stride >>= 1) //count down the stride back to 1 every time we bump it up
        {
            if (th_id % (meta_stride * 2) < meta_stride) //this element will try to swap with index i + stride.
            {
                if (th_id / (stride * 2) % 2 == 0 && !(stride >= num_elements / 2 && th_id >= invert_at_pos)) //determine whether we are swapping for largest or smallest, plus a clause so it sorts backwards in the last quarter.
                {
                    //smallest first
                    if (data[th_id] > data[th_id + meta_stride])
                    {
                        element_t temp = data[th_id];
                        data[th_id] = data[th_id + meta_stride];
                        data[th_id + meta_stride] = temp;
                    }
                }
                else
                {
                    //largest first
                    if (data[th_id] < data[th_id + meta_stride])
                    {
                        element_t temp = data[th_id];
                        data[th_id] = data[th_id + meta_stride];
                        data[th_id + meta_stride] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__
void radix_sort(element_t* data, std::size_t num_elements, std::size_t max_digits)
{
    int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (th_id >= num_elements)
        return;

    extern __shared__ element_t buffer[];
    __shared__ element_t histogram[10];

    for (std::size_t i{ 0 }; i < max_digits; i++)
    {
        buffer[th_id] = 0;
        if (th_id < 10) histogram[th_id] = 0;
        //figure out how many of each digit and record in histogram. Also record which occurrance in buffer
        buffer[th_id] = atomicAdd(&histogram[data[th_id] % (int)std::pow(10, i+1) / (int)(std::pow(10, i))], (element_t)1);
        
        //turn running count into prefix sum
        if (th_id == 0)
        {
            element_t total = 0;
            for (int j{ 0 }; j < 10; j++)
            {
                element_t temp = histogram[j];
                histogram[j] = total;
                total += temp;
            }
        }

        //calculate new index and save data
        int new_index = histogram[data[th_id] % (int)std::pow(10, i+1) / (int)(std::pow(10, i))] + (buffer[th_id]);
        element_t temp = data[th_id];
        __syncthreads(); //sync so we don't overwrite
        data[new_index] = temp; //order this decimal place
    }
}


/**
 * Performs all the logic of allocating device vectors and copying host/input
 * vectors to the device. Times the opposing_sort() kernel with wall time,
 * but excludes set up and tear down costs such as mallocs, frees, and memcpies.
 */
std::vector< element_t > run_gpu_soln( std::vector< element_t > data, std::size_t switch_at, std::size_t n )
{
    // Kernel launch configurations. Feel free to change these.
    // This is set to maximise the size of a thread block on a T4, but it hasn't
    // been tuned. It's not known if this is optimal.
    std::size_t const threads_per_block = 1024;
    std::size_t const num_blocks =  ( n + threads_per_block - 1 ) / threads_per_block;

    // Allocate arrays on the device/GPU
    element_t * d_data;
    cudaMalloc( (void**) & d_data, sizeof( element_t ) * n );
    CHECK_ERROR("Allocating input array on device");

    // Copy the input from the host to the device/GPU
    cudaMemcpy( d_data, data.data(), sizeof( element_t ) * n, cudaMemcpyHostToDevice );
    CHECK_ERROR("Copying input array to device");

    // Warm the cache on the GPU for a more fair comparison
    warm_the_gpu<<< num_blocks, threads_per_block>>>( d_data, switch_at, n );

    // Time the execution of the kernel that you implemented
    auto const kernel_start = std::chrono::high_resolution_clock::now();
    opposing_sort<<< num_blocks, threads_per_block>>>( d_data, switch_at, n );
    auto const kernel_end = std::chrono::high_resolution_clock::now();
    CHECK_ERROR("Executing kernel on device");

    // After the timer ends, copy the result back, free the device vector,
    // and echo out the timings and the results.
    cudaMemcpy( data.data(), d_data, sizeof( element_t ) * n, cudaMemcpyDeviceToHost );
    CHECK_ERROR("Transferring result back to host");
    cudaFree( d_data );
    CHECK_ERROR("Freeing device memory");

    std::cout << "GPU Solution time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
              << " ns" << std::endl;

    for( auto const x : data ) std::cout << x << " "; std::cout << std::endl;

    return data;
}

std::vector< element_t > run_gpu_radix(std::vector< element_t > data, std::size_t n, std::size_t max_digits)
{
    // Kernel launch configurations. Feel free to change these.
    // This is set to maximise the size of a thread block on a T4, but it hasn't
    // been tuned. It's not known if this is optimal.
    std::size_t const threads_per_block = 1024;
    std::size_t const num_blocks = (n + threads_per_block - 1) / threads_per_block;

    // Allocate arrays on the device/GPU
    element_t* d_data;
    cudaMalloc((void**)&d_data, sizeof(element_t) * n);
    CHECK_ERROR("Allocating input array on device");

    // Copy the input from the host to the device/GPU
    cudaMemcpy(d_data, data.data(), sizeof(element_t) * n, cudaMemcpyHostToDevice);
    CHECK_ERROR("Copying input array to device");

    //// Warm the cache on the GPU for a more fair comparison
    //warm_the_gpu << < num_blocks, threads_per_block >> > (d_dataIn, d_dataOut, n);

    // Time the execution of the kernel that you implemented
    auto const kernel_start = std::chrono::high_resolution_clock::now();
    radix_sort << < num_blocks, threads_per_block, n * sizeof(element_t) >> > (d_data, n, max_digits);
    auto const kernel_end = std::chrono::high_resolution_clock::now();
    CHECK_ERROR("Executing kernel on device");

    // After the timer ends, copy the result back, free the device vector,
    // and echo out the timings and the results.
    cudaMemcpy(data.data(), d_data, sizeof(element_t) * n, cudaMemcpyDeviceToHost);
    CHECK_ERROR("Transferring result back to host");
    cudaFree(d_data);
    CHECK_ERROR("Freeing device memory");

    std::cout << "GPU Radix Solution time: "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
        << " ns" << std::endl;

    for (auto const x : data) std::cout << x << " "; std::cout << std::endl;

    return data;
}

} // namespace gpu
} // namespace a1
} // namespace csc485b