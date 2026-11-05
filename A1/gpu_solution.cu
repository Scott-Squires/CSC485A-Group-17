
#include "algorithm_choices.h"

#include <chrono>    // for timing
#include <iostream>  // std::cout, std::endl

#include "cuda_common.h"

namespace csc485b {
    namespace a1 {
        namespace gpu {

            /**
             * The CPU baseline benefits from warm caches because the data was generated on
             * the CPU. Run the data through the GPU once with some arbitrary logic to
             * ensure that the GPU cache is warm too and the comparison is more fair.
             */
            __global__
                void warm_the_gpu(element_t* data, std::size_t invert_at_pos, std::size_t num_elements)
            {
                int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

                // We know this will never be true, because of the data generator logic,
                // but I doubt that the compiler will figure it out. Thus every element
                // should be read, but none of them should be modified.
                if (th_id < num_elements && data[th_id] > num_elements * 100)
                {
                    ++data[th_id]; // should not be possible.
                }
            }

            /**
             * Your solution. Should match the CPU output.
             */
            __global__
                void opposing_sort(element_t* data, std::size_t invert_at_pos, std::size_t num_elements, std::size_t start_stride)
            {
                int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

                int total_threads_per_block = blockDim.x * blockDim.y * blockDim.z;

                if (th_id >= num_elements)
                    return;

                //MAIN FOR LOOP: RUN UP THE STRIDE
                for (std::size_t stride = start_stride; stride < num_elements; stride <<= 1)
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
                            if (meta_stride >= total_threads_per_block) break;
                        }
                        if (stride >= total_threads_per_block) break;
                        __syncthreads();
                    }
                }
            }

            /**
             * Performs all the logic of allocating device vectors and copying host/input
             * vectors to the device. Times the opposing_sort() kernel with wall time,
             * but excludes set up and tear down costs such as mallocs, frees, and memcpies.
             */
            void run_gpu_soln(std::vector< element_t > data, std::size_t switch_at, std::size_t n)
            {
                // Kernel launch configurations. Feel free to change these.
                // This is set to maximise the size of a thread block on a T4, but it hasn't
                // been tuned. It's not known if this is optimal.
                std::size_t const threads_per_block = 16;
                std::size_t const num_blocks = (n + threads_per_block - 1) / threads_per_block;

                // Allocate arrays on the device/GPU
                element_t* d_data;
                cudaMalloc((void**)&d_data, sizeof(element_t) * n);
                CHECK_ERROR("Allocating input array on device");

                // Copy the input from the host to the device/GPU
                cudaMemcpy(d_data, data.data(), sizeof(element_t) * n, cudaMemcpyHostToDevice);
                CHECK_ERROR("Copying input array to device");

                // Warm the cache on the GPU for a more fair comparison
                warm_the_gpu << < num_blocks, threads_per_block >> > (d_data, switch_at, n);

                // Time the execution of the kernel that you implemented
                auto const kernel_start = std::chrono::high_resolution_clock::now();

                for (std::size_t stride = 1; stride < n; stride <<= 1)
                {
                    for (std::size_t meta_stride = stride; meta_stride > 0; meta_stride >>= 1)
                    {
                        std::cout << "running opposing sort with stride: " << meta_stride << std::endl;
                        opposing_sort << < num_blocks, threads_per_block >> > (d_data, switch_at, n, meta_stride);

                        cudaMemcpy(data.data(), d_data, sizeof(element_t) * n, cudaMemcpyDeviceToHost);
                        CHECK_ERROR("Transferring result back to host");
                        for (auto const x : data) std::cout << x << " "; std::cout << std::endl;

                        if (meta_stride < threads_per_block) //oppposing sort will handle walking down from 512 stride down to 0 without having to re-launch the kernel
                        {
                            meta_stride = 0;
                        }
                    }
                    if (stride < threads_per_block) //opposing sort will handle walking up to a stride of 512 without re-launching
                    {
                        stride = threads_per_block/2;
                    }
                }
                
                
                auto const kernel_end = std::chrono::high_resolution_clock::now();
                CHECK_ERROR("Executing kernel on device");

                // After the timer ends, copy the result back, free the device vector,
                // and echo out the timings and the results.
                cudaMemcpy(data.data(), d_data, sizeof(element_t) * n, cudaMemcpyDeviceToHost);
                CHECK_ERROR("Transferring result back to host");
                cudaFree(d_data);
                CHECK_ERROR("Freeing device memory");

                std::cout << "GPU Solution time: "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
                    << " ns" << std::endl;

                for (auto const x : data) std::cout << x << " "; std::cout << std::endl;
            }

            __global__
                void radix_sort(element_t* data, std::size_t num_elements, std::size_t num_bits)
            {
                int const th_id = threadIdx.x;
                int lane = th_id % 32;
                int warp = th_id / 32;
                int W = blockDim.x / 32; // should give number of warps since we only have a 1 dimentional block


                // maybe this messes up syncwarp/syncblock?
                if (th_id >= num_elements) {
                    return;
                }
                constexpr size_t max_buffer_size = 1024;
                constexpr size_t max_table_size = 2 /*rows*/ * 32 /*warps max*/;

                __shared__ element_t buffer_A[max_buffer_size];
                __shared__ element_t buffer_B[max_buffer_size];
                __shared__ element_t table[max_table_size]; // works as a 2D array.

                element_t* src = buffer_A;
                element_t* dest = buffer_B;

                buffer_A[th_id] = data[th_id];

                for (std::size_t i = 0; i < num_bits; i++)
                {
                    element_t key = src[th_id];
                    int d = (key >> i) & 0x1; // either a 0 or 1 

                    uint32_t zeros = __ballot_sync(__activemask(), d == 0);
                    uint32_t ones = ~zeros;
                    uint32_t same_thread_mask = d == 0 ? zeros : ones; // sets thread mask to be all the 0s or 1s depending on the current threads bit

                    int thread_rank = __popc(same_thread_mask & ((1u << lane) - 1));

                    if (lane == 0) {
                        table[0 * W + warp] = __popc(zeros);
                        table[1 * W + warp] = __popc(ones);
                    }

                    __syncthreads();

                    //turn running count into prefix sum
                    if (th_id == 0)
                    {
                        element_t total = 0;
                        for (int j = 0; j < 2 * W; j++)
                        {
                            element_t temp = table[j];
                            table[j] = total;
                            total += temp;
                        }
                    }
                    __syncthreads();

                    //calculate new index and save data
                    int new_index = table[d * W + warp] + thread_rank;
                    // int new_index = histogram[data[th_id] % (int)std::pow(10, i+1) / (int)(std::pow(10, i))] + (buffer[th_id]);
                    // element_t buffer_A = data[th_id];
                    // __syncthreads(); //sync so we don't overwrite
                    dest[new_index] = key;
                    __syncthreads();
                    // swap buffers
                    element_t* temp = src;
                    src = dest;
                    dest = temp;
                }
                //final copy back
                if (th_id < num_elements) {
                    data[th_id] = src[th_id];
                }
            }

            /**
            * Same as run_gpu_soln(), but times the radix_sort() kernel instead.
            * Runs in a single block, so n must be <= 1024.
            */
            void run_gpu_radix_soln(std::vector< element_t > data, std::size_t n)
            {
                // data is in [0, 2n], so count the number of bits in 2n
                std::size_t max_bits = 0;
                for (std::size_t x = 2 * n; x > 0; x >>= 1) ++max_bits;

                element_t* d_data;
                cudaMalloc((void**)&d_data, sizeof(element_t) * n);
                CHECK_ERROR("Allocating input array on device");

                cudaMemcpy(d_data, data.data(), sizeof(element_t) * n, cudaMemcpyHostToDevice);
                CHECK_ERROR("Copying input array to device");

                warm_the_gpu << < 1, n >> > (d_data, n, n);

                auto const kernel_start = std::chrono::high_resolution_clock::now();
                radix_sort << < 1, n >> > (d_data, n, std::min(sizeof(element_t) * 8, max_bits));
                // cudaDeviceSynchronize();
                auto const kernel_end = std::chrono::high_resolution_clock::now();
                CHECK_ERROR("Executing radix_sort kernel on device");

                cudaMemcpy(data.data(), d_data, sizeof(element_t) * n, cudaMemcpyDeviceToHost);
                CHECK_ERROR("Transferring result back to host");
                cudaFree(d_data);
                CHECK_ERROR("Freeing device memory");

                std::cout << "GPU Radix time: "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
                    << " ns" << std::endl;

                for (auto const x : data) std::cout << x << " "; std::cout << std::endl;
            }



        } // namespace gpu
    } // namespace a1
} // namespace csc485b