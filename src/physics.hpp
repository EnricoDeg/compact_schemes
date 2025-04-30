/*
 * @file physics.hpp
 *
 * @copyright Copyright (C) 2025 Enrico Degregori <enrico.degregori@gmail.com>
 *
 * @author Enrico Degregori <enrico.degregori@gmail.com>
 * 
 * MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions: 
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef CANARD_PHYSICS_HPP
#define CANARD_PHYSICS_HPP

#include "kernels/definitions.hpp"
#include "common/utils.hpp"
#include "transforms.hpp"
#include "kernels/reductionShMem.hpp"

template<typenamew Type>
CANARD_DEVICE inline void atomicMax(Type * addr, Type val) {
    if (*addr >= val) return;

    unsigned int *const addr_as_ui = (unsigned int *)addr;
    unsigned int old = *addr_as_ui, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
}

template<unsigned int BlockSize, typename Type>
CANARD_GLOBAL void calc_time_step_kernel(Type * xim, Type * etm,
                                         Type * zem, Type * de, 
                                         Type * yaco, Type * ssk, Type * umf, Type *res, unsigned int size)
{
    CANARD_SHMEM Type sdata[BlockSize];
    CANARD_SHMEM Type umf_shared[3];

    Type * xim_x = xim;
    Type * xim_y = xim + size;
    Type * xim_z = xim + 2 * size;

    Type * etm_x = etm;
    Type * etm_y = etm + size;
    Type * etm_z = etm + 2 * size;

    Type * zem_x = zem;
    Type * zem_y = zem + size;
    Type * zem_z = zem + 2 * size;

    Type * de1 = de;
    Type * de2 = de + size;
    Type * de3 = de + 2 * size;
    Type * de4 = de + 3 * size;
    Type * de5 = de + 4 * size;

    unsigned int thread_id = get_thread_global_idx();
    unsigned int thread_local_id = get_thread_local_idx();

    Type result;
    sdata[thread_local_id] = 0;

    // load umf in shared memory
    if(thread_local_id == 0)
    {
        umf_shared[0] = umf[0];
        umf_shared[1] = umf[1];
        umf_shared[2] = umf[2];
    }
    __syncthreads();

    // compute
    if(thread_id < size)
    {
        Type rr1 = xim_x[thread_id] * xim_x[thread_id] +
                   xim_y[thread_id] * xim_y[thread_id] +
                   xim_z[thread_id] * xim_z[thread_id] +
                   etm_x[thread_id] * etm_x[thread_id] +
                   etm_y[thread_id] * etm_y[thread_id] +
                   etm_z[thread_id] * etm_z[thread_id] +
                   zem_x[thread_id] * zem_x[thread_id] +
                   zem_y[thread_id] * zem_y[thread_id] +
                   zem_z[thread_id] * zem_z[thread_id];

        Type rr2 = abs(xim_x[thread_id] * (de2[thread_id] + umf_shared[0])  +
                       xim_y[thread_id] * (de3[thread_id] + umf_shared[1])  +
                       xim_z[thread_id] * (de4[thread_id] + umf_shared[2])) +
                   abs(etm_x[thread_id] * (de2[thread_id] + umf_shared[0])  +
                       etm_y[thread_id] * (de3[thread_id] + umf_shared[1])  +
                       etm_z[thread_id] * (de4[thread_id] + umf_shared[2])) +
                   abs(zem_x[thread_id] * (de2[thread_id] + umf_shared[0])  +
                       zem_y[thread_id] * (de3[thread_id] + umf_shared[1])  +
                       zem_z[thread_id] * (de4[thread_id] + umf_shared[2]));

        Type ssi = abs(yaco[thread_id]);

        result = (sqrt(de5[thread_id] * rr1) + rr2) * ssi;
    }

    // load into shmem
    sdata[thread_local_id] = result;
    __syncthreads();

    // do reduction in shared mem
    blockReduceShMemUnroll<BlockSize, Type>(sdata, thread_local_id);

    // write result for this block to global mem
    if (thread_local_id == 0)
        atomicMax(&res[0], sdata[0]);

    __syncthreads();

    // compute
    result = de1[thread_id] * ssk[thread_id] * rr1 * ssi * ssi;

    // load into shmem
    sdata[thread_local_id] = result;
    __syncthreads();

    // do reduction in shared mem
    blockReduceShMemUnroll<BlockSize, Type>(sdata, thread_local_id);

    // write result for this block to global mem
    if (thread_local_id == 0)
        atomicMax(&res[1], sdata[0]);
}

template<typename Type>
void calc_time_step(Type * xim,
                    Type * etm,
                    Type * zem,
                    Type * de,
                    Type * yaco,
                    Type * ssk,
                    Type * umf,
                    Type   cfl,
                    Type * dte,
                    unsigned int size)
{
    unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(size, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGridX, blockPerGridY);

    Type * max_value = (Type*)malloc(2 * sizeof(Type));
    Type * max_value_local = (Type*)malloc(2 * sizeof(Type));

    Type *d_max_value_local;
    check_cuda(cudaMalloc(&d_max_value_local, 2 * sizeof(Type)));

    TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
        CANARD_KERNEL_NAME(calc_time_step_kernel<256>),
        xim, etm, zem, de, yaco, ssk, umf, d_max_value_local, size);

    cudaMemcpy(max_value_local, d_max_value_local, 2 * sizeof(Type), cudaMemcpyDeviceToHost);

    MPI_Datatype mpi_type = mpi_get_type<Type>();
    MPI_Allreduce(&max_value_local[0], &max_value[0], 2, mpi_type, MPI_MAX, MPI_COMM_WORLD);

    max_value[0] = cfl / max_value[0];
    max_value[1] = 0.5 / max_value[1];
    *dte = min(max_value[0], max_value[1]);

    check_cuda(cudaFree(d_max_value_local));
    free(max_value);
    free(max_value_local);
}

#endif
