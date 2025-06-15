/*
 * @file numerics_pc.hpp
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

#ifndef CANARD_KERNELS_NUMERICS_PC_HPP
#define CANARD_KERNELS_NUMERICS_PC_HPP

#include "common/parameters.hpp"
#include "common/data_types.hpp"

#include "cuda/kernels/definitions.hpp"
#include "cuda/kernels/templateShmem.hpp"
#include "cuda/kernels/functional.hpp"
#include "cuda/kernels/transforms.hpp"
#include "cuda/kernels/numerics_device_pc.hpp"

template<
unsigned int Axis,
typename Type
>
CANARD_GLOBAL void fill_gpu_buffer_kernel(Type *infield,
                                          Type *buffer,
                                          Type *pbco,
                                          t_dcomp dcomp_info,
                                          int istart,
                                          int increment,
                                          int buffer_offset)
{
    int thread_stride;
    int block_stride;
    int stride;
    int grid_size;
    if constexpr (Axis == 0)
    {
        thread_stride = dcomp_info.lxi;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = 1;
        grid_size = dcomp_info.let * dcomp_info.lze;
    }
    else if constexpr (Axis == 1)
    {
        thread_stride = 1;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = dcomp_info.lxi;
        grid_size = dcomp_info.lxi * dcomp_info.lze;
    }
    else if constexpr (Axis == 2)
    {
        thread_stride = 1;
        block_stride = dcomp_info.lxi;
        stride = dcomp_info.let * dcomp_info.lxi;
        grid_size = dcomp_info.lxi * dcomp_info.let;
    }

    int thread_idx = blockIdx.x * block_stride + threadIdx.x * thread_stride;
    int thread_idx_store = blockIdx.x * blockDim.x + threadIdx.x;

    Type vgpr_data[lmd];

    // load pbco in shared memory
    CANARD_SHMEM Type pbco_shmem[2 * lmd];
    if(threadIdx.x == 0)
    {
        details::static_for<0, 2 * lmd, 1>{}([&](int i)
        {
            pbco_shmem[i] = pbco[i];
        });
    }

    // load data in vgpr
    details::static_for<0, lmd, 1>{}([&](int i)
    {
        int idx = thread_idx + (istart + increment * i) * stride;
        vgpr_data[i] = infield[idx];
    });
    __syncthreads();

    // compute first buffer value
    Type value= 0;
    details::static_for<0, lmd, 1>{}([&](int i)
    {
        value += pbco_shmem[i] * vgpr_data[i];
    });

    // store first buffer value
    buffer[buffer_offset + thread_idx_store] = value;

    // compute second buffer value
    value= 0;
    details::static_for<0, lmd, 1>{}([&](int i)
    {
        value += pbco_shmem[lmd + i] * vgpr_data[i];
    });

    // store second buffer value
    buffer[buffer_offset + grid_size + thread_idx_store] = value;
}

template<unsigned int Axis, typename Type>
CANARD_GLOBAL void deriv_kernel_1d(Type *infield, Type *outfield, Type * recv, Type * pbci,
                                   Type *drva,
                                   Type h_1, int nstart, int nend, t_dcomp dcomp_info,
                                   unsigned int variable_id)
{
    // use dynamic shared memory
    // needed for template
    SharedMemory<Type> smem;
    Type * sdata = smem.getPointer();

    unsigned int BlockSize;
    if constexpr(Axis == 0)
    {
        BlockSize = dcomp_info.lxi;
    }
    else if constexpr(Axis == 1)
    {
        BlockSize = dcomp_info.let;
    }
    else if constexpr(Axis == 2)
    {
        BlockSize = dcomp_info.lze;
    }

    Type * sx, *srhs, *sa, *sb, *sc;

    sx   = sdata;
    srhs = sdata + BlockSize;
    sa   = sdata + 2 * BlockSize;
    sb   = sdata + 3 * BlockSize;
    sc   = sdata + 4 * BlockSize;

    deriv_kernel_1d_impl<Axis>(infield,
                               outfield,
                               recv,
                               pbci,
                               drva,
                               h_1,
                               nstart,
                               nend,
                               dcomp_info,
                               variable_id,
                               sx,
                               srhs,
                               sa,
                               sb,
                               sc,
                               BlockSize);
}

template<unsigned int Axis, typename Type>
CANARD_GLOBAL void deriv_kernel_2d(Type *infield, Type * outfield, Type * recv, Type * pbci,
                                   Type *drva,
                                   Type h_1, int nstart, int nend, t_dcomp dcomp_info,
                                   unsigned int variable_id, unsigned int component_id)
{
    // use dynamic shared memory
    // needed for template
    SharedMemory<Type> smem;
    Type * sdata = smem.getPointer();
    Type * sx, *srhs, *sa, *sb, *sc;

    unsigned int BlockSize;
    if constexpr(Axis == 0)
    {
        BlockSize = dcomp_info.lxi;
    }
    else if constexpr(Axis == 1)
    {
        BlockSize = dcomp_info.let;
    }
    else if constexpr(Axis == 2)
    {
        BlockSize = dcomp_info.lze;
    }

    sx   = sdata;
    srhs = sdata + BlockSize;
    sa   = sdata + 2 * BlockSize;
    sb   = sdata + 3 * BlockSize;
    sc   = sdata + 4 * BlockSize;

    unsigned int thread_local_idx = get_thread_local_idx();
    unsigned int gmem_block_stridex = get_block_stridex<Axis>(dcomp_info);
    unsigned int gmem_block_stridey = get_block_stridey<Axis>(dcomp_info);
    unsigned int gmem_thread_stride = get_thread_stride<Axis>(dcomp_info);
    unsigned int gmem_idx = get_index(gmem_block_stridex, gmem_block_stridey, gmem_thread_stride);
    unsigned int face_stride = get_face_stride<Axis>(dcomp_info);
    unsigned int face_idx = blockIdx.y * face_stride + blockIdx.x;
    unsigned int face_size = get_face_size<Axis>(dcomp_info);

    Type * recv_variable = recv + variable_id * 2 * 2 * face_size;

    // load input field in shared memory
    sx[thread_local_idx] = infield[gmem_idx + component_id * dcomp_info.lmx];
    __syncthreads();

    // load shared memory: tridiagonal matrix and rhs
    build_system(BlockSize, thread_local_idx,
                 face_idx, face_size,
                 nstart, nend,
                 h_1,
                 sa, sb, sc, srhs, sx,
                 recv_variable, pbci);

    __syncthreads();

    // Parallel cyclic reduction
    PCR_solver(BlockSize, thread_local_idx, sa, sb, sc, srhs, sx);

    __syncthreads();

    // store results
    outfield[gmem_idx + Axis * dcomp_info.lmx] = sx[thread_local_idx];

    if(thread_local_idx == 0)
    {
        unsigned int idx = variable_id * face_size + face_idx;
        drva[idx] = sx[thread_local_idx];
    }
    if(thread_local_idx == BlockSize - 1)
    {
        unsigned int idx = NumberOfVariables * face_size +
            variable_id * face_size + face_idx;
        drva[idx] = sx[thread_local_idx];
    }
}

#endif
