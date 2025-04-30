/*
 * @file numerics.hpp
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

#ifndef CANARD_KERNELS_NUMERICS_HPP
#define CANARD_KERNELS_NUMERICS_HPP

#include "common/parameters.hpp"
#include "common/data_types.hpp"

#include "cuda/kernels/definitions.hpp"
#include "cuda/kernels/templateShmem.hpp"
#include "cuda/kernels/functional.hpp"
#include "cuda/kernels/transforms.hpp"
#include "transforms.hpp"

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
    if(Axis == 0)
    {
        thread_stride = dcomp_info.lxi;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = 1;
        grid_size = dcomp_info.let * dcomp_info.lze;
    }
    else if(Axis == 1)
    {
        thread_stride = 1;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = dcomp_info.lxi;
        grid_size = dcomp_info.lxi * dcomp_info.lze;
    }
    else if(Axis == 2)
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
CANARD_GLOBAL void deriv_kernel(Type *infield, Type *outfield, Type * recv, Type * pbci,
                                Type *drva,
                                Type h_1, int nstart, int nend, t_dcomp dcomp_info,
                                unsigned int variable_id)
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

    // load input field in shared memory
    sx[thread_local_idx] = infield[gmem_idx];
    __syncthreads();

    // load shared memory: tridiagonal matrix and rhs
    if(thread_local_idx == 0)
    {
        if(nstart == 0)
        {
            srhs[0] = (ab00 * sx[0] + ab01 * sx[1] + ab02 * sx[2]) * h_1;
            sa[0] = 0.0;
            sb[0] = 1.0;
            sc[0] = alpha01;
        }
        else
        {
            srhs[0] = 0;
            details::static_for<0, lmd, 1>{}([&](auto i)
            {
                srhs[0] += pbci[i] * sx[i];
            });
            srhs[0] += recv[face_idx];
            srhs[0] *= h_1;
            sa[0] = 0.0;
            sb[0] = 1.0;
            sc[0] = alpha;
        }
    }
    else if(thread_local_idx == 1)
    {
        if(nstart == 0)
        {
            srhs[1] = ab10 * (sx[2] - sx[0]) * h_1;
            sa[1] = alpha10;
            sb[1] = 1.0;
            sc[1] = alpha10;
        }
        else
        {
            srhs[1] = 0;
            details::static_for<0, lmd, 1>{}([&](auto i)
            {
                srhs[1] += pbci[lmd + i] * sx[i];
            });
            srhs[1] += recv[face_size + face_idx];
            srhs[1] *= h_1;
            sa[1] = alpha;
            sb[1] = 1.0;
            sc[1] = alpha;
        }
    }
    else if(thread_local_idx == BlockSize -1)
    {
        if(nend == 0)
        {
            srhs[BlockSize - 1] = -(ab00 * sx[BlockSize - 1] +
                                    ab01 * sx[BlockSize - 2] +
                                    ab02 * sx[BlockSize - 3]) * h_1;
            sa[BlockSize - 1] = alpha01;
            sb[BlockSize - 1] = 1.0;
            sc[BlockSize - 1] = 0.0;
        }
        else
        {
            srhs[BlockSize - 1] = 0;
            details::static_for<0, lmd, 1>{}([&](auto i)
            {
                srhs[BlockSize - 1] += pbci[i] * sx[BlockSize - 1 - i];
            });
            srhs[BlockSize - 1] += recv[2 * face_size + face_idx];
            srhs[BlockSize - 1] *= -h_1;
            sa[BlockSize - 1] = alpha;
            sb[BlockSize - 1] = 1.0;
            sc[BlockSize - 1] = 0.0;
        }
    }
    else if(thread_local_idx == BlockSize - 2)
    {
        if(nend == 0)
        {
            srhs[BlockSize - 2] = -ab10*(sx[BlockSize - 3] - sx[BlockSize - 1]) * h_1;
            sa[BlockSize - 2] = alpha10;
            sb[BlockSize - 2] = 1.0;
            sc[BlockSize - 2] = alpha10;
        }
        else
        {
            srhs[BlockSize - 2] = 0;
            details::static_for<0, lmd, 1>{}([&](auto i)
            {
                srhs[BlockSize - 2] += pbci[lmd + i] * sx[BlockSize - 1 - i];
            });
            srhs[BlockSize - 2] += recv[3 * face_size + face_idx];
            srhs[BlockSize - 2] *= -h_1;
            sa[BlockSize - 2] = alpha;
            sb[BlockSize - 2] = 1.0;
            sc[BlockSize - 2] = alpha;
        }
    }
    else
    {
        srhs[thread_local_idx] = aa * (sx[thread_local_idx+1] -
                                       sx[thread_local_idx-1]) * h_1 +
                                 bb * (sx[thread_local_idx+2]-
                                       sx[thread_local_idx-2]) * h_1;
        sa[thread_local_idx] = alpha;
        sb[thread_local_idx] = 1.0;
        sc[thread_local_idx] = alpha;
    }

    __syncthreads();

    // Parallel cyclic reduction
    int iter   = static_cast<int>(std::log2f(BlockSize / 2));
    int stride = 1;

    for(int j = 0; j < iter; j++)
    {
        int right = thread_local_idx + stride;
        if(right >= BlockSize)
            right = BlockSize - 1;

        int left = thread_local_idx - stride;
        if(left < 0)
            left = 0;

        Type k1 = sa[thread_local_idx] / sb[left];
        Type k2 = sc[thread_local_idx] / sb[right];

        Type tb   = sb[thread_local_idx] - sc[left] * k1 - sa[right] * k2;
        Type trhs = srhs[thread_local_idx] - srhs[left] * k1 - srhs[right] * k2;
        Type ta   = -sa[left] * k1;
        Type tc   = -sc[right] * k2;

        __syncthreads();

        sb[thread_local_idx]   = tb;
        srhs[thread_local_idx] = trhs;
        sa[thread_local_idx]   = ta;
        sc[thread_local_idx]   = tc;

        stride <<= 1; //stride *= 2;

        __syncthreads();
    }

    if(thread_local_idx < BlockSize / 2)
    {
        // Solve 2x2 systems
        int i    = thread_local_idx;
        int j    = thread_local_idx + stride;
        Type det = sb[j] * sb[i] - sc[i] * sa[j];
        det      = static_cast<Type>(1) / det;

        sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
        sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
    }

    __syncthreads();

    // store results
    outfield[gmem_idx] = sx[thread_local_idx];

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
