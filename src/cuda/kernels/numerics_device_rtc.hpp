/*
 * @file numerics_device_rtc.hpp
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

#ifndef CANARD_KERNELS_NUMERICS_DEVICE_RTC_HPP
#define CANARD_KERNELS_NUMERICS_DEVICE_RTC_HPP

#include "common/parameters.hpp"
#include "common/data_types.hpp"

#include "cuda/kernels/definitions.hpp"
#include "cuda/kernels/functional.hpp"
#include "cuda/kernels/transforms.hpp"

template<unsigned int BlockSize, int NumberOfIterations, typename Type>
CANARD_DEVICE inline void PCR_solver(unsigned int thread_local_idx,
                                     Type * sa,
                                     Type * sb,
                                     Type * sc,
                                     Type * srhs,
                                     Type * sx)
{
    int right, left;
    Type k1, k2, tb, trhs, ta, tc;
    int stride = 1;

    details::static_for<0, NumberOfIterations, 1>{}([&](auto j)
    {
        right = thread_local_idx + stride;
        if(right >= BlockSize)
            right = BlockSize - 1;

        left = thread_local_idx - stride;
        if(left < 0)
            left = 0;

        k1 = sa[thread_local_idx] / sb[left];
        k2 = sc[thread_local_idx] / sb[right];

        tb   = sb[thread_local_idx] - sc[left] * k1 - sa[right] * k2;
        trhs = srhs[thread_local_idx] - srhs[left] * k1 - srhs[right] * k2;
        ta   = -sa[left] * k1;
        tc   = -sc[right] * k2;

        __syncthreads();

        sb[thread_local_idx]   = tb;
        srhs[thread_local_idx] = trhs;
        sa[thread_local_idx]   = ta;
        sc[thread_local_idx]   = tc;

        stride <<= 1; //stride *= 2;

        __syncthreads();
    });

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
}

template<unsigned int nstart, unsigned int nend, unsigned int BlockSize, typename Type>
CANARD_DEVICE inline void build_system(unsigned int thread_local_idx,
                                       unsigned int face_idx,
                                       unsigned int face_size,
                                       Type * sa,
                                       Type * sb,
                                       Type * sc,
                                       Type * srhs,
                                       Type * sx,
                                       Type * recv,
                                       Type * pbci)
{
    if(thread_local_idx == 0)
    {
        if constexpr(nstart == 0)
        {
            srhs[0] = (ab00 * sx[0] + ab01 * sx[1] + ab02 * sx[2]);
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
            sa[0] = 0.0;
            sb[0] = 1.0;
            sc[0] = alpha;
        }
    }
    else if(thread_local_idx == 1)
    {
        if constexpr(nstart == 0)
        {
            srhs[1] = ab10 * (sx[2] - sx[0]);
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
            sa[1] = alpha;
            sb[1] = 1.0;
            sc[1] = alpha;
        }
    }
    else if(thread_local_idx == BlockSize -1)
    {
        if constexpr(nend == 0)
        {
            srhs[BlockSize - 1] = -(ab00 * sx[BlockSize - 1] +
                                    ab01 * sx[BlockSize - 2] +
                                    ab02 * sx[BlockSize - 3]);
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
            srhs[BlockSize - 1] *= -1;
            sa[BlockSize - 1] = alpha;
            sb[BlockSize - 1] = 1.0;
            sc[BlockSize - 1] = 0.0;
        }
    }
    else if(thread_local_idx == BlockSize - 2)
    {
        if constexpr(nend == 0)
        {
            srhs[BlockSize - 2] = -ab10*(sx[BlockSize - 3] - sx[BlockSize - 1]);
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
            srhs[BlockSize - 2] *= -1;
            sa[BlockSize - 2] = alpha;
            sb[BlockSize - 2] = 1.0;
            sc[BlockSize - 2] = alpha;
        }
    }
    else
    {
        srhs[thread_local_idx] = aa * (sx[thread_local_idx+1] -
                                       sx[thread_local_idx-1]) +
                                 bb * (sx[thread_local_idx+2]-
                                       sx[thread_local_idx-2]);
        sa[thread_local_idx] = alpha;
        sb[thread_local_idx] = 1.0;
        sc[thread_local_idx] = alpha;
    }
}

template<unsigned int Axis, unsigned int BlockSize, typename Type>
CANARD_DEVICE void deriv_kernel_1d_impl(Type *infield,
                                        Type *outfield,
                                        Type *recv,
                                        Type *pbci,
                                        Type *drva,
                                        t_dcomp dcomp_info,
                                        unsigned int variable_id,
                                        Type *sx, Type *srhs, Type *sa, Type *sb, Type *sc)
{
    constexpr int NumberOfIterations = ITERS;
    constexpr unsigned int nstart = NSTART;
    constexpr unsigned int nend = NEND;

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
    build_system<nstart, nend, BlockSize>(thread_local_idx,
                 face_idx, face_size,
                 sa, sb, sc, srhs, sx,
                 recv, pbci);

    __syncthreads();

    // Parallel cyclic reduction
    PCR_solver<BlockSize, NumberOfIterations>(thread_local_idx, sa, sb, sc, srhs, sx);

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
