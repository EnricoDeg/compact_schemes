/*
 * @file numerics_device_rtc1.hpp
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

#ifndef CANARD_KERNELS_NUMERICS_DEVICE_RTC1_HPP
#define CANARD_KERNELS_NUMERICS_DEVICE_RTC1_HPP

#include "common/parameters.hpp"
#include "common/data_types.hpp"

#include "cuda/kernels/definitions.hpp"
#include "cuda/kernels/functional.hpp"
#include "cuda/kernels/transforms.hpp"

template<
unsigned int BlockSizeLocal,
unsigned int LxiLocal,
int NumberOfIterations,
typename Type>
CANARD_DEVICE inline void PCR_solver(unsigned int thread_local_idx,
                                     Type * sa,
                                     Type * sb,
                                     Type * sc,
                                     Type * srhs,
                                     Type * sx)
{
    int group_id = thread_local_idx % LxiLocal;
    int thread_id_in_group = thread_local_idx / LxiLocal;

    int right, left;
    Type k1, k2, tb, trhs, ta, tc;
    int stride = 1;

    details::static_for<0, NumberOfIterations, 1>{}([&](auto j)
    {
        right = (thread_id_in_group + stride) * LxiLocal + group_id;
        if(right >= BlockSizeLocal * LxiLocal + group_id)
            right = BlockSizeLocal * LxiLocal + group_id - 1;

        left = (thread_id_in_group - stride) * LxiLocal + group_id;
        if(left < 0)
            left = 0;

        k1 = sa[thread_id_in_group * LxiLocal + group_id] / sb[left];
        k2 = sc[thread_id_in_group * LxiLocal + group_id] / sb[right];

        tb   = sb[thread_id_in_group * LxiLocal + group_id] - sc[left] * k1 - sa[right] * k2;
        trhs = srhs[thread_id_in_group * LxiLocal + group_id] - srhs[left] * k1 - srhs[right] * k2;
        ta   = -sa[left] * k1;
        tc   = -sc[right] * k2;

        __syncthreads();

        sb[thread_id_in_group * LxiLocal + group_id]   = tb;
        srhs[thread_id_in_group * LxiLocal + group_id] = trhs;
        sa[thread_id_in_group * LxiLocal + group_id]   = ta;
        sc[thread_id_in_group * LxiLocal + group_id]   = tc;

        stride <<= 1; //stride *= 2;

        __syncthreads();
    });

    if(thread_id_in_group < BlockSizeLocal / 2)
    {
        // Solve 2x2 systems
        int i    = thread_id_in_group * LxiLocal + group_id;
        int j    = (thread_id_in_group + stride) * LxiLocal + group_id;
        Type det = sb[j] * sb[i] - sc[i] * sa[j];
        det      = static_cast<Type>(1) / det;

        sx[i] = (sb[j] * srhs[i] - sc[i] * srhs[j]) * det;
        sx[j] = (srhs[j] * sb[i] - srhs[i] * sa[j]) * det;
    }
}

template<
unsigned int nstart,
unsigned int nend,
unsigned int BlockSizeLocal,
unsigned int LxiLocal,
typename Type>
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
    int group_id = thread_local_idx % LxiLocal;
    int thread_id_in_group = thread_local_idx / LxiLocal;
    if(thread_id_in_group == 0)
    {
        if constexpr(nstart == 0)
        {
            srhs[0 * LxiLocal + group_id] = (ab00 * sx[0 * LxiLocal + group_id] +
                                        ab01 * sx[1 * LxiLocal + group_id] +
                                        ab02 * sx[2 * LxiLocal + group_id]);
            sa[0 * LxiLocal + group_id] = 0.0;
            sb[0 * LxiLocal + group_id] = 1.0;
            sc[0 * LxiLocal + group_id] = alpha01;
        }
        else
        {
            srhs[0 * LxiLocal + group_id] = 0;
            details::static_for<0, lmd, 1>{}([&](auto i)
            {
                srhs[0] += pbci[i] * sx[i * LxiLocal + group_id];
            });
            srhs[0 * LxiLocal + group_id] += recv[face_idx];
            sa[0 * LxiLocal + group_id] = 0.0;
            sb[0 * LxiLocal + group_id] = 1.0;
            sc[0 * LxiLocal + group_id] = alpha;
        }
    }
    else if(thread_id_in_group == 1)
    {
        if constexpr(nstart == 0)
        {
            srhs[1 * LxiLocal + group_id] =
                ab10 * (sx[2 * LxiLocal + group_id] - sx[0 * LxiLocal + group_id]);
            sa[1 * LxiLocal + group_id] = alpha10;
            sb[1 * LxiLocal + group_id] = 1.0;
            sc[1 * LxiLocal + group_id] = alpha10;
        }
        else
        {
            srhs[1 * LxiLocal + group_id] = 0;
            details::static_for<0, lmd, 1>{}([&](auto i)
            {
                srhs[1 * LxiLocal + group_id] += pbci[lmd + i] * sx[i * LxiLocal + group_id];
            });
            srhs[1 * LxiLocal + group_id] += recv[face_size + face_idx];
            sa[1 * LxiLocal + group_id] = alpha;
            sb[1 * LxiLocal + group_id] = 1.0;
            sc[1 * LxiLocal + group_id] = alpha;
        }
    }
    else if(thread_id_in_group == BlockSizeLocal -1)
    {
        if constexpr(nend == 0)
        {
            srhs[(BlockSizeLocal - 1) * LxiLocal + group_id] =
                                -(ab00 * sx[(BlockSizeLocal - 1) * LxiLocal + group_id] +
                                    ab01 * sx[(BlockSizeLocal - 2) * LxiLocal + group_id] +
                                    ab02 * sx[(BlockSizeLocal - 3) * LxiLocal + group_id]);
            sa[(BlockSizeLocal - 1) * LxiLocal + group_id] = alpha01;
            sb[(BlockSizeLocal - 1) * LxiLocal + group_id] = 1.0;
            sc[(BlockSizeLocal - 1) * LxiLocal + group_id] = 0.0;
        }
        else
        {
            srhs[(BlockSizeLocal - 1) * LxiLocal + group_id] = 0;
            details::static_for<0, lmd, 1>{}([&](auto i)
            {
                srhs[(BlockSizeLocal - 1) * LxiLocal + group_id] += pbci[i] *
                    sx[(BlockSizeLocal - 1 - i) * LxiLocal + group_id];
            });
            srhs[(BlockSizeLocal - 1) * LxiLocal + group_id] += recv[2 * face_size + face_idx];
            srhs[(BlockSizeLocal - 1) * LxiLocal + group_id] *= -1;
            sa[(BlockSizeLocal - 1) * LxiLocal + group_id] = alpha;
            sb[(BlockSizeLocal - 1) * LxiLocal + group_id] = 1.0;
            sc[(BlockSizeLocal - 1) * LxiLocal + group_id] = 0.0;
        }
    }
    else if(thread_id_in_group == BlockSizeLocal - 2)
    {
        if constexpr(nend == 0)
        {
            srhs[(BlockSizeLocal - 2) * LxiLocal + group_id] =
                -ab10*(sx[(BlockSizeLocal - 3) * LxiLocal + group_id] -
                    sx[(BlockSizeLocal - 1) * LxiLocal + group_id]);
            sa[(BlockSizeLocal - 2) * LxiLocal + group_id] = alpha10;
            sb[(BlockSizeLocal - 2) * LxiLocal + group_id] = 1.0;
            sc[(BlockSizeLocal - 2) * LxiLocal + group_id] = alpha10;
        }
        else
        {
            srhs[(BlockSizeLocal - 2) * LxiLocal + group_id] = 0;
            details::static_for<0, lmd, 1>{}([&](auto i)
            {
                srhs[(BlockSizeLocal - 2) * LxiLocal + group_id] += pbci[lmd + i] *
                    sx[(BlockSizeLocal - 1 - i) * LxiLocal + group_id];
            });
            srhs[(BlockSizeLocal - 2) * LxiLocal + group_id] += recv[3 * face_size + face_idx];
            srhs[(BlockSizeLocal - 2) * LxiLocal + group_id] *= -1;
            sa[(BlockSizeLocal - 2) * LxiLocal + group_id] = alpha;
            sb[(BlockSizeLocal - 2) * LxiLocal + group_id] = 1.0;
            sc[(BlockSizeLocal - 2) * LxiLocal + group_id] = alpha;
        }
    }
    else
    {
        srhs[thread_id_in_group * LxiLocal + group_id] =
            aa * (sx[(thread_id_in_group + 1) * LxiLocal + group_id] -
                  sx[(thread_id_in_group - 1) * LxiLocal + group_id]) +
            bb * (sx[(thread_id_in_group + 2) * LxiLocal + group_id]-
                  sx[(thread_id_in_group - 2) * LxiLocal + group_id]);
        sa[thread_id_in_group * LxiLocal + group_id] = alpha;
        sb[thread_id_in_group * LxiLocal + group_id] = 1.0;
        sc[thread_id_in_group * LxiLocal + group_id] = alpha;
    }
}

template<
unsigned int Axis,
unsigned int AxisOut,
unsigned int BlockSize,
typename Type>
CANARD_DEVICE void deriv_kernel_impl(Type *infield,
                                     Type *outfield,
                                     Type *recv,
                                     Type *pbci,
                                     Type *drva,
                                     t_dcomp dcomp_info,
                                     unsigned int variable_id,
                                     unsigned int component_id,
                                     Type *sx,
                                     Type *srhs,
                                     Type *sa,
                                     Type *sb,
                                     Type *sc)
{
    constexpr int NumberOfIterations = ITERS;
    constexpr unsigned int nstart = NSTART;
    constexpr unsigned int nend = NEND;
    constexpr unsigned int blocksize_local = BLOCKSIZE_LOCAL;
    constexpr unsigned int lxi_local = LXI_LOCAL;

    unsigned int thread_local_idx = get_thread_local_idx();
    unsigned int gmem_idx, group_id, thread_id_in_group, face_idx, face_size, face_stride;

    if constexpr(Axis == 0)
    {
        group_id = thread_local_idx % lxi_local;
        thread_id_in_group = thread_local_idx / lxi_local;
        gmem_idx = blockIdx.y  * dcomp_info.let * dcomp_info.lxi   +
                   blockIdx.x  * dcomp_info.lxi    +
                   thread_id_in_group;
        face_stride = get_face_stride<Axis>(dcomp_info);
        face_idx = blockIdx.y * face_stride + blockIdx.x * lxi_local + group_id;
        face_size = get_face_size<Axis>(dcomp_info);
    }
    else if constexpr(Axis == 1)
    {
        group_id = thread_local_idx % lxi_local;
        thread_id_in_group = thread_local_idx / lxi_local;
        gmem_idx = blockIdx.y  * dcomp_info.let * dcomp_info.lxi   +
                   blockIdx.x  * lxi_local    +
                   thread_id_in_group * dcomp_info.lxi +
                   group_id;
        face_stride = get_face_stride<Axis>(dcomp_info);
        face_idx = blockIdx.y * face_stride + blockIdx.x * lxi_local + group_id;
        face_size = get_face_size<Axis>(dcomp_info);
    }
    else if constexpr(Axis == 2)
    {
        group_id = thread_local_idx % lxi_local;
        thread_id_in_group = thread_local_idx / lxi_local;
        gmem_idx = blockIdx.y  * dcomp_info.lxi +
                   blockIdx.x  * lxi_local +
                   thread_id_in_group * dcomp_info.lxi * dcomp_info.let +
                   group_id;
        face_stride = get_face_stride<Axis>(dcomp_info);
        face_idx = blockIdx.y * face_stride + blockIdx.x * lxi_local + group_id;
        face_size = get_face_size<Axis>(dcomp_info);
    }

    // load input field in shared memory
    sx[thread_local_idx] = infield[gmem_idx + component_id * dcomp_info.lmx];
    __syncthreads();

    // load shared memory: tridiagonal matrix and rhs
    Type * recv_variable = recv + variable_id * 2 * 2 * face_size;
    build_system<nstart, nend, blocksize_local, lxi_local>(thread_local_idx,
                 face_idx, face_size,
                 sa, sb, sc, srhs, sx,
                 recv_variable, pbci);

    __syncthreads();

    // Parallel cyclic reduction
    PCR_solver<blocksize_local, lxi_local, NumberOfIterations>(thread_local_idx,
        sa, sb, sc, srhs, sx);

    __syncthreads();

    // store results
    outfield[gmem_idx + AxisOut * dcomp_info.lmx] = sx[thread_local_idx];

    if(thread_id_in_group == 0)
    {
        unsigned int idx = variable_id * face_size + face_idx;
        drva[idx] = sx[thread_id_in_group * lxi_local + group_id];
    }
    if(thread_id_in_group == lxi_local - 1)
    {
        unsigned int idx = NumberOfVariables * face_size +
            variable_id * face_size + face_idx;
        drva[idx] = sx[thread_id_in_group * lxi_local + group_id];
    }
}

#endif
