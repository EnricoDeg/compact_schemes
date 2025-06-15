/*
 * @file numerics_rtc.hpp
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

#ifndef CANARD_KERNELS_NUMERICS_RTC_HPP
#define CANARD_KERNELS_NUMERICS_RTC_HPP

#include "cuda/kernels/numerics_device_rtc.hpp"

template<
unsigned int BlockSize,
unsigned int Axis,
typename Type>
CANARD_GLOBAL void deriv_kernel_1d(Type *infield,
                                   Type *outfield,
                                   Type *recv,
                                   Type *pbci,
                                   Type *drva,
                                   t_dcomp dcomp_info,
                                   unsigned int variable_id)
{
    CANARD_SHMEM Type sdata[5 * BlockSize];

    Type * sx, *srhs, *sa, *sb, *sc;

    sx   = sdata;
    srhs = sdata + BlockSize;
    sa   = sdata + 2 * BlockSize;
    sb   = sdata + 3 * BlockSize;
    sc   = sdata + 4 * BlockSize;

    deriv_kernel_impl<Axis, 0, BlockSize>(infield,
                                          outfield,
                                          recv,
                                          pbci,
                                          drva,
                                          dcomp_info,
                                          variable_id,
                                          0,
                                          sx,
                                          srhs,
                                          sa,
                                          sb,
                                          sc);
}

template<
unsigned int BlockSize,
unsigned int Axis,
typename Type>
CANARD_GLOBAL void deriv_kernel_2d(Type *infield,
                                   Type *outfield,
                                   Type *recv,
                                   Type *pbci,
                                   Type *drva,
                                   t_dcomp dcomp_info,
                                   unsigned int variable_id,
                                   unsigned int component_id)
{
    CANARD_SHMEM Type sdata[5 * BlockSize];

    Type * sx, *srhs, *sa, *sb, *sc;

    sx   = sdata;
    srhs = sdata + BlockSize;
    sa   = sdata + 2 * BlockSize;
    sb   = sdata + 3 * BlockSize;
    sc   = sdata + 4 * BlockSize;

    deriv_kernel_impl<Axis, Axis, BlockSize>(infield,
                                             outfield,
                                             recv,
                                             pbci,
                                             drva,
                                             dcomp_info,
                                             variable_id,
                                             component_id,
                                             sx,
                                             srhs,
                                             sa,
                                             sb,
                                             sc);
}

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

#endif
