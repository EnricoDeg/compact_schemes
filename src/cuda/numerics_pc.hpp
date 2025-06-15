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

#ifndef CANARD_CUDA_NUMERICS_PC_HPP
#define CANARD_CUDA_NUMERICS_PC_HPP

#include <cmath>

#include "cuda/kernels/numerics_pc.hpp"
#include "cuda/check.hpp"
#include "cuda/dispatch.hpp"

template<
unsigned int Axis,
typename Type
>
struct gpu_buffer
{
    gpu_buffer(Type *infield_, Type *buffer_, Type * pbco_, t_dcomp dcomp_info_)
        : infield{infield_},
          buffer{buffer_},
          pbco{pbco_},
          dcomp_info{dcomp_info_}
    {
    }

    void fill(int istart, int increment, int buffer_offset, cudaStream_t *stream)
    {
        static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");
    
        unsigned int blockSize;
        unsigned int blockPerGrid;
        if constexpr(Axis == 0)
        {
            blockSize = dcomp_info.let;
            blockPerGrid = dcomp_info.lze;
        }
        else if constexpr(Axis == 1)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.lze;
        }
        else if constexpr(Axis == 2)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.let;
        }
    
        dim3 threadsPerBlock(blockSize);
        dim3 blocksPerGrid(blockPerGrid);
    
        TIME(blocksPerGrid, threadsPerBlock, 0, *stream, true,
            CANARD_KERNEL_NAME(fill_gpu_buffer_kernel<Axis>),
            infield, buffer, pbco, dcomp_info, istart, increment, buffer_offset);
    }

    Type * infield;
    Type * buffer;
    Type * pbco;
    t_dcomp dcomp_info;
};

#endif
