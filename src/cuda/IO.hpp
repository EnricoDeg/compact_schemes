/*
 * @file IO.hpp
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

#ifndef CANARD_CUDA_IO_HPP
#define CANARD_CUDA_IO_HPP

#include "common/parameters.hpp"
#include "common/utils.hpp"

#include "cuda/common.hpp"
#include "cuda/kernels/IO.hpp"
#include "cuda/check.hpp"
#include "cuda/dispatch.hpp"
#include "cuda/common.hpp"

template<typename Type>
void fill_IO_buffer(Type *vart,
                    Type *qb,
                    Type *qo,
                    Type *qa,
                    t_patch<Type> *patch,
                    Type fctr,
                    Type dtsum,
                    t_point<Type> umf,
                    bool is_output_step,
                    unsigned int size,
                    int file_number)
{
    unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(size, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGrid);

    TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
        CANARD_KERNEL_NAME(fill_IO_buffer_kernel),
        vart, qb, qo, qa, patch, fctr, dtsum,
        umf, is_output_step, size, file_number);
}

template<typename Type>
void reset_IO_buffer(Type *qb,
                     unsigned int size)
{
    unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(size, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGrid);

    TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
        CANARD_KERNEL_NAME(reset_IO_buffer_kernel),
        qb, size);
}

#endif
