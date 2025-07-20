/*
 * @file general.hpp
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

#ifndef CANARD_CUDA_GENERAL_HPP
#define CANARD_CUDA_GENERAL_HPP

#include "common/parameters.hpp"
#include "common/utils.hpp"

#include "cuda/common.hpp"
#include "cuda/kernels/general.hpp"
#include "cuda/check.hpp"
#include "cuda/dispatch.hpp"
#include "cuda/common.hpp"

template<typename Type>
void init_main_loop(Type *qa,
                    Type *qo,
                    unsigned int size)
{
    unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(size, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGrid);

    TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
        CANARD_KERNEL_NAME(init_main_loop_kernel),
        qa, qo, size);
}

template<typename Type>
void init_runge_kutta(Type *de,
                      Type *qa,
                      Type *pressure,
                      Type *ss,
                      Type srefp1dre,
                      Type srefoo,
                      unsigned int size)
{
    unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(size, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGrid);

    TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
        CANARD_KERNEL_NAME(init_runge_kutta_kernel),
        de, qa, pressure, ss, srefp1dre, srefoo, size);
}

#endif
