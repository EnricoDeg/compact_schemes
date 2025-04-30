/*
 * @file reductionShMem.hpp
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

#ifndef CANARD_REDUCTIONSHMEM_HPP
#define CANARD_REDUCTIONSHMEM_HPP

#include "cuda/kernels/definitions.hpp"

template <
unsigned int blockSize,
typename Type
>
CANARD_DEVICE
void warpReduceShMem(Type* sdata, unsigned int tid) {

    if (blockSize >= 64) {
        sdata[tid] = max(sdata[tid], sdata[tid + 32]);
        __syncwarp();
    }
    if (blockSize >= 32) {
        sdata[tid] = max(sdata[tid], sdata[tid + 16]);
        __syncwarp();
    }
    if (blockSize >= 16) {
        sdata[tid] = max(sdata[tid], sdata[tid +  8]);
        __syncwarp();
    }
    if (blockSize >= 8) {
        sdata[tid] = max(sdata[tid], sdata[tid +  4]);
        __syncwarp();
    }
    if (blockSize >= 4) {
        sdata[tid] = max(sdata[tid], sdata[tid +  2]);
        __syncwarp();
    }
    if (blockSize >= 2) {
        sdata[tid] = max(sdata[tid], sdata[tid +  1]);
        __syncwarp();
    }
};

template<
unsigned int blockSize,
typename Type,
>
CANARD_DEVICE
void blockReduceShMemUnroll(Type* sdata, unsigned int tid, op_t<T> &Op) {

    if (blockSize >= 1024) {
        if (tid < 512) sdata[tid] = max(sdata[tid], sdata[tid + 512]);
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] = max(sdata[tid], sdata[tid + 256]);
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] = max(sdata[tid], sdata[tid + 128]);
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64) sdata[tid] = max(sdata[tid], sdata[tid +  64]);
        __syncthreads();
    }

    if (tid < 32)
        warpReduceShMem<blockSize, T, op_t>(sdata, tid, Op);
}

#endif
