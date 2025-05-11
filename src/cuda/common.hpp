/*
 * @file common.hpp
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

#ifndef CANARD_CUDA_COMMON_HPP
#define CANARD_CUDA_COMMON_HPP

#include "check.hpp"
#include "cuda/check.hpp"

template<typename Tdata>
Tdata * allocate_cuda(size_t elements) {
    Tdata *p;
    check_cuda( cudaMalloc(&p, elements * sizeof(Tdata)) );
    return p;
}

template<typename Tdata>
void free_cuda(Tdata *p)
{
    check_cuda(cudaFree(p));
}

template<typename Tdata>
void memcpy_cuda_h2d(Tdata *dst, Tdata *src, size_t nelements)
{
    cudaMemcpy(dst, src, nelements * sizeof(Tdata), cudaMemcpyHostToDevice);
}

template<typename Tdata>
void memcpy_cuda_d2h(Tdata *dst, Tdata *src, size_t nelements)
{
    cudaMemcpy(dst, src, nelements * sizeof(Tdata), cudaMemcpyDeviceToHost);
}

#endif
