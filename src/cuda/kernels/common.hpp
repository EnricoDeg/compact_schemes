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

#ifndef CANARD_KERNELS_COMMON_HPP
#define CANARD_KERNELS_COMMON_HPP

#include "cuda/kernels/definitions.hpp"
#include "cuda/kernels/functional.hpp"

template<unsigned int Size, typename Type>
CANARD_DEVICE void MatVecMul(Type *mat, Type * vec, Type * res)
{
    details::static_for<0, Size, 1>{}([&](int i)
    {
        Type value = 0;
        details::static_for<0, Size, 1>{}([&](int j)
        {
            value += mat[i * Size + j] * vec[j];
        });
        res[i] = value;
    });
}

template<typename Type>
CANARD_DEVICE inline void atomicMax(Type * addr, Type val) {
    if (*addr >= val) return;

    unsigned int *const addr_as_ui = (unsigned int *)addr;
    unsigned int old = *addr_as_ui, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;
        old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
    } while (assumed != old);
}

#endif
