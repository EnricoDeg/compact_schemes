/*
 * @file sponge.hpp
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

#ifndef CANARD_KERNELS_SPONGE_HPP
#define CANARD_KERNELS_SPONGE_HPP

#include "common/parameters.hpp"
#include "cuda/kernels/definitions.hpp"

template<typename Type>
CANARD_GLOBAL void spongego_kernel(Type *qa,
                                   Type *de,
                                   Type *asz,
                                   Type *bsz,
                                   int *lcsz,
                                   int lsz,
                                   unsigned int lmx)
{
    unsigned int thread_id = get_thread_global_idx();
    if(thread_id < lsz + 1)
    {
        unsigned int l = lcsz[thread_id];
        de[l + 0 * lmx] += asz[thread_id] * (qa[l + 0 * lmx] - 1.0);
        de[l + 1 * lmx] += bsz[thread_id] * (qa[l + 1 * lmx] - 0.0);
        de[l + 2 * lmx] += bsz[thread_id] * (qa[l + 2 * lmx] - 0.0);
        de[l + 3 * lmx] += bsz[thread_id] * (qa[l + 3 * lmx] - 0.0);
        de[l + 4 * lmx] += asz[thread_id] * (qa[l + 4 * lmx] - hamhamm1);
    }
}

#endif
