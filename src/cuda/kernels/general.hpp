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

#ifndef CANARD_KERNELS_GENERAL_HPP
#define CANARD_KERNELS_GENERAL_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "cuda/kernels/definitions.hpp"
#include "cuda/kernels/common.hpp"
#include "cuda/kernels/transforms.hpp"

template<typename Type>
CANARD_GLOBAL void init_main_loop_kernel(Type *qa,
                                         Type *qo,
                                         unsigned int size)
{
    unsigned int thread_id = get_thread_global_idx();

    if(thread_id < size)
    {
        for(unsigned int i = 0; i < NumberOfVariables; ++i)
        {
            qo[thread_id + i * size] = qa[thread_id + i * size];
        }
    }
}

template<typename Type>
CANARD_GLOBAL void init_runge_kutta_kernel(Type *de,
                                           Type *qa,
                                           Type *pressure,
                                           Type *ss,
                                           Type srefp1dre,
                                           Type srefoo,
                                           unsigned int size)
{
    unsigned int thread_id = get_thread_global_idx();

    if(thread_id < size)
    {
        de[thread_id + 0 * size] = 1.0 / qa[thread_id];
        de[thread_id + 1 * size] = qa[thread_id + 1 * size] * de[thread_id];
        de[thread_id + 2 * size] = qa[thread_id + 2 * size] * de[thread_id];
        de[thread_id + 3 * size] = qa[thread_id + 3 * size] * de[thread_id];
        pressure[thread_id] = gamm1 * ( qa[thread_id + 4 * size] - 0.5 *
            ( qa[thread_id + 1 * size] * de[thread_id + 1 * size] +
              qa[thread_id + 2 * size] * de[thread_id + 2 * size] +
              qa[thread_id + 3 * size] * de[thread_id + 3 * size] ) );
        de[thread_id + 4 * size] = gam * pressure[thread_id] * de[thread_id];
        ss[thread_id] = srefp1dre * std::pow(de[thread_id + 4 * size], 1.5) /
            ( de[thread_id + 4 * size] + srefoo );
    }
}

#endif
