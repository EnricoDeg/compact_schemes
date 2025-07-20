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

#ifndef CANARD_KERNELS_IO_HPP
#define CANARD_KERNELS_IO_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "cuda/kernels/definitions.hpp"
#include "cuda/kernels/common.hpp"
#include "cuda/kernels/transforms.hpp"

template<typename Type>
CANARD_GLOBAL
void fill_IO_buffer_kernel(Type *vart,
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
    unsigned int thread_id = get_thread_global_idx();
    if(is_output_step)
    {
        if(file_number == 0)
        {
            for(unsigned int i = 0; i < NumberOfVariables; ++i)
            {
                qb[thread_id + i * size] = qo[thread_id + i * size];
            }

        }
        else
        {
            for(unsigned int i = 0; i < NumberOfVariables; ++i)
            {
                qb[thread_id + i * size] += fctr*(qo[thread_id + i * size] +
                                                  qa[thread_id + i * size]);
                qb[thread_id + i * size] /= dtsum;
            }
        }

        Type rr = 1.0 / qb[thread_id];

        vart[thread_id + 0 * size] = patch->x[thread_id];
        vart[thread_id + 1 * size] = patch->y[thread_id];
        vart[thread_id + 2 * size] = patch->z[thread_id];

        unsigned int offset = NumberOfSpatialDims * size;
        vart[thread_id + 0 * size + offset] = qb[thread_id];
        vart[thread_id + 1 * size + offset] = rr * qb[thread_id + 1 * size] + umf.x;
        vart[thread_id + 2 * size + offset] = rr * qb[thread_id + 2 * size] + umf.y;
        vart[thread_id + 3 * size + offset] = rr * qb[thread_id + 3 * size] + umf.z;
        vart[thread_id + 4 * size + offset] = gamm1 * (qb[thread_id + 4 * size] -
            0.5 * rr * (qb[thread_id + 1 * size] * qb[thread_id + 1 * size] +
                        qb[thread_id + 2 * size] * qb[thread_id + 2 * size] +
                        qb[thread_id + 3 * size] * qb[thread_id + 3 * size]));
    }
    else
    {
        for(unsigned int i = 0; i < NumberOfVariables; ++i)
        {
            qb[thread_id + i * size] += fctr*(qo[thread_id + i * size] +
                                              qa[thread_id + i * size]);
        }
    }
}

template<typename Type>
CANARD_GLOBAL
void reset_IO_buffer_kernel(Type *qb,
                            unsigned int size)
{
    unsigned int thread_id = get_thread_global_idx();
    for(unsigned int i = 0; i < NumberOfVariables; ++i)
    {
        qb[thread_id + i * size] = 0.0;
    }
}

#endif
