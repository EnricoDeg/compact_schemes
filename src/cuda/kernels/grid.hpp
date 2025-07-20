/*
 * @file grid.hpp
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

#ifndef CANARD_KERNELS_GRID_HPP
#define CANARD_KERNELS_GRID_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "cuda/kernels/definitions.hpp"
#include "cuda/kernels/common.hpp"
#include "cuda/kernels/transforms.hpp"

template<typename Type>
CANARD_GLOBAL void grid_metrics_fill_kernel(Type *xim,
                                            Type *etm,
                                            Type *zem,
                                            Type *qok,
                                            Type *qak,
                                            Type *dek,
                                            unsigned int size)
{
    unsigned int thread_id = get_thread_global_idx();

    if(thread_id < size)
    {
        xim[thread_id + 0 * size] = qak[thread_id + 1 * size] *
                                    dek[thread_id + 2 * size] -
                                    dek[thread_id + 1 * size] *
                                    qak[thread_id + 2 * size];
        xim[thread_id + 1 * size] = dek[thread_id + 1 * size] *
                                    qok[thread_id + 2 * size] -
                                    qok[thread_id + 1 * size] *
                                    dek[thread_id + 2 * size];
        xim[thread_id + 2 * size] = qok[thread_id + 1 * size] *
                                    qak[thread_id + 2 * size] -
                                    qak[thread_id + 1 * size] *
                                    qok[thread_id + 2 * size];
        etm[thread_id + 0 * size] = qak[thread_id + 2 * size] *
                                    dek[thread_id + 0 * size] -
                                    dek[thread_id + 2 * size] *
                                    qak[thread_id + 0 * size];
        etm[thread_id + 1 * size] = dek[thread_id + 2 * size] *
                                    qok[thread_id + 0 * size] -
                                    qok[thread_id + 2 * size] *
                                    dek[thread_id + 0 * size];
        etm[thread_id + 2 * size] = qok[thread_id + 2 * size] *
                                    qak[thread_id + 0 * size] -
                                    qak[thread_id + 2 * size] *
                                    qok[thread_id + 0 * size];
        zem[thread_id + 0 * size] = qak[thread_id + 0 * size] *
                                    dek[thread_id + 1 * size] -
                                    dek[thread_id + 0 * size] *
                                    qak[thread_id + 1 * size];
        zem[thread_id + 1 * size] = dek[thread_id + 0 * size] *
                                    qok[thread_id + 1 * size] -
                                    qok[thread_id + 0 * size] *
                                    dek[thread_id + 1 * size];
        zem[thread_id + 2 * size] = qok[thread_id + 0 * size] *
                                    qak[thread_id + 1 * size] -
                                    qak[thread_id + 0 * size] *
                                    qok[thread_id + 1 * size];
    }
}

template<typename Type>
CANARD_GLOBAL void grid_metrics_yaco_fill_kernel(Type *yaco,
                                                 Type *xim,
                                                 Type *etm,
                                                 Type *zem,
                                                 Type *qok,
                                                 Type *qak,
                                                 Type *dek,
                                                 unsigned int size)
{
    unsigned int thread_id = get_thread_global_idx();

    if(thread_id < size)
    {
        yaco[thread_id] = 3.0 /
            (qok[thread_id + 0 * size] * xim[thread_id + 0 * size] +
             qok[thread_id + 1 * size] * etm[thread_id + 0 * size] +
             qok[thread_id + 2 * size] * zem[thread_id + 0 * size] +
             qak[thread_id + 0 * size] * xim[thread_id + 1 * size] +
             qak[thread_id + 1 * size] * etm[thread_id + 1 * size] +
             qak[thread_id + 2 * size] * zem[thread_id + 1 * size] +
             dek[thread_id + 0 * size] * xim[thread_id + 2 * size] +
             dek[thread_id + 1 * size] * etm[thread_id + 2 * size] +
             dek[thread_id + 2 * size] * zem[thread_id + 2 * size]);
    }
}

#endif
