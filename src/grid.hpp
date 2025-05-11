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

#ifndef CANARD_PHYSICS_HPP
#define CANARD_PHYSICS_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "cuda/common.hpp"

template<typename Type>
struct grid
{
    grid(t_dcomp dcomp_info)
    {
        size_t nelements = dcomp_info.lmx;

        xim = allocate_cuda<Type>(NumberOfSpatialDims * nelements);
        etm = allocate_cuda<Type>(NumberOfSpatialDims * nelements);
        zem = allocate_cuda<Type>(NumberOfSpatialDims * nelements);
    }

    ~grid()
    {
        free_cuda(xim);
        free_cuda(etm);
        free_cuda(zem);
    }

    Type * xim;
    Type * etm;
    Type * zem;
};

#endif