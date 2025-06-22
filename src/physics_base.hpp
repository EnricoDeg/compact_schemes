/*
 * @file physics_base.hpp
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

#ifndef CANARD_PHYSICS_BASE_HPP
#define CANARD_PHYSICS_BASE_HPP

#include <cstddef>
#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "cuda/common.hpp"

template<bool EnableViscous, typename Type>
struct physics_base
{
    physics_base(t_dcomp dcomp_info)
    {
        size_t nelements = dcomp_info.lmx;

        buffer       = allocate_cuda<Type>(NumberOfSpatialDims * NumberOfVariables * nelements);
        buffer_deriv = allocate_cuda<Type>(NumberOfSpatialDims * NumberOfVariables * nelements);
        buffer_ss0   = allocate_cuda<Type>(nelements);
        buffer_ss1   = allocate_cuda<Type>(nelements);
        buffer_ss2   = allocate_cuda<Type>(nelements);

        if constexpr(EnableViscous)
        {
            stress_tensor = allocate_cuda<t_stress_tensor<Type>>(1);

            Type *tmp = allocate_cuda<Type>(nelements);
            memcpy_cuda_h2d(&stress_tensor->xx, &tmp, 1);

            tmp =  allocate_cuda<Type>(nelements);
            memcpy_cuda_h2d(&stress_tensor->yy, &tmp, 1);

            tmp =  allocate_cuda<Type>(nelements);
            memcpy_cuda_h2d(&stress_tensor->zz, &tmp, 1);

            tmp =  allocate_cuda<Type>(nelements);
            memcpy_cuda_h2d(&stress_tensor->xy, &tmp, 1);

            tmp =  allocate_cuda<Type>(nelements);
            memcpy_cuda_h2d(&stress_tensor->yz, &tmp, 1);

            tmp =  allocate_cuda<Type>(nelements);
            memcpy_cuda_h2d(&stress_tensor->zx, &tmp, 1);

            heat_fluxes = allocate_cuda<t_heat_fluxes<Type>>(1);

            tmp =  allocate_cuda<Type>(nelements);
            memcpy_cuda_h2d(&heat_fluxes->xx, &tmp, 1);

            tmp =  allocate_cuda<Type>(nelements);
            memcpy_cuda_h2d(&heat_fluxes->yy, &tmp, 1);

            tmp =  allocate_cuda<Type>(nelements);
            memcpy_cuda_h2d(&heat_fluxes->zz, &tmp, 1);
        }

    }

    ~physics_base()
    {
        free_cuda(buffer);
        free_cuda(buffer_deriv);
        free_cuda(buffer_ss0);
        free_cuda(buffer_ss1);
        free_cuda(buffer_ss2);

        if constexpr(EnableViscous)
        {
            Type *tmp;
            memcpy_cuda_d2h(&tmp, &stress_tensor->xx, 1);
            free_cuda(tmp);

            memcpy_cuda_d2h(&tmp, &stress_tensor->yy, 1);
            free_cuda(tmp);

            memcpy_cuda_d2h(&tmp, &stress_tensor->zz, 1);
            free_cuda(tmp);

            memcpy_cuda_d2h(&tmp, &stress_tensor->xy, 1);
            free_cuda(tmp);

            memcpy_cuda_d2h(&tmp, &stress_tensor->yz, 1);
            free_cuda(tmp);

            memcpy_cuda_d2h(&tmp, &stress_tensor->zx, 1);
            free_cuda(tmp);

            free_cuda(stress_tensor);

            memcpy_cuda_d2h(&tmp, &heat_fluxes->xx, 1);
            free_cuda(tmp);

            memcpy_cuda_d2h(&tmp, &heat_fluxes->yy, 1);
            free_cuda(tmp);

            memcpy_cuda_d2h(&tmp, &heat_fluxes->zz, 1);
            free_cuda(tmp);

            free_cuda(heat_fluxes);
        }
    }

    void movef(Type dtko, Type dtk, Type timo)
    {

    }

    Type * buffer;
    Type * buffer_deriv;
    Type * buffer_ss0;
    Type * buffer_ss1;
    Type * buffer_ss2;

    t_stress_tensor<Type> * stress_tensor;
    t_heat_fluxes<Type> * heat_fluxes;
};

#endif
