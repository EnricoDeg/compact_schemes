/*
 * @file physics.hpp
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

#ifndef CANARD_KERNELS_PHYSICS_HPP
#define CANARD_KERNELS_PHYSICS_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "cuda/kernels/definitions.hpp"
#include "cuda/kernels/common.hpp"
#include "cuda/kernels/transforms.hpp"
#include "cuda/kernels/reductionShMem.hpp"
#include "cuda/kernels/functional.hpp"
#include "cuda/kernels/vector_types.hpp"

template<typename Type>
CANARD_GLOBAL void init_physics_kernel(Type *qa,
                                       t_patch<Type> *patch,
                                       unsigned int size)
{
    unsigned int thread_id = get_thread_global_idx();

    if(thread_id < size)
    {
        Type radv = 1.0;
        Type k1 = 12.5;
        Type k2 = 1.0;
        Type vee[NumberOfSpatialDims];

        Type ao       = k2 / 2.0 / pi * std::sqrt( std::exp( 1.0 - k1 * k1 * 
            (patch->x[thread_id] * patch->x[thread_id] +
             patch->y[thread_id] * patch->y[thread_id]) / (radv * radv)));
        Type bo       = std::pow(( 1.0 - 0.5 * gamm1 * ao * ao ), hamm1);
        vee[0]        =  k1 * patch->y[thread_id] * ao / radv;
        vee[1]        = -k1 * patch->x[thread_id] * ao / radv;
        vee[2]        = 0.0;
        Type hv2      = 0.5 * ( vee[0] * vee[0] + vee[1] * vee[1] + vee[2] * vee[2] );

        qa[thread_id           ] = bo;
        qa[thread_id + 1 * size] = bo * vee[0];
        qa[thread_id + 2 * size] = bo * vee[1];
        qa[thread_id + 3 * size] = bo * vee[2];
        qa[thread_id + 4 * size] = hamhamm1 * std::pow(bo, gam) + hv2 * bo;
    }
}

template<bool EnableViscous, int VectorSize, typename Type>
CANARD_GLOBAL void calc_fluxes_pre_compute_kernel(Type *buffer,
                                                  Type *qa,
                                                  Type *pressure,
                                                  Type *de,
                                                  Type *xim,
                                                  Type *etm,
                                                  Type *zem,
                                                  t_stress_tensor<Type> * stress_tensor,
                                                  t_heat_fluxes<Type> * heat_fluxes,
                                                  t_point<Type> umf,
                                                  unsigned int size)
{
    using VecType = Vector_Type<Type,VectorSize>;

    unsigned int thread_id = get_thread_global_idx();

    if(thread_id < size / VectorSize)
    {
        // Gmem -> VGPR
        VecType de_vgpr[NumberOfSpatialDims];
        de_vgpr[0] = *reinterpret_cast<VecType*>(de + thread_id * VectorSize + size);
        de_vgpr[1] = *reinterpret_cast<VecType*>(de + thread_id * VectorSize + 2 * size);
        de_vgpr[2] = *reinterpret_cast<VecType*>(de + thread_id * VectorSize + 3 * size);

        VecType xim_vgpr[NumberOfSpatialDims];
        xim_vgpr[0] = *reinterpret_cast<VecType*>(xim + thread_id * VectorSize);
        xim_vgpr[1] = *reinterpret_cast<VecType*>(xim + thread_id * VectorSize + size);
        xim_vgpr[2] = *reinterpret_cast<VecType*>(xim + thread_id * VectorSize + 2 * size);

        VecType etm_vgpr[NumberOfSpatialDims];
        etm_vgpr[0] = *reinterpret_cast<VecType*>(etm + thread_id * VectorSize);
        etm_vgpr[1] = *reinterpret_cast<VecType*>(etm + thread_id * VectorSize + size);
        etm_vgpr[2] = *reinterpret_cast<VecType*>(etm + thread_id * VectorSize + 2 * size);

        VecType zem_vgpr[NumberOfSpatialDims];
        zem_vgpr[0] = *reinterpret_cast<VecType*>(zem + thread_id * VectorSize);
        zem_vgpr[1] = *reinterpret_cast<VecType*>(zem + thread_id * VectorSize + size);
        zem_vgpr[2] = *reinterpret_cast<VecType*>(zem + thread_id * VectorSize + 2 * size);

        VecType qa_vgpr;

        VecType p_vgpr = *reinterpret_cast<VecType*>(pressure + thread_id * VectorSize);

        VecType rr[NumberOfSpatialDims];
        VecType ss[NumberOfSpatialDims];

        // Compute
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[0].vals[j] = de_vgpr[0].vals[j] + umf.x;
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[1].vals[j] = de_vgpr[1].vals[j] + umf.y;
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[2].vals[j] = de_vgpr[2].vals[j] + umf.z;
        });

        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            ss[0].vals[j] = xim_vgpr[0].vals[j] * rr[0].vals[j] +
                            xim_vgpr[1].vals[j] * rr[1].vals[j] +
                            xim_vgpr[2].vals[j] * rr[2].vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            ss[1].vals[j] = etm_vgpr[0].vals[j] * rr[0].vals[j] +
                            etm_vgpr[1].vals[j] * rr[1].vals[j] +
                            etm_vgpr[2].vals[j] * rr[2].vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            ss[2].vals[j] = zem_vgpr[0].vals[j] * rr[0].vals[j] +
                            zem_vgpr[1].vals[j] * rr[1].vals[j] +
                            zem_vgpr[2].vals[j] * rr[2].vals[j];
        });

        qa_vgpr = *reinterpret_cast<VecType*>(qa + thread_id * VectorSize);
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[0].vals[j] = qa_vgpr.vals[j] * ss[0].vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[1].vals[j] = qa_vgpr.vals[j] * ss[1].vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[2].vals[j] = qa_vgpr.vals[j] * ss[2].vals[j];
        });

        details::static_for<0, NumberOfSpatialDims, 1>{}([&](unsigned int j)
        {
            *(VecType*)(buffer + thread_id * VectorSize + j * size + 0 * size * NumberOfSpatialDims) = rr[j];
        });

        qa_vgpr = *reinterpret_cast<VecType*>(qa + thread_id * VectorSize + size);
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[0].vals[j] = qa_vgpr.vals[j] * ss[0].vals[j] +
                            xim_vgpr[0].vals[j] * p_vgpr.vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[1].vals[j] = qa_vgpr.vals[j] * ss[1].vals[j] +
                            etm_vgpr[0].vals[j] * p_vgpr.vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[2].vals[j] = qa_vgpr.vals[j] * ss[2].vals[j] +
                            zem_vgpr[0].vals[j] * p_vgpr.vals[j];
        });

        VecType txy = *reinterpret_cast<VecType*>(stress_tensor->xy + thread_id * VectorSize);
        VecType tzx = *reinterpret_cast<VecType*>(stress_tensor->zx + thread_id * VectorSize);
        if constexpr(EnableViscous)
        {
            VecType txx = *reinterpret_cast<VecType*>(stress_tensor->xx + thread_id * VectorSize);
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[0].vals[j] -= (xim_vgpr[0].vals[j] * txx.vals[j] +
                                  xim_vgpr[1].vals[j] * txy.vals[j] +
                                  xim_vgpr[2].vals[j] * tzx.vals[j]);
            });
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[1].vals[j] -= (etm_vgpr[0].vals[j] * txx.vals[j] +
                                  etm_vgpr[1].vals[j] * txy.vals[j] +
                                  etm_vgpr[2].vals[j] * tzx.vals[j]);
            });
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[2].vals[j] -= (zem_vgpr[0].vals[j] * txx.vals[j] +
                                  zem_vgpr[1].vals[j] * txy.vals[j] +
                                  zem_vgpr[2].vals[j] * tzx.vals[j]);
            });
        }

        details::static_for<0, NumberOfSpatialDims, 1>{}([&](unsigned int j)
        {
            *(VecType*)(buffer + thread_id * VectorSize + j * size + 1 * size * NumberOfSpatialDims) = rr[j];
        });

        qa_vgpr = *reinterpret_cast<VecType*>(qa + thread_id * VectorSize + 2 * size);
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[0].vals[j] = qa_vgpr.vals[j] * ss[0].vals[j] +
                            xim_vgpr[1].vals[j] * p_vgpr.vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[1].vals[j] = qa_vgpr.vals[j] * ss[1].vals[j] +
                            etm_vgpr[1].vals[j] * p_vgpr.vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[2].vals[j] = qa_vgpr.vals[j] * ss[2].vals[j] +
                            zem_vgpr[1].vals[j] * p_vgpr.vals[j];
        });

        VecType tyz = *reinterpret_cast<VecType*>(stress_tensor->yz + thread_id * VectorSize);
        if constexpr(EnableViscous)
        {
            VecType tyy = *reinterpret_cast<VecType*>(stress_tensor->yy + thread_id * VectorSize);
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[0].vals[j] -= (xim_vgpr[0].vals[j] * txy.vals[j] +
                                  xim_vgpr[1].vals[j] * tyy.vals[j] +
                                  xim_vgpr[2].vals[j] * tyz.vals[j]);
            });
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[1].vals[j] -= (etm_vgpr[0].vals[j] * txy.vals[j] +
                                  etm_vgpr[1].vals[j] * tyy.vals[j] +
                                  etm_vgpr[2].vals[j] * tyz.vals[j]);
            });
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[2].vals[j] -= (zem_vgpr[0].vals[j] * txy.vals[j] +
                                  zem_vgpr[1].vals[j] * tyy.vals[j] +
                                  zem_vgpr[2].vals[j] * tyz.vals[j]);
            });
        }

        details::static_for<0, NumberOfSpatialDims, 1>{}([&](unsigned int j)
        {
            *(VecType*)(buffer + thread_id * VectorSize + j * size + 2 * size * NumberOfSpatialDims) = rr[j];
        });

        qa_vgpr = *reinterpret_cast<VecType*>(qa + thread_id * VectorSize + 3 * size);
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[0].vals[j] = qa_vgpr.vals[j] * ss[0].vals[j] + xim_vgpr[2].vals[j] * p_vgpr.vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[1].vals[j] = qa_vgpr.vals[j] * ss[1].vals[j] + etm_vgpr[2].vals[j] * p_vgpr.vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[2].vals[j] = qa_vgpr.vals[j] * ss[2].vals[j] + zem_vgpr[2].vals[j] * p_vgpr.vals[j];
        });

        if constexpr(EnableViscous)
        {
            VecType tzz = *reinterpret_cast<VecType*>(stress_tensor->zz + thread_id * VectorSize);
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[0].vals[j] -= (xim_vgpr[0].vals[j] * tzx.vals[j] +
                                  xim_vgpr[1].vals[j] * tyz.vals[j] +
                                  xim_vgpr[2].vals[j] * tzz.vals[j]);
            });
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[1].vals[j] -= (etm_vgpr[0].vals[j] * tzx.vals[j] +
                                  etm_vgpr[1].vals[j] * tyz.vals[j] +
                                  etm_vgpr[2].vals[j] * tzz.vals[j]);
            });
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[2].vals[j] -= (zem_vgpr[0].vals[j] * tzx.vals[j] +
                                  zem_vgpr[1].vals[j] * tyz.vals[j] +
                                  zem_vgpr[2].vals[j] * tzz.vals[j]);
            });
        }

        details::static_for<0, NumberOfSpatialDims, 1>{}([&](unsigned int j)
        {
            *(VecType*)(buffer +thread_id * VectorSize + j * size + 3 * size * NumberOfSpatialDims) = rr[j];
        });

        VecType de_vgpr4;
        qa_vgpr = *reinterpret_cast<VecType*>(qa + thread_id * VectorSize + 4 * size);
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            de_vgpr4.vals[j] = qa_vgpr.vals[j] + p_vgpr.vals[j];
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[0].vals[j] = de_vgpr4.vals[j] * ss[0].vals[j] -
                            p_vgpr.vals[j] *
                            (umf.x * xim_vgpr[0].vals[j] +
                             umf.y * xim_vgpr[1].vals[j] +
                             umf.z * xim_vgpr[2].vals[j]);
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[1].vals[j] = de_vgpr4.vals[j] * ss[1].vals[j] -
                            p_vgpr.vals[j] *
                            (umf.x * etm_vgpr[0].vals[j] +
                             umf.y * etm_vgpr[1].vals[j] +
                             umf.z * etm_vgpr[2].vals[j]);
        });
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rr[2].vals[j] = de_vgpr4.vals[j] * ss[2].vals[j] -
                            p_vgpr.vals[j] *
                            (umf.x * zem_vgpr[0].vals[j] +
                             umf.y * zem_vgpr[1].vals[j] +
                             umf.z * zem_vgpr[2].vals[j]);
        });
        if constexpr(EnableViscous)
        {
            VecType hxx = *reinterpret_cast<VecType*>(heat_fluxes->xx + thread_id * VectorSize);
            VecType hyy = *reinterpret_cast<VecType*>(heat_fluxes->yy + thread_id * VectorSize);
            VecType hzz = *reinterpret_cast<VecType*>(heat_fluxes->zz + thread_id * VectorSize);

            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[0].vals[j] -= (xim_vgpr[0].vals[j] * hxx.vals[j] +
                                  xim_vgpr[1].vals[j] * hyy.vals[j] +
                                  xim_vgpr[2].vals[j] * hzz.vals[j]);
            });
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[1].vals[j] -= (etm_vgpr[0].vals[j] * hxx.vals[j] +
                                  etm_vgpr[1].vals[j] * hyy.vals[j] +
                                  etm_vgpr[2].vals[j] * hzz.vals[j]);
            });
            details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
            {
                rr[2].vals[j] -= (zem_vgpr[0].vals[j] * hxx.vals[j] +
                                  zem_vgpr[1].vals[j] * hyy.vals[j] +
                                  zem_vgpr[2].vals[j] * hzz.vals[j]);
            });
        }

        details::static_for<0, NumberOfSpatialDims, 1>{}([&](unsigned int j)
        {
            *(VecType*)(buffer + thread_id * VectorSize + j * size + 4 * size * NumberOfSpatialDims) = rr[j];
        });

        *(VecType*)(de + thread_id * VectorSize + 4 * size) = de_vgpr4;
    }
}

template<unsigned int VariableId, typename Type>
CANARD_GLOBAL void calc_fluxes_post_compute_kernel(Type *de,
                                                   Type *buffer,
                                                   unsigned int size)
{
    unsigned int thread_id = get_thread_global_idx();
    if(thread_id < size)
    {
        unsigned int offset1 = VariableId * size;
        unsigned int offset2 = VariableId * size * NumberOfSpatialDims;
        de[thread_id + offset1] = buffer[thread_id + offset2 + 0 * size] +
                                  buffer[thread_id + offset2 + 1 * size] +
                                  buffer[thread_id + offset2 + 2 * size];
    }
}

template<typename Type>
CANARD_GLOBAL void calc_viscous_shear_stress_init_kernel(Type *buffer,
                                                         Type *de,
                                                         unsigned int size)
{
    unsigned int thread_id = get_thread_global_idx();
    if(thread_id < size)
    {
        details::static_for<1, NumberOfVariables, 1>{}([&](unsigned int m)
        {
            unsigned int offset2 = m * size * NumberOfSpatialDims;
            buffer[thread_id + 0 * size + offset2] = de[thread_id + m * size];
        });
    }
}


template<unsigned int VariableId, int VectorSize, typename Type>
CANARD_GLOBAL void
__launch_bounds__(256, 1)
calc_viscous_shear_stress_post_compute_kernel(t_stress_tensor<Type> * stress_tensor,
                                              t_heat_fluxes<Type> * heat_fluxes,
                                              Type *buffer_ss0,
                                              Type *buffer_ss1,
                                              Type *buffer_ss2,
                                              Type *buffer,
                                              Type *xim,
                                              Type *etm,
                                              Type *zem,
                                              unsigned int size)
{
    using VecType = Vector_Type<Type,VectorSize>;

    Type *out0, *out1, *out2;
    if constexpr(VariableId == 1)
    {
        out0 = stress_tensor->xx;
        out1 = heat_fluxes->zz;
        out2 = stress_tensor->zx;
    }
    else if constexpr(VariableId == 2)
    {
        out0 = stress_tensor->xy;
        out1 = stress_tensor->yy;
        out2 = heat_fluxes->xx;
    }
    else if constexpr(VariableId == 3)
    {
        out0 = heat_fluxes->yy;
        out1 = stress_tensor->yz;
        out2 = stress_tensor->zz;
    }
    else
    {
        out0 = buffer_ss0;
        out1 = buffer_ss1;
        out2 = buffer_ss2;
    }

    unsigned int thread_id = get_thread_global_idx();
    if(thread_id < size)
    {
        unsigned int offset2 = VariableId * size * NumberOfSpatialDims;
        VecType b0_vgpr, b1_vgpr, b2_vgpr;
        b0_vgpr =
            *reinterpret_cast<VecType*>(buffer + thread_id * VectorSize + offset2 + 0 * size);
        b1_vgpr =
            *reinterpret_cast<VecType*>(buffer + thread_id * VectorSize + offset2 + 1 * size);
        b2_vgpr =
            *reinterpret_cast<VecType*>(buffer + thread_id * VectorSize + offset2 + 2 * size);

        VecType tmp0, tmp1, tmp2;
        tmp0 = *reinterpret_cast<VecType*>(xim + thread_id * VectorSize + 0 * size);
        tmp1 = *reinterpret_cast<VecType*>(etm + thread_id * VectorSize + 0 * size);
        tmp2 = *reinterpret_cast<VecType*>(zem + thread_id * VectorSize + 0 * size);
        VecType rsl;
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rsl.vals[j] = tmp0.vals[j] * b0_vgpr.vals[j] + 
                          tmp1.vals[j] * b1_vgpr.vals[j] + 
                          tmp2.vals[j] * b2_vgpr.vals[j] ;
        });
        *(VecType*)(out0 + thread_id * VectorSize) = rsl;

        tmp0 = *reinterpret_cast<VecType*>(xim + thread_id * VectorSize + 1 * size);
        tmp1 = *reinterpret_cast<VecType*>(etm + thread_id * VectorSize + 1 * size);
        tmp2 = *reinterpret_cast<VecType*>(zem + thread_id * VectorSize + 1 * size);
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rsl.vals[j] = tmp0.vals[j] * b0_vgpr.vals[j] + 
                          tmp1.vals[j] * b1_vgpr.vals[j] + 
                          tmp2.vals[j] * b2_vgpr.vals[j] ;
        });
        *(VecType*)(out1 + thread_id * VectorSize) = rsl;

        tmp0 = *reinterpret_cast<VecType*>(xim + thread_id * VectorSize + 2 * size);
        tmp1 = *reinterpret_cast<VecType*>(etm + thread_id * VectorSize + 2 * size);
        tmp2 = *reinterpret_cast<VecType*>(zem + thread_id * VectorSize + 2 * size);
        details::static_for<0, VectorSize, 1>{}([&](unsigned int j)
        {
            rsl.vals[j] = tmp0.vals[j] * b0_vgpr.vals[j] + 
                          tmp1.vals[j] * b1_vgpr.vals[j] + 
                          tmp2.vals[j] * b2_vgpr.vals[j] ;
        });
        *(VecType*)(out2 + thread_id * VectorSize) = rsl;
    }
}

template<typename Type>
CANARD_GLOBAL void calc_viscous_shear_stress_final_kernel(Type *de,
                                                          Type *buffer,
                                                          Type *ssk,
                                                          Type *yaco,
                                                          Type *buffer_ss0,
                                                          Type *buffer_ss1,
                                                          Type *buffer_ss2,
                                                          t_stress_tensor<Type> * stress_tensor,
                                                          t_heat_fluxes<Type> * heat_fluxes,
                                                          unsigned int size)
{
    unsigned int thread_id = get_thread_global_idx();
    if(thread_id < size)
    {
        de[thread_id] = ssk[thread_id];
        buffer[thread_id] = de[thread_id] * yaco[thread_id];
        buffer[thread_id + size] = gamm1prndtli * buffer[thread_id];
        de[thread_id + 4 * size] = 2.0 / 3.0 * (stress_tensor->xx[thread_id] +
            stress_tensor->yy[thread_id] + stress_tensor->zz[thread_id]);

        stress_tensor->xx[thread_id] = buffer[thread_id] *
            (2.0 * stress_tensor->xx[thread_id] - de[thread_id + 4 * size]);
        stress_tensor->yy[thread_id] = buffer[thread_id] *
            (2.0 * stress_tensor->yy[thread_id] - de[thread_id + 4 * size]);
        stress_tensor->zz[thread_id] = buffer[thread_id] *
            (2.0 * stress_tensor->zz[thread_id] - de[thread_id + 4 * size]);

        stress_tensor->xy[thread_id] = buffer[thread_id] *
            (stress_tensor->xy[thread_id] + heat_fluxes->zz[thread_id]);
        stress_tensor->yz[thread_id] = buffer[thread_id] *
            (stress_tensor->yz[thread_id] + heat_fluxes->xx[thread_id]);
        stress_tensor->zx[thread_id] = buffer[thread_id] *
            (stress_tensor->zx[thread_id] + heat_fluxes->yy[thread_id]);

        heat_fluxes->xx[thread_id] = buffer[thread_id + size] * buffer_ss0[thread_id] +
            de[thread_id + size] * stress_tensor->xx[thread_id] +
            de[thread_id + 2 * size] * stress_tensor->xy[thread_id] +
            de[thread_id + 3 * size] * stress_tensor->zx[thread_id];
        heat_fluxes->yy[thread_id] = buffer[thread_id + size] * buffer_ss1[thread_id] +
            de[thread_id + size] * stress_tensor->xy[thread_id] +
            de[thread_id + 2 * size] * stress_tensor->yy[thread_id] +
            de[thread_id + 3 * size] * stress_tensor->yz[thread_id];
        heat_fluxes->zz[thread_id] = buffer[thread_id + size] * buffer_ss2[thread_id] +
            de[thread_id + size] * stress_tensor->zx[thread_id] +
            de[thread_id + 2 * size] * stress_tensor->yz[thread_id] +
            de[thread_id + 3 * size] * stress_tensor->zz[thread_id];
    }
}

template<unsigned int BlockSize, typename Type>
CANARD_GLOBAL void calc_time_step_kernel(Type * xim, Type * etm,
                                         Type * zem, Type * de, 
                                         Type * yaco, Type * ssk,
                                         t_point<Type> umf,
                                         Type *res, unsigned int size)
{
    CANARD_SHMEM Type sdata[BlockSize];

    Type * xim_x = xim;
    Type * xim_y = xim + size;
    Type * xim_z = xim + 2 * size;

    Type * etm_x = etm;
    Type * etm_y = etm + size;
    Type * etm_z = etm + 2 * size;

    Type * zem_x = zem;
    Type * zem_y = zem + size;
    Type * zem_z = zem + 2 * size;

    Type * de1 = de;
    Type * de2 = de + size;
    Type * de3 = de + 2 * size;
    Type * de4 = de + 3 * size;
    Type * de5 = de + 4 * size;

    unsigned int thread_id = get_thread_global_idx();
    unsigned int thread_local_id = get_thread_local_idx();

    Type result;
    sdata[thread_local_id] = 0;
    __syncthreads();

    // compute
    Type rr1, rr2, ssi;
    if(thread_id < size)
    {
        rr1 = xim_x[thread_id] * xim_x[thread_id] +
              xim_y[thread_id] * xim_y[thread_id] +
              xim_z[thread_id] * xim_z[thread_id] +
              etm_x[thread_id] * etm_x[thread_id] +
              etm_y[thread_id] * etm_y[thread_id] +
              etm_z[thread_id] * etm_z[thread_id] +
              zem_x[thread_id] * zem_x[thread_id] +
              zem_y[thread_id] * zem_y[thread_id] +
              zem_z[thread_id] * zem_z[thread_id];

        rr2 = abs(xim_x[thread_id] * (de2[thread_id] + umf.x)  +
                  xim_y[thread_id] * (de3[thread_id] + umf.y)  +
                  xim_z[thread_id] * (de4[thread_id] + umf.z)) +
              abs(etm_x[thread_id] * (de2[thread_id] + umf.x)  +
                  etm_y[thread_id] * (de3[thread_id] + umf.y)  +
                  etm_z[thread_id] * (de4[thread_id] + umf.z)) +
              abs(zem_x[thread_id] * (de2[thread_id] + umf.x)  +
                  zem_y[thread_id] * (de3[thread_id] + umf.y)  +
                  zem_z[thread_id] * (de4[thread_id] + umf.z));

        ssi = abs(yaco[thread_id]);

        result = (sqrt(de5[thread_id] * rr1) + rr2) * ssi;
    }

    // load into shmem
    sdata[thread_local_id] = result;
    __syncthreads();

    // do reduction in shared mem
    blockReduceShMemUnroll<BlockSize, Type>(sdata, thread_local_id);

    // write result for this block to global mem
    if (thread_local_id == 0)
        atomicMax(&res[0], sdata[0]);

    __syncthreads();

    // compute
    if(thread_id < size)
    {
        result = de1[thread_id] * ssk[thread_id] * rr1 * ssi * ssi;
    }

    // load into shmem
    sdata[thread_local_id] = result;
    __syncthreads();

    // do reduction in shared mem
    blockReduceShMemUnroll<BlockSize, Type>(sdata, thread_local_id);

    // write result for this block to global mem
    if (thread_local_id == 0)
        atomicMax(&res[1], sdata[0]);
}

#endif
