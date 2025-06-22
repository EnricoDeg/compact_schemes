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

template<bool EnableViscous, typename Type>
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
    unsigned int thread_id = get_thread_global_idx();

    if(thread_id < size)
    {
        // Gmem -> VGPR
        Type de_vgpr[NumberOfVariables];
        de_vgpr[0] = de[thread_id];
        de_vgpr[1] = de[thread_id + size];
        de_vgpr[2] = de[thread_id + 2 * size];
        de_vgpr[3] = de[thread_id + 3 * size];
        de_vgpr[4] = de[thread_id + 4 * size];

        Type xim_vgpr[NumberOfSpatialDims];
        xim_vgpr[0] = xim[thread_id];
        xim_vgpr[1] = xim[thread_id + size];
        xim_vgpr[2] = xim[thread_id + 2 * size];

        Type etm_vgpr[NumberOfSpatialDims];
        etm_vgpr[0] = etm[thread_id];
        etm_vgpr[1] = etm[thread_id + size];
        etm_vgpr[2] = etm[thread_id + 2 * size];

        Type zem_vgpr[NumberOfSpatialDims];
        zem_vgpr[0] = zem[thread_id];
        zem_vgpr[1] = zem[thread_id + size];
        zem_vgpr[2] = zem[thread_id + 2 * size];

        Type qa_vgpr[NumberOfVariables];
        qa_vgpr[0] = qa[thread_id];
        qa_vgpr[1] = qa[thread_id + size];
        qa_vgpr[2] = qa[thread_id + 2 * size];
        qa_vgpr[3] = qa[thread_id + 3 * size];
        qa_vgpr[4] = qa[thread_id + 4 * size];

        Type txx = stress_tensor->xx[thread_id];
        Type tyy = stress_tensor->yy[thread_id];
        Type tzz = stress_tensor->zz[thread_id];
        Type txy = stress_tensor->xy[thread_id];
        Type tyz = stress_tensor->yz[thread_id];
        Type tzx = stress_tensor->zx[thread_id];

        Type hxx = heat_fluxes->xx[thread_id];
        Type hyy = heat_fluxes->yy[thread_id];
        Type hzz = heat_fluxes->zz[thread_id];

        Type p_vgpr = pressure[thread_id];

        Type rr[NumberOfSpatialDims];
        Type ss[NumberOfSpatialDims];

        // Compute
        rr[0] = de_vgpr[1] + umf.x;
        rr[1] = de_vgpr[2] + umf.y;
        rr[2] = de_vgpr[3] + umf.z;

        ss[0] = xim_vgpr[0] * rr[0] + xim_vgpr[1] * rr[1] + xim_vgpr[2] * rr[2];
        ss[1] = etm_vgpr[0] * rr[0] + etm_vgpr[1] * rr[1] + etm_vgpr[2] * rr[2];
        ss[2] = zem_vgpr[0] * rr[0] + zem_vgpr[1] * rr[1] + zem_vgpr[2] * rr[2];

        rr[0] = qa_vgpr[0] * ss[0];
        rr[1] = qa_vgpr[0] * ss[1];
        rr[2] = qa_vgpr[0] * ss[2];

        details::static_for<0, NumberOfSpatialDims, 1>{}([&](unsigned int j)
        {
            buffer[thread_id + j * size + 0 * size * NumberOfSpatialDims] = rr[j];
        });

        rr[0] = qa_vgpr[1] * ss[0] + xim_vgpr[0] * p_vgpr;
        rr[1] = qa_vgpr[1] * ss[1] + etm_vgpr[0] * p_vgpr;
        rr[2] = qa_vgpr[1] * ss[2] + zem_vgpr[0] * p_vgpr;

        if constexpr(EnableViscous)
        {
            rr[0] -= (xim_vgpr[0] * txx + xim_vgpr[1] * txy + xim_vgpr[2] * tzx);
            rr[1] -= (etm_vgpr[0] * txx + etm_vgpr[1] * txy + etm_vgpr[2] * tzx);
            rr[2] -= (zem_vgpr[0] * txx + zem_vgpr[1] * txy + zem_vgpr[2] * tzx);
        }

        details::static_for<0, NumberOfSpatialDims, 1>{}([&](unsigned int j)
        {
            buffer[thread_id + j * size + 1 * size * NumberOfSpatialDims] = rr[j];
        });

        rr[0] = qa_vgpr[2] * ss[0] + xim_vgpr[1] * p_vgpr;
        rr[1] = qa_vgpr[2] * ss[1] + etm_vgpr[1] * p_vgpr;
        rr[2] = qa_vgpr[2] * ss[2] + zem_vgpr[1] * p_vgpr;

        if constexpr(EnableViscous)
        {
            rr[0] -= (xim_vgpr[0] * txy + xim_vgpr[1] * tyy + xim_vgpr[2] * tyz);
            rr[1] -= (etm_vgpr[0] * txy + etm_vgpr[1] * tyy + etm_vgpr[2] * tyz);
            rr[2] -= (zem_vgpr[0] * txy + zem_vgpr[1] * tyy + zem_vgpr[2] * tyz);
        }

        details::static_for<0, NumberOfSpatialDims, 1>{}([&](unsigned int j)
        {
            buffer[thread_id + j * size + 2 * size * NumberOfSpatialDims] = rr[j];
        });

        rr[0] = qa_vgpr[3] * ss[0] + xim_vgpr[2] * p_vgpr;
        rr[1] = qa_vgpr[3] * ss[1] + etm_vgpr[2] * p_vgpr;
        rr[2] = qa_vgpr[3] * ss[2] + zem_vgpr[2] * p_vgpr;

        if constexpr(EnableViscous)
        {
            rr[0] -= (xim_vgpr[0] * tzx + xim_vgpr[1] * tyz + xim_vgpr[2] * tzz);
            rr[1] -= (etm_vgpr[0] * tzx + etm_vgpr[1] * tyz + etm_vgpr[2] * tzz);
            rr[2] -= (zem_vgpr[0] * tzx + zem_vgpr[1] * tyz + zem_vgpr[2] * tzz);
        }

        details::static_for<0, NumberOfSpatialDims, 1>{}([&](unsigned int j)
        {
            buffer[thread_id + j * size + 3 * size * NumberOfSpatialDims] = rr[j];
        });

        de_vgpr[4] = qa_vgpr[4] + p_vgpr;
        rr[0] = de_vgpr[4] * ss[0] -
            p_vgpr * (umf.x * xim_vgpr[0] + umf.y * xim_vgpr[1] + umf.z * xim_vgpr[2]);
        rr[1] = de_vgpr[4] * ss[1] -
            p_vgpr * (umf.x * etm_vgpr[0] + umf.y * etm_vgpr[1] + umf.z * etm_vgpr[2]);
        rr[2] = de_vgpr[4] * ss[2] -
            p_vgpr * (umf.x * zem_vgpr[0] + umf.y * zem_vgpr[1] + umf.z * zem_vgpr[2]);

        if constexpr(EnableViscous)
        {
            rr[0] -= (xim_vgpr[0] * hxx + xim_vgpr[1] * hyy + xim_vgpr[2] * hzz);
            rr[1] -= (etm_vgpr[0] * hxx + etm_vgpr[1] * hyy + etm_vgpr[2] * hzz);
            rr[2] -= (zem_vgpr[0] * hxx + zem_vgpr[1] * hyy + zem_vgpr[2] * hzz);
        }

        details::static_for<0, NumberOfSpatialDims, 1>{}([&](unsigned int j)
        {
            buffer[thread_id + j * size + 4 * size * NumberOfSpatialDims] = rr[j];
        });

        de[thread_id + 4 * size] = de_vgpr[4];
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


template<unsigned int VariableId, typename Type>
CANARD_GLOBAL void calc_viscous_shear_stress_post_compute_kernel(t_stress_tensor<Type> * stress_tensor,
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

        out0[thread_id] = xim[thread_id + 0 * size] * buffer[thread_id + offset2 + 0 * size] + 
                          etm[thread_id + 0 * size] * buffer[thread_id + offset2 + 1 * size] +
                          zem[thread_id + 0 * size] * buffer[thread_id + offset2 + 2 * size] ;

        out1[thread_id] = xim[thread_id + 1 * size] * buffer[thread_id + offset2 + 0 * size] + 
                          etm[thread_id + 1 * size] * buffer[thread_id + offset2 + 1 * size] +
                          zem[thread_id + 1 * size] * buffer[thread_id + offset2 + 2 * size] ;

        out2[thread_id] = xim[thread_id + 2 * size] * buffer[thread_id + offset2 + 0 * size] + 
                          etm[thread_id + 2 * size] * buffer[thread_id + offset2 + 1 * size] +
                          zem[thread_id + 2 * size] * buffer[thread_id + offset2 + 2 * size] ;
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
