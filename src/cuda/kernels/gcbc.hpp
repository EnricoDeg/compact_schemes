/*
 * @file gcbc.hpp
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

#ifndef CANARD_KERNELS_GCBC_HPP
#define CANARD_KERNELS_GCBC_HPP

#include "common.hpp"
#include "common/data_types.hpp"
#include "common/parameters.hpp"
#include "cuda/kernels/common.hpp"

template<typename Type>
struct gcbc_vgpr
{
    CANARD_DEVICE gcbc_vgpr()
    {}

    CANARD_DEVICE void eleme(Type *cm,
                             Type qa1,
                             Type *qa24,
                             Type pp,
                             t_point<Type> umf)
    {
        Type rhoi  = 1.0 / qa1;
        ao    = sqrt( gam * rhoi * pp );
        aoi   = 1.0 / ao;
        ve[0] = rhoi * qa24[0];
        ve[1] = rhoi * qa24[1];
        ve[2] = rhoi * qa24[2];
        hv2   = 0.5 * ( ve[0] * ve[0] + ve[1] * ve[1] + ve[2] * ve[2] );
        vn    = cm[0] * ve[0] + cm[1] * ve[1] + cm[2] * ve[2];
        vs    = cm[0] * umf.x + cm[1] * umf.y + cm[2] * umf.z;
    }

    CANARD_DEVICE void xtr2q(Type *cm)
    {
        Type bo, co;

        bo    = hv2 + hamm1 * ao * ao;
        co    = ao * vn;

        dm[0] = ao * cm[0];
        dm[1] = ao * cm[1];
        dm[2] = ao * cm[2];

        xt[0][0] = cm[0];
        xt[0][1] = cm[1];
        xt[0][2] = cm[2];
        xt[0][3] = 0.5;
        xt[0][4] = 0.5;

        xt[1][0] = cm[0] * ve[0];
        xt[1][1] = cm[1] * ve[0] - dm[2];
        xt[1][2] = cm[2] * ve[0] + dm[1];
        xt[1][3] = 0.5 * (ve[0] + dm[0]);
        xt[1][4] = xt[1][3] - dm[0];

        xt[2][0] = cm[0] * ve[1] + dm[2];
        xt[2][1] = cm[1] * ve[1];
        xt[2][2] = cm[2] * ve[1] - dm[0];
        xt[2][3] = 0.5 * (ve[1] + dm[1]);
        xt[2][4] = xt[2][3] - dm[1];

        xt[3][0] = cm[0] * ve[2] - dm[1];
        xt[3][1] = cm[1] * ve[2] + dm[0];
        xt[3][2] = cm[2] * ve[2];
        xt[3][3] = 0.5 * (ve[2] + dm[2]);
        xt[3][4] = xt[3][3] - dm[2];

        xt[4][0] = hv2 * cm[0] + dm[2] * ve[1] - dm[1] * ve[2];
        xt[4][1] = hv2 * cm[1] + dm[0] * ve[2] - dm[2] * ve[0];
        xt[4][2] = hv2 * cm[2] + dm[1] * ve[0] - dm[0] * ve[1];
        xt[4][3] = 0.5 * (bo + co);
        xt[4][4] = xt[4][3] - co;
    }

    CANARD_DEVICE void xtq2r(Type *cm)
    {
        Type rv[3];
        Type ho, bo, co;

        ho    = gamm1 * aoi * aoi;
        bo    = 1.0 - ho * hv2;
        co    = aoi * vn;
        dm[0] = aoi * cm[0];
        dm[1] = aoi * cm[1];
        dm[2] = aoi * cm[1];

        rv[0] = ho * ve[0];
        rv[1] = ho * ve[1];
        rv[2] = ho * ve[2];

        xt[0][0] = bo * cm[0] + dm[1] * ve[2] - dm[2] * ve[1];
        xt[0][1] = cm[0] * rv[0];
        xt[0][2] = cm[0] * rv[1] + dm[2];
        xt[0][3] = cm[0] * rv[2] - dm[1];
        xt[0][4] = -ho * cm[1];

        xt[1][0] = bo * cm[1] + dm[2] * ve[0] - dm[0] * ve[2];
        xt[1][1] = cm[1] * rv[0] - dm[2];
        xt[1][2] = cm[1] * rv[1];
        xt[1][3] = cm[1] * rv[2] + dm[0];
        xt[1][4] = -ho * cm[1];

        xt[2][0] = bo * cm[2] + dm[0] * ve[1] - dm[1] * ve[0];
        xt[2][1] = cm[2] * rv[0] + dm[1];
        xt[2][2] = cm[2] * rv[1] - dm[0];
        xt[2][3] = cm[2] * rv[2];
        xt[2][4] = -ho * cm[2];

        xt[3][0] = 1.0 - bo - co;
        xt[3][1] = dm[0] - rv[0];
        xt[3][2] = dm[1] - rv[1];
        xt[3][3] = dm[2] - rv[2];
        xt[3][4] = ho;

        xt[4][0] = 1.0 - bo - co;
        xt[4][1] = -dm[0] - rv[0];
        xt[4][2] = -dm[1] - rv[1];
        xt[4][3] = -dm[2] - rv[2];
        xt[4][4] = ho;
    }

    Type ao;
    Type aoi;
    Type hv2;
    Type vn;
    Type vs;
    Type ve[NumberOfSpatialDims];
    Type dm[NumberOfSpatialDims];
    Type xt[NumberOfVariables][NumberOfVariables];
};

template<unsigned int Axis, typename Type>
CANARD_GLOBAL void wall_inviscid_kernel(Type * qa,
                                        int * npex,
                                        unsigned int face_offset,
                                        t_dcomp dcomp_info)
{

    Type * qa0 = qa;
    Type * qa1 = qa +     dcomp_info.lmx;
    Type * qa2 = qa + 2 * dcomp_info.lmx;
    Type * qa3 = qa + 3 * dcomp_info.lmx;
    Type * qa4 = qa + 4 * dcomp_info.lmx;

    int thread_stride;
    int block_stride;
    int stride;
    if(Axis == 0)
    {
        thread_stride = dcomp_info.lxi;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = 1;
    }
    else if(Axis == 1)
    {
        thread_stride = 1;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = dcomp_info.lxi;
    }
    else if(Axis == 2)
    {
        thread_stride = 1;
        block_stride = dcomp_info.lxi;
        stride = dcomp_info.let * dcomp_info.lxi;
    }

    int thread_idx = blockIdx.x * block_stride + threadIdx.x * thread_stride;
    int idx = thread_idx + face_offset * stride;

    Type velocity2 = qa1[idx] * qa1[idx] + qa2[idx] * qa2[idx] + qa3[idx] * qa3[idx];
    qa4[idx] = npex[idx] * qa4[idx] + ( 1.0 - npex[idx] ) * 
               ( hamhamm1 * pow(qa0[idx], gam) + 0.5 * velocity2 / qa0[idx] );
}

template<unsigned int Axis, typename Type>
CANARD_GLOBAL void wall_viscous_kernel(Type * qa,
                                       t_point<Type> umf,
                                       int * npex,
                                       unsigned int face_offset,
                                       t_dcomp dcomp_info)
{

    Type * qa0 = qa;
    Type * qa1 = qa +     dcomp_info.lmx;
    Type * qa2 = qa + 2 * dcomp_info.lmx;
    Type * qa3 = qa + 3 * dcomp_info.lmx;
    Type * qa4 = qa + 4 * dcomp_info.lmx;

    int thread_stride;
    int block_stride;
    int stride;
    if(Axis == 0)
    {
        thread_stride = dcomp_info.lxi;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = 1;
    }
    else if(Axis == 1)
    {
        thread_stride = 1;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = dcomp_info.lxi;
    }
    else if(Axis == 2)
    {
        thread_stride = 1;
        block_stride = dcomp_info.lxi;
        stride = dcomp_info.let * dcomp_info.lxi;
    }

    int thread_idx = blockIdx.x * block_stride + threadIdx.x * thread_stride;
    int idx = thread_idx + face_offset * stride;

    Type velocity2 = qa1[idx] * qa1[idx] + qa2[idx] * qa2[idx] + qa3[idx] * qa3[idx];
    Type fctr      = ( 1 - npex[idx] ) * qa0[idx];
    qa1[idx] = npex[idx] * qa1[idx] - fctr * umf.x;
    qa2[idx] = npex[idx] * qa2[idx] - fctr * umf.y;
    qa3[idx] = npex[idx] * qa3[idx] - fctr * umf.z;
    qa4[idx] = npex[idx] * qa4[idx] + ( 1 - npex[idx] ) *
               ( hamhamm1 * qa0[idx] + 0.5 * velocity2 / qa0[idx] );
}

template<unsigned int Axis, typename Type>
CANARD_GLOBAL void gcbc_setup_kernel(Type * cm,
                                     Type * drva,
                                     Type * qa,
                                     Type * de,
                                     Type * pressure,
                                     Type * yaco,
                                     t_point<Type> umf,
                                     unsigned int face_id,
                                     unsigned int face_offset,
                                     unsigned int flag,
                                     t_dcomp dcomp_info)
{
    Type * qa0 = qa;
    Type * qa1 = qa +     dcomp_info.lmx;
    Type * qa2 = qa + 2 * dcomp_info.lmx;
    Type * qa3 = qa + 3 * dcomp_info.lmx;
    Type * qa4 = qa + 4 * dcomp_info.lmx;

    int thread_stride;
    int block_stride;
    int stride;
    if(Axis == 0)
    {
        thread_stride = dcomp_info.lxi;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = 1;
    }
    else if(Axis == 1)
    {
        thread_stride = 1;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = dcomp_info.lxi;
    }
    else if(Axis == 2)
    {
        thread_stride = 1;
        block_stride = dcomp_info.lxi;
        stride = dcomp_info.let * dcomp_info.lxi;
    }
    unsigned int face_stride = get_face_stride<Axis>(dcomp_info);
    unsigned int face_idx = blockIdx.y * face_stride + blockIdx.x;
    unsigned int face_size = get_face_size<Axis>(dcomp_info);

    int thread_idx = blockIdx.x * block_stride + threadIdx.x * thread_stride;
    int idx = thread_idx + face_offset * stride;

    // load cm to vgpr
    Type cm_vgpr[NumberOfSpatialDims];
    for(unsigned int dim_id = 0; dim_id < NumberOfSpatialDims; ++dim_id)
    {
        unsigned int i = face_id * NumberOfSpatialDims * face_size +
            dim_id * face_size + face_idx;
        cm_vgpr[dim_id] = cm[i];
    }

    // load drva to vgpr
    Type drva_vgpr[NumberOfVariables];
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        unsigned int i = face_id * NumberOfVariables * face_size +
                variable_id * face_size + face_idx;
        drva_vgpr[variable_id] = drva[i];
    }

    Type vel_vgpr[NumberOfSpatialDims];
    vel_vgpr[0] = qa1[idx];
    vel_vgpr[1] = qa2[idx];
    vel_vgpr[2] = qa3[idx];

    auto gcbc_params = gcbc_vgpr<Type>();
    gcbc_params.eleme(&cm_vgpr[0], qa0[idx], &vel_vgpr[0],
                      pressure[idx], umf);

    gcbc_params.xtq2r(&cm_vgpr[0]);

    Type cha_vgpr[NumberOfVariables];
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        cha_vgpr[variable_id] = yaco[idx] * (flag * drva_vgpr[variable_id] + 
                                             (1 - flag) * de[idx + variable_id * dcomp_info.lmx]);
    }

    MatVecMul<NumberOfVariables>(&gcbc_params.xt[0][0], &cha_vgpr[0], &drva_vgpr[0]);

    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        unsigned int i = face_id * NumberOfVariables * face_size +
                variable_id * face_size + face_idx;
        drva[i] = drva_vgpr[variable_id];
    }
}

template<unsigned int Axis, typename Type>
CANARD_GLOBAL void gcbc_update_non_reflective_kernel(Type * cm,
                                                     Type * drva,
                                                     Type * qa,
                                                     Type * de,
                                                     Type * pressure,
                                                     Type * sbcc,
                                                     t_point<Type> umf,
                                                     unsigned int face_id,
                                                     unsigned int face_offset,
                                                     t_dcomp dcomp_info)
{
    Type * qa0 = qa;
    Type * qa1 = qa +     dcomp_info.lmx;
    Type * qa2 = qa + 2 * dcomp_info.lmx;
    Type * qa3 = qa + 3 * dcomp_info.lmx;
    Type * qa4 = qa + 4 * dcomp_info.lmx;

    int thread_stride;
    int block_stride;
    int stride;
    if(Axis == 0)
    {
        thread_stride = dcomp_info.lxi;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = 1;
    }
    else if(Axis == 1)
    {
        thread_stride = 1;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = dcomp_info.lxi;
    }
    else if(Axis == 2)
    {
        thread_stride = 1;
        block_stride = dcomp_info.lxi;
        stride = dcomp_info.let * dcomp_info.lxi;
    }
    unsigned int face_stride = get_face_stride<Axis>(dcomp_info);
    unsigned int face_idx = blockIdx.y * face_stride + blockIdx.x;
    unsigned int face_size = get_face_size<Axis>(dcomp_info);

    int thread_idx = blockIdx.x * block_stride + threadIdx.x * thread_stride;
    int idx = thread_idx + face_offset * stride;

    // load cm to vgpr
    Type cm_vgpr[NumberOfSpatialDims];
    for(unsigned int dim_id = 0; dim_id < NumberOfSpatialDims; ++dim_id)
    {
        unsigned int i = face_id * NumberOfSpatialDims * face_size +
            dim_id * face_size + face_idx;
        cm_vgpr[dim_id] = cm[i];
    }

    // load drva to vgpr
    Type drva_vgpr[NumberOfVariables];
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        unsigned int i = face_id * NumberOfVariables * face_size +
                variable_id * face_size + face_idx;
        drva_vgpr[variable_id] = drva[i];
    }

    Type vel_vgpr[NumberOfSpatialDims];
    vel_vgpr[0] = qa1[idx];
    vel_vgpr[1] = qa2[idx];
    vel_vgpr[2] = qa3[idx];

    auto gcbc_params = gcbc_vgpr<Type>();
    gcbc_params.eleme(&cm_vgpr[0], qa0[idx], &vel_vgpr[0],
                      pressure[idx], umf);

    Type ra0 = 1 - 2 * face_id;

    // compute cha in vgpr
    Type cha_vgpr[NumberOfVariables];
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        cha_vgpr[variable_id] = drva_vgpr[variable_id];
    }
    if ( ra0 * ( gcbc_params.vn + gcbc_params.vs + gcbc_params.ao ) > 0 )
    {
        cha_vgpr[3] = 0;
    }
    if ( ra0 * ( gcbc_params.vn + gcbc_params.vs - gcbc_params.ao ) > 0 )
    {
        cha_vgpr[4] = 0;
    }
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        cha_vgpr[variable_id] -= drva_vgpr[variable_id];
    }

    gcbc_params.xtr2q(&cm_vgpr[0]);

    MatVecMul<NumberOfVariables>(&gcbc_params.xt[0][0], &cha_vgpr[0], &drva_vgpr[0]);

    // update de
    unsigned int out_idx;
    int iq = 1 - 2 * face_id;
    for(unsigned int ii = 0; ii < mbci; ++ii)
    {
        out_idx = idx + ii * iq * stride;
        for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
        {
            de[out_idx] += sbcc[out_idx] * drva_vgpr[variable_id];
            out_idx += variable_id * dcomp_info.lmx;
        }
    }
}

template<unsigned int Axis, typename Type>
CANARD_GLOBAL void gcbc_update_wall_kernel(Type * cm,
                                           Type * drva,
                                           Type * qa,
                                           Type * de,
                                           Type * pressure,
                                           Type * sbcc,
                                           t_point<Type> umf,
                                           t_point<Type> dudtmf,
                                           unsigned int face_id,
                                           unsigned int face_offset,
                                           Type dtwi,
                                           t_dcomp dcomp_info)
{
    Type * qa0 = qa;
    Type * qa1 = qa +     dcomp_info.lmx;
    Type * qa2 = qa + 2 * dcomp_info.lmx;
    Type * qa3 = qa + 3 * dcomp_info.lmx;
    Type * qa4 = qa + 4 * dcomp_info.lmx;

    int thread_stride;
    int block_stride;
    int stride;
    if(Axis == 0)
    {
        thread_stride = dcomp_info.lxi;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = 1;
    }
    else if(Axis == 1)
    {
        thread_stride = 1;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = dcomp_info.lxi;
    }
    else if(Axis == 2)
    {
        thread_stride = 1;
        block_stride = dcomp_info.lxi;
        stride = dcomp_info.let * dcomp_info.lxi;
    }
    unsigned int face_stride = get_face_stride<Axis>(dcomp_info);
    unsigned int face_idx = blockIdx.y * face_stride + blockIdx.x;
    unsigned int face_size = get_face_size<Axis>(dcomp_info);

    int thread_idx = blockIdx.x * block_stride + threadIdx.x * thread_stride;
    int idx = thread_idx + face_offset * stride;

    // load cm to vgpr
    Type cm_vgpr[NumberOfSpatialDims];
    for(unsigned int dim_id = 0; dim_id < NumberOfSpatialDims; ++dim_id)
    {
        unsigned int i = face_id * NumberOfSpatialDims * face_size +
            dim_id * face_size + face_idx;
        cm_vgpr[dim_id] = cm[i];
    }

    // load drva to vgpr
    Type drva_vgpr[NumberOfVariables];
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        unsigned int i = face_id * NumberOfVariables * face_size +
                variable_id * face_size + face_idx;
        drva_vgpr[variable_id] = drva[i];
    }

    Type vel_vgpr[NumberOfSpatialDims];
    vel_vgpr[0] = qa1[idx];
    vel_vgpr[1] = qa2[idx];
    vel_vgpr[2] = qa3[idx];

    auto gcbc_params = gcbc_vgpr<Type>();
    gcbc_params.eleme(&cm_vgpr[0], qa0[idx], &vel_vgpr[0],
                      pressure[idx], umf);

    Type ra0 = 1 - 2 * face_id;

    // compute cha in vgpr
    Type cha_vgpr[NumberOfVariables];
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        cha_vgpr[variable_id] = drva_vgpr[variable_id];
    }
    Type accum = cm_vgpr[0] * dudtmf.x +
                 cm_vgpr[1] * dudtmf.y +
                 cm_vgpr[2] * dudtmf.z;
    cha_vgpr[3+face_id] = cha_vgpr[4-face_id] + 2 * ra0 * gcbc_params.aoi * qa0[idx] *
                    ( accum + dtwi * ( gcbc_params.vn + gcbc_params.vs ) );

    gcbc_params.xtr2q(&cm_vgpr[0]);
    MatVecMul<NumberOfVariables>(&gcbc_params.xt[0][0], &cha_vgpr[0], &drva_vgpr[0]);

    // update de
    unsigned int out_idx;
    int iq = 1 - 2 * face_id;
    for(unsigned int ii = 0; ii < mbci; ++ii)
    {
        out_idx = idx + ii * iq * stride;
        for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
        {
            de[out_idx] += sbcc[out_idx] * drva_vgpr[variable_id];
            out_idx += variable_id * dcomp_info.lmx;
        }
    }
}

template<unsigned int Axis, typename Type>
CANARD_GLOBAL void gcbc_update_inter_curv_kernel(Type * cm,
                                                 Type * drva,
                                                 Type * drvb,
                                                 Type * qa,
                                                 Type * de,
                                                 Type * pressure,
                                                 Type * sbcc,
                                                 t_point<Type> umf,
                                                 t_point<Type> dudtmf,
                                                 unsigned int face_id,
                                                 unsigned int face_offset,
                                                 t_dcomp dcomp_info)
{
    Type * qa0 = qa;
    Type * qa1 = qa +     dcomp_info.lmx;
    Type * qa2 = qa + 2 * dcomp_info.lmx;
    Type * qa3 = qa + 3 * dcomp_info.lmx;
    Type * qa4 = qa + 4 * dcomp_info.lmx;

    int thread_stride;
    int block_stride;
    int stride;
    if(Axis == 0)
    {
        thread_stride = dcomp_info.lxi;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = 1;
    }
    else if(Axis == 1)
    {
        thread_stride = 1;
        block_stride = dcomp_info.let * dcomp_info.lxi;
        stride = dcomp_info.lxi;
    }
    else if(Axis == 2)
    {
        thread_stride = 1;
        block_stride = dcomp_info.lxi;
        stride = dcomp_info.let * dcomp_info.lxi;
    }
    unsigned int face_stride = get_face_stride<Axis>(dcomp_info);
    unsigned int face_idx = blockIdx.y * face_stride + blockIdx.x;
    unsigned int face_size = get_face_size<Axis>(dcomp_info);

    int thread_idx = blockIdx.x * block_stride + threadIdx.x * thread_stride;
    int idx = thread_idx + face_offset * stride;

    // load cm to vgpr
    Type cm_vgpr[NumberOfSpatialDims];
    for(unsigned int dim_id = 0; dim_id < NumberOfSpatialDims; ++dim_id)
    {
        unsigned int i = face_id * NumberOfSpatialDims * face_size +
            dim_id * face_size + face_idx;
        cm_vgpr[dim_id] = cm[i];
    }

    // load drva to vgpr
    Type drva_vgpr[NumberOfVariables];
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        unsigned int i = face_id * NumberOfVariables * face_size +
                variable_id * face_size + face_idx;
        drva_vgpr[variable_id] = drva[i];
    }

    // load drvb to vgpr
    Type drvb_vgpr[NumberOfVariables];
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        unsigned int i = face_id * NumberOfVariables * face_size +
                variable_id * face_size + face_idx;
        drvb_vgpr[variable_id] = drvb[i];
    }

    Type vel_vgpr[NumberOfSpatialDims];
    vel_vgpr[0] = qa1[idx];
    vel_vgpr[1] = qa2[idx];
    vel_vgpr[2] = qa3[idx];

    auto gcbc_params = gcbc_vgpr<Type>();
    gcbc_params.eleme(&cm_vgpr[0], qa0[idx], &vel_vgpr[0],
                      pressure[idx], umf);

    // compute cha in vgpr
    Type cha_vgpr[NumberOfVariables];
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        cha_vgpr[variable_id] = drva_vgpr[variable_id];
    }

    Type ra0 = 1 - 2 * face_id;

    Type dha_vgpr[NumberOfVariables];
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        dha_vgpr[variable_id] = drvb_vgpr[variable_id];
    }
    if ( ra0 * ( gcbc_params.vn + gcbc_params.vs ) > 0 )
    {
        cha_vgpr[0] = dha_vgpr[0];
        cha_vgpr[1] = dha_vgpr[1];
        cha_vgpr[2] = dha_vgpr[2];
    }
    if ( ra0 * ( gcbc_params.vn + gcbc_params.vs + gcbc_params.ao ) > 0 )
    {
        cha_vgpr[3] = dha_vgpr[3];
    }
    if ( ra0 * ( gcbc_params.vn + gcbc_params.vs - gcbc_params.ao ) > 0 )
    {
        cha_vgpr[4] = dha_vgpr[4];
    }
    for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
    {
        cha_vgpr[variable_id] -= drva_vgpr[variable_id];
    }

    gcbc_params.xtr2q(&cm_vgpr[0]);

    MatVecMul<NumberOfVariables>(&gcbc_params.xt[0][0], &cha_vgpr[0], &dha_vgpr[0]);

    // update de
    unsigned int out_idx;
    int iq = 1 - 2 * face_id;
    for(unsigned int ii = 0; ii < mbci; ++ii)
    {
        out_idx = idx + ii * iq * stride;
        for(unsigned int variable_id = 0; variable_id < NumberOfVariables; ++variable_id)
        {
            de[out_idx] += sbcc[out_idx] * dha_vgpr[variable_id];
            out_idx += variable_id * dcomp_info.lmx;
        }
    }
}

#endif
