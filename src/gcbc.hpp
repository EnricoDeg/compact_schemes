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

#ifndef CANARD_GCBC_HPP
#define CANARD_GCBC_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "host/functional.hpp"
#include "host/transforms.hpp"

#include "cuda/gcbc.hpp"
#include "cuda/common.hpp"

#include "mpi/exchange.hpp"

template<typename Type, typename TypeIndex>
struct gcbc
{
    gcbc(t_dcomp dcomp_info)
    {
        size_t nelements = dcomp_info.lmx;

        npex  = allocate_cuda<TypeIndex>(nelements);

        drvb0 = allocate_cuda<Type>(2 * NumberOfVariables * dcomp_info.let * dcomp_info.lze);
        drvb1 = allocate_cuda<Type>(2 * NumberOfVariables * dcomp_info.lxi * dcomp_info.lze);
        drvb2 = allocate_cuda<Type>(2 * NumberOfVariables * dcomp_info.lxi * dcomp_info.let);
        drvb[0] = drvb0;
        drvb[1] = drvb1;
        drvb[2] = drvb2;

        sbcc = allocate_cuda<Type>(nelements);
    }

    // initialize npex
    void init()
    {

    }

    ~gcbc()
    {
        free_cuda(npex);

        free_cuda(drvb0);
        free_cuda(drvb1);
        free_cuda(drvb2);

        free_cuda(sbcc);
    }

    TypeIndex *npex;
    Type *drvb0, *drvb1, *drvb2;
    Type *drvb[3];
    Type *sbcc;
};

template<typename Type>
void gcbc_go(Type *drva_buffer[3], Type *cm_buffer[3], Type *drvb_buffer[3],
             Type *qa, Type *de, Type *pressure, Type *yaco, Type * sbcc,
             t_point<Type> umf, t_point<Type> dudtmf, t_dcomp dcomp_info, Type dtwi,
             int nbc[2][3], int mcd[2][3])
{
    // Preparation for GCBC & GCIC
    unsigned int dim;
    Type *drva, *cm, *drvb;

    auto gcbc_instance = gcbc_dispatch<Type>(cm_buffer[0],
        drva_buffer[0],
        drvb_buffer[0],
        qa,
        de,
        pressure,
        yaco,
        sbcc,
        umf,
        dudtmf,
        dcomp_info);

    host::static_for<0, NumberOfSpatialDims, 1>{}([&](auto nn)
    {
        dim = host::get_dimension<nn>(dcomp_info);
        drva = drva_buffer[nn];
        cm   = cm_buffer[nn];
        gcbc_instance.reset_buffer_pointer(drva, cm);

        for(unsigned int ip = 0; ip < 2; ++ip)
        {
            const unsigned int np = nbc[ip][nn];
            if( ( np - BC_NON_REFLECTIVE ) *
                ( np - BC_WALL_INVISCID  ) *
                ( np - BC_WALL_VISCOUS   ) *
                ( np - BC_INTER_CURV     ) == 0 )
            {
                unsigned int flag = ( BC_WALL_INVISCID - np ) *
                                    ( BC_WALL_VISCOUS  - np ) *
                                    ( BC_INTER_CURV    - np ) / 3000;
                unsigned int face_offset = ip * dim;
                gcbc_instance.template setup<nn>(ip, face_offset, flag);
            }
        }
    });

    // Internode communication for GCIC
    auto exchange_instance = exchange<Type>();
    int itag = 30;
    host::static_for<0, NumberOfSpatialDims, 1>{}([&](auto nn)
    {
        drva = drva_buffer[nn];
        drvb = drvb_buffer[nn];
        exchange_instance.reset_buffer_pointer(drva, drvb);
        size_t mpi_size = host::get_face_size<nn, NumberOfVariables>(dcomp_info);
        host::static_for<0, NumberOfFaces, 1>{}([&](auto ip)
        {
            static constexpr auto iq = 1 - ip;
            const int pointer_offset = ip * mpi_size;
            const int np = nbc[ip][nn];
            if(( np - BC_INTER_CURV ) *
               ( 1 + std::abs((np-BC_WALL_INVISCID)*(np-BC_WALL_VISCOUS)) ) == 0)
            {
                exchange_instance.trigger(mpi_size,
                    pointer_offset,
                    mcd[ip][nn],
                    itag + iq,
                    itag + ip);
            }
        });
    });
    exchange_instance.reset();

    // Implementation of GCBC & GCIC
    host::static_for<0, NumberOfSpatialDims, 1>{}([&](auto nn)
    {
        dim = host::get_dimension<nn>(dcomp_info);
        drva = drva_buffer[nn];
        drvb = drvb_buffer[nn];
        cm   = cm_buffer[nn];
        gcbc_instance.reset_buffer_pointer(drva, drvb, cm);

        for(unsigned int ip = 0; ip < 2; ++ip)
        {
            const unsigned int np = nbc[ip][nn];
            unsigned int face_offset = ip * dim;
            if(np == BC_NON_REFLECTIVE)
            {
                gcbc_instance.template update_non_reflective<nn>(ip, face_offset);
            }
            else if(np == BC_WALL_INVISCID || np == BC_WALL_VISCOUS)
            {
                gcbc_instance.template update_wall<nn>(ip, face_offset, dtwi);
            }
            else if(np == BC_INTER_CURV)
            {
                gcbc_instance.template update_inter_curv<nn>(ip, face_offset);
            }
        }
    });
}

template<typename Type, typename TypeIndex>
void wall_condition_update(Type *qa,
                           TypeIndex *npex,
                           t_point<Type> umf,
                           t_dcomp dcomp_info,
                           int nbc[2][3])
{

    auto wall_bc_instance = wall_bc_dispatch<Type, TypeIndex>(qa, npex, umf, dcomp_info);
    unsigned int dim;
    host::static_for<0, 3, 1>{}([&](auto nn)
    {
        if constexpr(nn == 0)
        {
            dim = dcomp_info.lxi;
        }
        else if constexpr(nn == 1)
        {
            dim = dcomp_info.let;
        }
        else if constexpr(nn == 2)
        {
            dim = dcomp_info.lze;
        }

        for(unsigned int ip = 0; ip < 2; ++ip)
        {

            const unsigned int np = nbc[ip][nn];
            unsigned int face_offset = ip * dim;
            if(np == BC_WALL_INVISCID)
            {
                wall_bc_instance.template apply_inviscid<nn>(face_offset);
            }
            else if(np == BC_WALL_VISCOUS)
            {
                wall_bc_instance.template apply_viscous<nn>(face_offset);
            }
        }
    });
}

#endif
