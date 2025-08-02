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
#include "common/utils.hpp"

#include "host/functional.hpp"
#include "host/transforms.hpp"

#include "cuda/gcbc.hpp"
#include "cuda/common.hpp"

#include "mpi/exchange.hpp"

#include "domdcomp.hpp"

template<typename Type, typename TypeIndex>
struct gcbc
{
    gcbc(const domdcomp& domdcomp_instance,
         Type *yaco)
    {
        size_t nelements = domdcomp_instance.lmx + 1;

        npex = allocate_cuda<TypeIndex>(nelements);
        nrr  = allocate_cuda<TypeIndex>(nelements);

        Type *drvb0, *drvb1, *drvb2;
        unsigned int offset;
        offset = (domdcomp_instance.let + 1) * (domdcomp_instance.lze + 1);
        drvb0  = allocate_cuda<Type>(2 * NumberOfVariables * offset);
        offset = (domdcomp_instance.lxi + 1) * (domdcomp_instance.lze + 1);
        drvb1  = allocate_cuda<Type>(2 * NumberOfVariables * offset);
        offset = (domdcomp_instance.lxi + 1) * (domdcomp_instance.let + 1);
        drvb2  = allocate_cuda<Type>(2 * NumberOfVariables * offset);
        drvb_buffer[0] = drvb0;
        drvb_buffer[1] = drvb1;
        drvb_buffer[2] = drvb2;

        Type *yaco_h = (Type *)malloc((domdcomp_instance.lmx+1) * sizeof(Type));
        memcpy_cuda_d2h(yaco_h, yaco, domdcomp_instance.lmx + 1);

        Type cbca[mbci][mbci];
        Type cbcs[mbci][mbci];
        Type rbci[mbci];
        Type sbci[mbci];

        for(unsigned int i = 0; i < mbci; ++i)
        {
            for(unsigned int j = 0; j < mbci; ++j)
            {
                cbca[i][j] = 0.0;
            }
        }
        cbca[0][0] = alpha01;
        cbca[1][0] = 1.0;
        cbca[1][1] = alpha10;
        cbca[2][0] = alpha;
        cbca[2][1] = 1.0;
        cbca[2][2] = alpha;

        for(unsigned int i = 0; i < mbci; ++i)
        {
            rbci[i] = 0.0;
        }
        rbci[0] = 1.0;
        rbci[1] = alpha10;
        mtrxi(&cbca[0][0], &cbcs[0][0], mbci*mbci);
        matvecmul(&cbcs[0][0], &rbci[0], &sbci[0], mbci);
        for(unsigned int i = 0; i < mbci; ++i)
        {
            sbci[i] *= -1.0;
        }
        Type fctr = pi / ( mbci + 1 );
        Type res = 0.0;
        for(unsigned int i = 0; i < mbci; ++i)
        {
            res = res + 1.0;
            sbci[i] = 0.5 * sbci[i] * (1.0 + std::cos(res * fctr));
        }

        int ll = -1;
        int *npex_h = (int *)malloc((domdcomp_instance.lmx + 1) * sizeof(int));
        int *nrr_h = (int *)malloc((domdcomp_instance.lmx + 1) * sizeof(int));
        Type *rr_h = (Type *)malloc(NumberOfSpatialDims *
            (domdcomp_instance.lmx + 1) * sizeof(Type));
        for(unsigned int i = 0; i < domdcomp_instance.lmx + 1; ++i)
        {
            npex_h[i] = 0;
            nrr_h[i] = 0;
            rr_h[i] = 0.0;
        }

        for(unsigned int nn = 0; nn < NumberOfSpatialDims; ++nn)
        {
            for(unsigned int ip = 0; ip < NumberOfFaces; ++ip)
            {
                int np = domdcomp_instance.nbc[ip][nn];
                int i = ip * domdcomp_instance.ijk[0][nn];
                int iq = 1 - 2 * ip;

                if ( ( np - BC_NON_REFLECTIVE ) *
                     ( np - BC_WALL_INVISCID  ) *
                     ( np - BC_WALL_VISCOUS   ) *
                     ( np - BC_INTER_CURV     ) == 0 )
                {
                    for(unsigned int k = 0; k <= domdcomp_instance.ijk[2][nn]; ++k)
                    {
                        for(unsigned int j = 0; j <= domdcomp_instance.ijk[1][nn]; ++j)
                        {
                            int l = host::indx3(i, j, k, nn,
                                domdcomp_instance.lxi, domdcomp_instance.let);
                            ll++;
                            res = 1.0 / yaco_h[l];
                            rr_h[l] += 1.0;
                            rr_h[ll + 1 * (domdcomp_instance.lmx + 1)] = res;
                            rr_h[ll + 2 * (domdcomp_instance.lmx + 1)] = l + sml;
                            for(unsigned int ii = 1; ii <= mbci; ++ii)
                            {
                                l = host::indx3(i+iq*ii, j, k, nn,
                                    domdcomp_instance.lxi, domdcomp_instance.let);
                                ll++;
                                rr_h[l] += 1.0;
                                rr_h[ll + 1 * (domdcomp_instance.lmx + 1)] = res * sbci[ii];
                                rr_h[ll + 2 * (domdcomp_instance.lmx + 1)] = l + sml;
                            }
                        }
                    }
                }
                if ( ( np - BC_INTER_CURV     ) *
                     ( np - BC_INTER_STRAIGHT ) *
                     ( np - BC_PERIODIC       ) == 0 )
                {
                    for(unsigned int k = 0; k <= domdcomp_instance.ijk[2][nn]; ++k)
                    {
                        for(unsigned int j = 0; j <= domdcomp_instance.ijk[1][nn]; ++j)
                        {
                            int l = host::indx3(i, j, k, nn,
                                domdcomp_instance.lxi, domdcomp_instance.let);
                            nrr_h[l] = 1;
                        }
                    }
                }
            }
        }
        for(unsigned int l = 0; l < domdcomp_instance.lmx + 1; ++l)
        {
            nrr_h[l] = std::min(nrr_h[l] + npex_h[l], 1);
        }
        int lq = ll;
        Type *sbcc_h = (Type *)malloc((lq + 1) * sizeof(Type));
        for(ll = 0; ll <= lq; ++ll)
        {
            int l = rr_h[ll + 2 * (domdcomp_instance.lmx + 1)];
            sbcc_h[ll] = rr_h[ll + 1 * (domdcomp_instance.lmx + 1)] /
                         rr_h[ll + 0 * (domdcomp_instance.lmx + 1)];
        }
        sbcc = allocate_cuda<Type>(lq + 1);
        memcpy_cuda_h2d(sbcc, sbcc_h, lq + 1);
        memcpy_cuda_h2d(npex, npex_h, domdcomp_instance.lmx + 1);
        memcpy_cuda_h2d(nrr , nrr_h , domdcomp_instance.lmx + 1);

        free(sbcc_h);
        free(rr_h);
        free(npex_h);
        free(nrr_h);
        free(yaco_h);
    }

    void go(Type *drva_buffer[3],
            Type *cm_buffer[3],
            Type *qa,
            Type *de,
            Type *pressure,
            Type *yaco,
            t_point<Type> umf,
            t_point<Type> dudtmf,
            t_dcomp dcomp_info,
            Type dtwi,
            int nbc[2][3],
            int mcd[2][3])
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

    void wall_condition_go(Type *qa,
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

    ~gcbc()
    {
        free_cuda(npex);
        free_cuda(nrr);

        free_cuda(drvb_buffer[0]);
        free_cuda(drvb_buffer[1]);
        free_cuda(drvb_buffer[2]);

        free_cuda(sbcc);
    }

    TypeIndex *npex, *nrr;
    Type *drvb_buffer[3];
    Type *sbcc;
};

#endif
