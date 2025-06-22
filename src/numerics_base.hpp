/*
 * @file numerics_base.hpp
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

#ifndef CANARD_NUMERICS_BASE_HPP
#define CANARD_NUMERICS_BASE_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "mpi/exchange.hpp"

#include "cuda/common.hpp"

template<typename Type>
struct numerics_base
{
    numerics_base(t_dcomp dcomp_info)
    {
        size_t face_size0 = dcomp_info.let * dcomp_info.lze;
        size_t face_size1 = dcomp_info.lxi * dcomp_info.lze;
        size_t face_size2 = dcomp_info.lxi * dcomp_info.let;

        drva0 = allocate_cuda<Type>(2 * NumberOfVariables * face_size0);
        drva1 = allocate_cuda<Type>(2 * NumberOfVariables * face_size1);
        drva2 = allocate_cuda<Type>(2 * NumberOfVariables * face_size2);
        drva_buffer[0] = drva0;
        drva_buffer[1] = drva1;
        drva_buffer[2] = drva2;

        send0 = allocate_cuda<Type>(2 * 2 * face_size0 * NumberOfVariables);
        send1 = allocate_cuda<Type>(2 * 2 * face_size1 * NumberOfVariables);
        send2 = allocate_cuda<Type>(2 * 2 * face_size2 * NumberOfVariables);
        send_buffer[0] = send0;
        send_buffer[1] = send1;
        send_buffer[2] = send2;

        recv0 = allocate_cuda<Type>(2 * 2 * face_size0 * NumberOfVariables);
        recv1 = allocate_cuda<Type>(2 * 2 * face_size1 * NumberOfVariables);
        recv2 = allocate_cuda<Type>(2 * 2 * face_size2 * NumberOfVariables);
        recv_buffer[0] = recv0;
        recv_buffer[1] = recv1;
        recv_buffer[2] = recv2;

        pbco = allocate_cuda<Type>(2 * lmd);
        pbci = allocate_cuda<Type>(2 * lmd);
    }

    ~numerics_base()
    {
        free_cuda(drva0);
        free_cuda(drva1);
        free_cuda(drva2);

        free_cuda(send0);
        free_cuda(send1);
        free_cuda(send2);

        free_cuda(recv0);
        free_cuda(recv1);
        free_cuda(recv2);

        free_cuda(pbco);
        free_cuda(pbci);
    }

    // 2D infield
    template<
    typename StreamType,
    typename SyncFunction
    >
    void mpigo(t_dcomp dcomp_info,
               unsigned int ndf[2][3],
               int mcd[2][3],
               int itag,
               unsigned int variable_id,
               StreamType *stream,
               SyncFunction sync)
    {
        sync(stream);
        // cudaStreamSynchronize(*stream);

        // Get the rank of the process
        auto exchange_instance = exchange<Type>();

        Type *send, *recv;
        host::static_for<0, 3, 1>{}([&](auto nn)
        {
            size_t mpi_size;
            size_t face_size;
            int dim;
            if constexpr(nn == 0)
            {
                mpi_size = 2 * dcomp_info.let * dcomp_info.lze;
                face_size = dcomp_info.let * dcomp_info.lze;
                dim = dcomp_info.lxi;
            }
            else if constexpr(nn == 1)
            {
                mpi_size = 2 * dcomp_info.lxi * dcomp_info.lze;
                face_size = dcomp_info.lxi * dcomp_info.lze;
                dim = dcomp_info.let;
            }
            else if constexpr(nn == 2)
            {
                mpi_size = 2 * dcomp_info.lxi * dcomp_info.let;
                face_size = dcomp_info.lxi * dcomp_info.let;
                dim = dcomp_info.lze;
            }

            send = send_buffer[nn] + variable_id * 2 * 2 * face_size;
            recv = recv_buffer[nn] + variable_id * 2 * 2 * face_size;
            exchange_instance.reset_buffer_pointer(send, recv);

            host::static_for<0, 2, 1>{}([&](auto ip)
            {
                static constexpr auto iq = 1 - ip;
                const int istart = ip * (dim - 1);
                const int increment = 1 - 2 * ip;
                const int buffer_offset = ip * mpi_size;
                const int pointer_offset = ip * mpi_size;
                if(ndf[ip][nn] == 1)
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
    }

    void deriv_setup()
    {
        unsigned int size = 2 * lmd;

        // Fill matrix Q
        Type Q[size][size];

        for(unsigned int i = 0; i < size; ++i)
            for(unsigned int j = 0; j < size; ++j)
                Q[i][j] = 0.0;

        Q[0][0] =  ab00;
        Q[0][1] =  ab01;
        Q[0][2] =  ab02;

        Q[1][0] = -ab10;
        Q[1][1] =  0;
        Q[1][2] =  ab10;

        Q[size-1][size-1] = -ab02;
        Q[size-1][size-2] = -ab01;
        Q[size-1][size-3] = -ab00;

        Q[size-2][size-1] =  ab10;
        Q[size-2][size-2] =  0.0;
        Q[size-2][size-3] = -ab10;

        for(unsigned int i = 2; i < size-2; ++i)
        {
            Q[i][i]   =  0.0;
            Q[i][i-1] = -aa;
            Q[i][i+1] =  aa;
            Q[i][i-2] = -bb;
            Q[i][i+2] =  bb;
        }

        // Fill matrix P
        Type P[size][size];

        for(unsigned int i = 0; i < size; ++i)
            for(unsigned int j = 0; j < size; ++j)
                P[i][j] = 0.0;

        P[0][0] = 1.0;
        P[0][1] = alpha01;

        P[size-1][size-1] = 1.0;
        P[size-1][size-2] = alpha01;

        P[1][0] = alpha10;
        P[1][1] = 1.0;
        P[1][2] = alpha10;

        P[size-2][size-1] = alpha10;
        P[size-2][size-2] = 1.0;
        P[size-2][size-3] = alpha10;

        for(unsigned int i = 2; i < size-2; ++i)
        {
            P[i][i]   = 1.0;
            P[i][i-1] = alpha;
            P[i][i+1] = alpha;
        }

        // Compute P^-1
        Type P_1[size][size];

        for(unsigned int i = 0; i < size; ++i)
            for(unsigned int j = 0; j < size; ++j)
                P_1[i][j] = 0.0;

        mtrxi(&P[0][0], &P_1[0][0], size*size);

        // Fill R
        unsigned int i = size / 2 - 2;
        P[i][i+2] = 0.0;
        P[i+1][i+2] = 0.0;
        P[i+1][i+3] = 0.0;

        i = size / 2 + 1;
        P[i][i-2] = 0.0;
        P[i-1][i-2] = 0.0;
        P[i-1][i-3] = 0.0;

        Type tmp1[size][size];
        Type rslt[size][size];
        matmul_square(&P_1[0][0],
                      &Q[0][0],
                      &tmp1[0][0],
                      size*size);

        matmul_square(&P[0][0],
                      &tmp1[0][0],
                      &rslt[0][0],
                      size*size);

        Type h_pbco[lmd * 2];
        Type h_pbci[lmd * 2];

        for(unsigned int i = 0; i < lmd; ++i)
        {
            unsigned int j = size / 2;
            h_pbco[lmd - 1 - i] = rslt[j][i];
            h_pbci[i] = rslt[j][lmd+i];
        }

        for(unsigned int i = 0; i < lmd; ++i)
        {
            unsigned int j = size / 2 + 1;
            h_pbco[lmd + lmd - 1 - i] = rslt[j][i];
            h_pbci[lmd + i] = rslt[j][lmd+i];
        }

        memcpy_cuda_h2d(pbco, h_pbco, 2 * lmd);
        memcpy_cuda_h2d(pbci, h_pbci, 2 * lmd);
    }

    Type *drva0, *drva1, *drva2;
    Type *drva_buffer[3];
    Type *send0, *send1, *send2;
    Type *send_buffer[3];
    Type *recv0, *recv1, *recv2;
    Type *recv_buffer[3];
    Type *pbco, *pbci;

    private:
    int maxloc(Type *p, int size)
    {
        int imax;
        Type dmax;
        imax = 0;
        dmax = p[0];
        for(int i=1; i<size; i++) {
            if (p[i] > dmax) {
                dmax = p[i];
                imax = i;
            }
        }
        return imax;
    }

    void mtrxi(Type *mtrx, Type *imtrx, int size)
    {
        int size1d = sqrt(size);
        int ipvt[size1d];
        Type rx[size];
        Type arx[size];
        Type temp[size1d];
        Type dum;

        // initialize work array
        for (int i=0; i<size; i++) {
            rx[i] = mtrx[i];
        }

        // initialize index work array
        for (int i=0; i<size1d; i++)
            ipvt[i] = i;

        for (int i=0; i<size1d; i++) {
            // absolute value of work array
            for (int k=0; k<size; k++) {
                arx[k] = std::abs(rx[k]);
            }
            int loc = i*size1d+i;
            int imax = maxloc(arx+loc, size1d-i-1);
            int mk = i + imax;

            // swap elements of ipvt and rx
            if (mk!=i) {
                dum = ipvt[mk];
                ipvt[mk] = ipvt[i];
                ipvt[i] = dum;
                for (int l=0; l<size1d; l++) {
                    int mmk = mk + l * size1d;
                    int ii  = i  + l * size1d;
                    dum = rx[mmk];
                    rx[mmk] = rx[ii];
                    rx[ii] = dum;
                }
            }

            Type ra0 = 1.0 / rx[i + i*size1d];
            // fill temporary array
            for (int k=0; k<size1d; k++)
                temp[k] = rx[k+i*size1d];

            for (int k=0; k<size1d; k++) {
                Type ra1 = ra0 * rx[i+k*size1d];
                for (int l=0; l<size1d; l++)
                    rx[l+k*size1d] = rx[l+k*size1d] - ra1 * temp[l];
                rx[i+k*size1d] = ra1;
            }

            for (int k=0; k<size1d; k++)
                rx[k+i*size1d] = -ra0 * temp[k];
            rx[i+i*size1d] = ra0;
        }

        // inverse matrix
        for (int j=0; j<size1d; j++)
            for (int i=0; i<size1d; i++)
                imtrx[i+ipvt[j]*size1d] = rx[i+j*size1d];
    }

    void matmul_square(Type *mat1, Type *mat2, Type *rslt, int size)
    {
        int size1d = sqrt(size);
        for (int j = 0; j < size1d; j++) {
            for (int i = 0; i < size1d; i++) {
                rslt[i+j*size1d] = 0.0;
                for (int k = 0; k < size1d; k++)
                    rslt[i+j*size1d] += mat1[k+j*size1d] * mat2[i+k*size1d];
            }
        }
    }
};

#endif
