/*
 * @file numerics_pc.hpp
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

#ifndef CANARD_NUMERICS_PC_HPP
#define CANARD_NUMERICS_PC_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "host/functional.hpp"

#include "cuda/numerics_pc.hpp"
#include "cuda/common.hpp"

#include "numerics_base.hpp"

template<typename Type>
struct numerics_pc : public numerics_base<Type>
{

    using Base = numerics_base<Type>;
    using Base::drva0;
    using Base::drva1;
    using Base::drva2;
    using Base::drva_buffer;
    using Base::send0;
    using Base::send1;
    using Base::send2;
    using Base::send_buffer;
    using Base::recv0;
    using Base::recv1;
    using Base::recv2;
    using Base::recv_buffer;
    using Base::pbco;
    using Base::pbci;
    using Base::mpigo;
    using Base::ndf;

    numerics_pc(t_dcomp dcomp_info, int nbc[2][3])
        : numerics_base<Type>(dcomp_info, nbc)
    {
    }

    ~numerics_pc()
    {
    }

    // 1D field
    template<unsigned int Axis>
    void deriv1d(Type *infield,
                Type *outfield,
                int nstart,
                int nend,
                t_dcomp dcomp_info,
                unsigned int variable_id)
    {
        static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");

        unsigned int blockSize;
        unsigned int blockPerGridX;
        unsigned int blockPerGridY;
        unsigned int shmem_bytes;
        if constexpr(Axis == 0)
        {
            blockSize = dcomp_info.lxi;
            blockPerGridX = dcomp_info.let;
            blockPerGridY = dcomp_info.lze;
            shmem_bytes = (5 * dcomp_info.lxi) * sizeof(Type);
        }
        else if constexpr(Axis == 1)
        {
            blockSize = dcomp_info.let;
            blockPerGridX = dcomp_info.lxi;
            blockPerGridY = dcomp_info.lze;
            shmem_bytes = (5 * dcomp_info.let) * sizeof(Type);
        }
        else if constexpr(Axis == 2)
        {
            blockSize = dcomp_info.lze;
            blockPerGridX = dcomp_info.lxi;
            blockPerGridY = dcomp_info.let;
            shmem_bytes = (5 * dcomp_info.lze) * sizeof(Type);
        }

        dim3 threadsPerBlock(blockSize, 1);
        dim3 blocksPerGrid(blockPerGridX, blockPerGridY);

        TIME(blocksPerGrid, threadsPerBlock, shmem_bytes, 0, false,
            CANARD_KERNEL_NAME(deriv_kernel_1d<Axis>),
            infield, outfield,
            recv_buffer[Axis], pbci,
            drva_buffer[Axis], nstart, nend, dcomp_info, variable_id);
    }

    // 2D field
    template<unsigned int Axis>
    void deriv2d(Type *infield,
                Type *outfield,
                int nstart,
                int nend,
                t_dcomp dcomp_info,
                unsigned int variable_id,
                unsigned int component_id,
                cudaStream_t *stream)
    {
        static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");

        unsigned int blockSize;
        unsigned int blockPerGridX;
        unsigned int blockPerGridY;
        unsigned int shmem_bytes;
        if constexpr(Axis == 0)
        {
            blockSize = dcomp_info.lxi;
            blockPerGridX = dcomp_info.let;
            blockPerGridY = dcomp_info.lze;
            shmem_bytes = (5 * dcomp_info.lxi) * sizeof(Type);
        }
        else if constexpr(Axis == 1)
        {
            blockSize = dcomp_info.let;
            blockPerGridX = dcomp_info.lxi;
            blockPerGridY = dcomp_info.lze;
            shmem_bytes = (5 * dcomp_info.let) * sizeof(Type);
        }
        else if constexpr(Axis == 2)
        {
            blockSize = dcomp_info.lze;
            blockPerGridX = dcomp_info.lxi;
            blockPerGridY = dcomp_info.let;
            shmem_bytes = (5 * dcomp_info.lze) * sizeof(Type);
        }

        dim3 threadsPerBlock(blockSize, 1);
        dim3 blocksPerGrid(blockPerGridX, blockPerGridY);

        TIME(blocksPerGrid, threadsPerBlock, shmem_bytes, *stream, true,
            CANARD_KERNEL_NAME(deriv_kernel_2d<Axis>),
            infield, outfield,
            recv_buffer[Axis], pbci,
            drva_buffer[Axis], nstart, nend, dcomp_info, variable_id, component_id);
    }

    // 1D infield
    void mpigo(Type *infield,
               t_dcomp dcomp_info,
               int mcd[2][3],
               int itag)
    {
        // Get the rank of the process
        auto exchange_instance = exchange<Type>();

        Type *send, *recv;
        host::static_for<0, 3, 1>{}([&](auto nn)
        {
            send = send_buffer[nn];
            recv = recv_buffer[nn];
            exchange_instance.reset_buffer_pointer(send, recv);

            size_t mpi_size;
            int dim;
            if constexpr(nn == 0)
            {
                mpi_size = 2 * dcomp_info.let * dcomp_info.lze;
                dim = dcomp_info.lxi;
            }
            else if constexpr(nn == 1)
            {
                mpi_size = 2 * dcomp_info.lxi * dcomp_info.lze;
                dim = dcomp_info.let;
            }
            else if constexpr(nn == 2)
            {
                mpi_size = 2 * dcomp_info.lxi * dcomp_info.let;
                dim = dcomp_info.lze;
            }

            auto buffer_instance = gpu_buffer<nn, Type>(infield,
                                                        send,
                                                        pbco,
                                                        dcomp_info);

            host::static_for<0, 2, 1>{}([&](auto ip)
            {
                static constexpr auto iq = 1 - ip;
                const int istart = ip * (dim - 1);
                const int increment = 1 - 2 * ip;
                const int buffer_offset = ip * mpi_size;
                const int pointer_offset = ip * mpi_size;
                if(ndf[ip][nn] == 1)
                {
                    buffer_instance.fill(istart, increment, buffer_offset);
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

    // 2D infield
    template<unsigned int min_value, unsigned int max_value>
    void fill_buffers(Type *infield,
                      int nrt,
                      t_dcomp dcomp_info,
                      cudaStream_t *streams)
    {
        // Get the rank of the process
        Type *send, *recv;
        host::static_for<min_value, max_value, 1>{}([&](auto m)
        {
            host::static_for<0, 3, 1>{}([&](auto nn)
            {
                int nzk = ( 1 - nrt ) * ( nn );

                size_t face_size;
                int dim;
                if constexpr(nn == 0)
                {
                    face_size = dcomp_info.let * dcomp_info.lze;
                    dim = dcomp_info.lxi;
                }
                else if constexpr(nn == 1)
                {
                    face_size = dcomp_info.lxi * dcomp_info.lze;
                    dim = dcomp_info.let;
                }
                else if constexpr(nn == 2)
                {
                    face_size = dcomp_info.lxi * dcomp_info.let;
                    dim = dcomp_info.lze;
                }

                send = send_buffer[nn] + m * 2 * 2 * face_size;
                recv = recv_buffer[nn] + m * 2 * 2 * face_size;

                unsigned int infield_offset = nzk * dcomp_info.lmx +
                    m * dcomp_info.lmx * NumberOfSpatialDims;

                auto buffer_instance = gpu_buffer<nn, Type>(infield + infield_offset,
                                                            send,
                                                            pbco,
                                                            dcomp_info);

                host::static_for<0, 2, 1>{}([&](auto ip)
                {
                    const int istart = ip * (dim - 1);
                    const int increment = 1 - 2 * ip;
                    const int buffer_offset = ip * 2 * face_size;
                    if(ndf[ip][nn] == 1)
                    {
                        buffer_instance.fill(istart, increment, buffer_offset, &streams[m]);
                    }
                });
            });
        });
    }
};

#endif
