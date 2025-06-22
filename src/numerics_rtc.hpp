/*
 * @file numerics_rtc.hpp
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

#ifndef CANARD_NUMERICS_RTC_HPP
#define CANARD_NUMERICS_RTC_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "host/functional.hpp"

#include "cuda/common.hpp"
#include "cuda/compiler.hpp"
#include "cuda/dispatch.hpp"

#include "numerics_base.hpp"

template<typename Type>
struct numerics_rtc : public numerics_base<Type>
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

    numerics_rtc(t_dcomp dcomp_info) : numerics_base<Type>(dcomp_info)
    {
    }

    ~numerics_rtc()
    {
    }

    template <unsigned int Axis, typename T>
    std::string getKernelNameForType(std::string kernel_name,
        unsigned int BlockSize)
    {
        // Look up the source level name string for the type "T" using
        // nvrtcGetTypeName() and use it to create the kernel name
        std::string type_name = GetTypeName<T>();

        return kernel_name + std::string("<") +
            std::to_string(BlockSize) + std::string(",") +
            std::to_string(Axis) + std::string(",") + type_name +
            std::string(">");
    }

    template<unsigned int Axis>
    void deriv1d_compile(t_dcomp dcomp_info,
        int nstart,
        int nend)
    {
        int iter;
        unsigned int blockSize;
        unsigned int blocksize_local;
        unsigned int lxi_local;
        if constexpr(Axis == 0)
        {
            blockSize = dcomp_info.lxi;
            iter = static_cast<int>(std::log2f(dcomp_info.lxi / 2));
            blocksize_local = blockSize;
            lxi_local = blockSize / (blocksize_local);
        }
        else if constexpr(Axis == 1)
        {
            blockSize = 1024;
            iter = static_cast<int>(std::log2f(dcomp_info.let / 2));
            blocksize_local = dcomp_info.let;
            lxi_local = blockSize / (blocksize_local);
        }
        else if constexpr(Axis == 2)
        {
            blockSize = 1024;
            iter = static_cast<int>(std::log2f(dcomp_info.lze / 2));
            blocksize_local = dcomp_info.lze;
            lxi_local = blockSize / (blocksize_local);
        }

        std::vector<std::string> kernel_opts;
        kernel_opts.push_back(std::string("-DDEBUG_INFO=1"));
        kernel_opts.push_back(std::string("-DITERS="+std::to_string(iter)));
        kernel_opts.push_back(std::string("-DNSTART="+std::to_string(nstart)));
        kernel_opts.push_back(std::string("-DNEND="+std::to_string(nend)));
        kernel_opts.push_back(std::string("-DBLOCKSIZE_LOCAL="+std::to_string(blocksize_local)));
        kernel_opts.push_back(std::string("-DLXI_LOCAL="+std::to_string(lxi_local)));

        std::string kernel_name("deriv_kernel_1d");
        std::vector<std::string> kernel_name_vec;
        kernel_name_vec.push_back(getKernelNameForType<Axis, Type>(kernel_name,
                                                                   blockSize));
        std::cout << "Compiling " << kernel_name_vec[0] << std::endl;
        rt_compiler * deriv1d_instance = new rt_compiler(std::string("cuda/kernels/numerics_rtc.hpp"),
                                           std::string("deriv1d.cu"),
                                           kernel_name_vec,
                                           kernel_opts);
        deriv1d_compiler[Axis] = deriv1d_instance;
    }

    template<unsigned int Axis>
    void deriv2d_compile(t_dcomp dcomp_info,
        int nstart,
        int nend)
    {
        int iter;
        unsigned int blockSize;
        unsigned int blocksize_local;
        unsigned int lxi_local;
        if constexpr(Axis == 0)
        {
            blockSize = dcomp_info.lxi;
            iter = static_cast<int>(std::log2f(dcomp_info.lxi / 2));
            blocksize_local = blockSize;
            lxi_local = blockSize / (blocksize_local);
        }
        else if constexpr(Axis == 1)
        {
            blockSize = 1024;
            iter = static_cast<int>(std::log2f(dcomp_info.let / 2));
            blocksize_local = dcomp_info.let;
            lxi_local = blockSize / (blocksize_local);
        }
        else if constexpr(Axis == 2)
        {
            blockSize = 1024;
            iter = static_cast<int>(std::log2f(dcomp_info.lze / 2));
            blocksize_local = dcomp_info.lze;
            lxi_local = blockSize / (blocksize_local);
        }

        std::vector<std::string> kernel_opts;
        kernel_opts.push_back(std::string("-DDEBUG_INFO=1"));
        kernel_opts.push_back(std::string("-DITERS="+std::to_string(iter)));
        kernel_opts.push_back(std::string("-DNSTART="+std::to_string(nstart)));
        kernel_opts.push_back(std::string("-DNEND="+std::to_string(nend)));
        kernel_opts.push_back(std::string("-DBLOCKSIZE_LOCAL="+std::to_string(blocksize_local)));
        kernel_opts.push_back(std::string("-DLXI_LOCAL="+std::to_string(lxi_local)));

        std::string kernel_name("deriv_kernel_2d");
        std::vector<std::string> kernel_name_vec;
        kernel_name_vec.push_back(getKernelNameForType<Axis, Type>(kernel_name,
                                                                   blockSize));
        std::cout << "Compiling " << kernel_name_vec[0] << std::endl;
        rt_compiler * deriv2d_instance = new rt_compiler(std::string("cuda/kernels/numerics_rtc.hpp"),
                                           std::string("deriv2d.cu"),
                                           kernel_name_vec,
                                           kernel_opts);
        deriv2d_compiler[Axis] = deriv2d_instance;
    }

    template<unsigned int Axis, unsigned int FaceId>
    void fill_buffer_compile()
    {

        std::vector<std::string> kernel_opts;
        kernel_opts.push_back(std::string("-DDEBUG_INFO=1"));

        std::string kernel_name("fill_gpu_buffer_kernel");
        std::string type_name = GetTypeName<Type>();

        kernel_name += std::string("<") +
            std::to_string(Axis) + std::string(",") + type_name +
            std::string(">");

        std::vector<std::string> kernel_name_vec;
        kernel_name_vec.push_back(kernel_name);
        std::cout << "Compiling " << kernel_name_vec[0] << std::endl;
        rt_compiler * fill_buffer_instance = new rt_compiler(std::string("cuda/kernels/numerics_rtc.hpp"),
                                               std::string("fill_buffer.cu"),
                                               kernel_name_vec,
                                               kernel_opts);
        fill_buffer_compiler[Axis + FaceId * NumberOfSpatialDims] = fill_buffer_instance;
    }

    // 1D field
    template<unsigned int Axis>
    void deriv1d(Type *infield,
                Type *outfield,
                t_dcomp dcomp_info,
                unsigned int variable_id,
                CUstream *stream)
    {
        static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");

        unsigned int blockSize;
        unsigned int blockPerGridX;
        unsigned int blockPerGridY;
        if constexpr(Axis == 0)
        {
            blockSize = dcomp_info.lxi;
            blockPerGridX = dcomp_info.let;
            blockPerGridY = dcomp_info.lze;
        }
        else if constexpr(Axis == 1)
        {
            blockSize = 1024;
            blockPerGridX = dcomp_info.lxi / (blockSize / dcomp_info.let);
            blockPerGridY = dcomp_info.lze;
        }
        else if constexpr(Axis == 2)
        {
            blockSize = 1024;
            blockPerGridX = dcomp_info.lxi / (blockSize / dcomp_info.lze);
            blockPerGridY = dcomp_info.let;
        }

        CUfunction kernel = deriv1d_compiler[Axis]->get_kernel(0);
        void *args[] = { &infield, &outfield, &recv_buffer[Axis],
            &pbci, &drva_buffer[Axis],
            &dcomp_info, &variable_id };

        TIME_RTC(blockPerGridX, blockPerGridY, 1,
            blockSize, 1, 1,
            0, *stream, false, kernel, args);
    }

    // 2D infield
    template<unsigned int Axis>
    void deriv2d(Type *infield,
                 Type *outfield,
                 t_dcomp dcomp_info,
                 unsigned int variable_id,
                 unsigned int component_id,
                 CUstream *stream)
    {
        static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");

        unsigned int blockSize;
        unsigned int blockPerGridX;
        unsigned int blockPerGridY;
        if constexpr(Axis == 0)
        {
            blockSize = dcomp_info.lxi;
            blockPerGridX = dcomp_info.let;
            blockPerGridY = dcomp_info.lze;
        }
        else if constexpr(Axis == 1)
        {
            blockSize = 1024;
            blockPerGridX = dcomp_info.lxi / (blockSize / dcomp_info.let);
            blockPerGridY = dcomp_info.lze;
        }
        else if constexpr(Axis == 2)
        {
            blockSize = 1024;
            blockPerGridX = dcomp_info.lxi / (blockSize / dcomp_info.lze);
            blockPerGridY = dcomp_info.let;
        }

        CUfunction kernel = deriv2d_compiler[Axis]->get_kernel(0);
        void *args[] = { &infield, &outfield, &recv_buffer[Axis],
            &pbci, &drva_buffer[Axis],
            &dcomp_info, &variable_id, &component_id };

        TIME_RTC(blockPerGridX, blockPerGridY, 1,
            blockSize, 1, 1,
            0, *stream, true, kernel, args);
    }

    template<unsigned int min_value, unsigned int max_value>
    void fill_buffers(Type *infield,
                      int nrt,
                      t_dcomp dcomp_info,
                      unsigned int ndf[2][3],
                      CUstream *streams)
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

                {
                    const int ip = 0;
                    const int istart = ip * (dim - 1);
                    const int increment = 1 - 2 * ip;
                    const int buffer_offset = ip * 2 * face_size;
                    if(ndf[ip][nn] == 1)
                    {
                        fill_buffer<nn, ip>(infield + infield_offset,
                                            send,
                                            pbco,
                                            dcomp_info,
                                            istart,
                                            increment,
                                            buffer_offset,
                                            &streams[m]);
                    }
                }
                {
                    const int ip = 1;
                    const int istart = ip * (dim - 1);
                    const int increment = 1 - 2 * ip;
                    const int buffer_offset = ip * 2 * face_size;
                    if(ndf[ip][nn] == 1)
                    {
                        fill_buffer<nn, ip>(infield + infield_offset,
                                            send,
                                            pbco,
                                            dcomp_info,
                                            istart,
                                            increment,
                                            buffer_offset,
                                            &streams[m]);
                    }
                }
            });
        });
    }

    private:

    template<unsigned int Axis, unsigned int FaceId>
    void fill_buffer(Type *infield,
                     Type *buffer,
                     Type *pbco,
                     t_dcomp dcomp_info,
                     int istart,
                     int increment,
                     int buffer_offset,
                     CUstream *stream)
    {
        static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");

        unsigned int blockSize;
        unsigned int blockPerGridX;
        if constexpr(Axis == 0)
        {
            blockSize = dcomp_info.let;
            blockPerGridX = dcomp_info.lze;
        }
        else if constexpr(Axis == 1)
        {
            blockSize = dcomp_info.lxi;
            blockPerGridX = dcomp_info.lze;
        }
        else if constexpr(Axis == 2)
        {
            blockSize = dcomp_info.lxi;
            blockPerGridX = dcomp_info.let;
        }

        CUfunction kernel = fill_buffer_compiler[Axis + FaceId * NumberOfSpatialDims]->get_kernel(0);
        void *args[] = { &infield, &buffer, &pbco,
            &dcomp_info, &istart,
            &increment, &buffer_offset };

        TIME_RTC(blockPerGridX, 1, 1,
            blockSize, 1, 1,
            0, *stream, true, kernel, args);
    }

    rt_compiler * deriv1d_compiler[NumberOfSpatialDims];
    rt_compiler * deriv2d_compiler[NumberOfSpatialDims];
    rt_compiler * fill_buffer_compiler[2 * NumberOfSpatialDims];
};

#endif
