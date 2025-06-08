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
        deriv1d_compiler = new rt_compiler(std::string("cuda/kernels/numerics_rtc.hpp"),
                                           std::string("deriv1d.cu"),
                                           kernel_name_vec,
                                           kernel_opts);
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

        CUfunction kernel = deriv1d_compiler->get_kernel(0);
        void *args[] = { &infield, &outfield, &recv_buffer[Axis],
            &pbci, &drva_buffer[Axis],
            &dcomp_info, &variable_id };

        TIME_RTC(blockPerGridX, blockPerGridY, 1,
            blockSize, 1, 1,
            0, *stream, false, kernel, args);
    }

    rt_compiler * deriv1d_compiler;
};

#endif
