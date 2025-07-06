/*
 * @file physics_rtc.hpp
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

#ifndef CANARD_PHYSICS_RTC_HPP
#define CANARD_PHYSICS_RTC_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"
#include "common/nvtx_utils.hpp"
#include "common/utils.hpp"

#include "cuda/compiler.hpp"
#include "numerics_rtc.hpp"
#include "physics_base.hpp"

auto sync_function = [] (CUstream *stream) {
    check_cuda_driver( cuStreamSynchronize ( *stream ) );
};

template<bool EnableViscous, typename Type>
struct physics_rtc : public physics_base<EnableViscous, Type>
{
    using Base = physics_base<EnableViscous, Type>;
    using Base::buffer;
    using Base::buffer_deriv;
    using Base::buffer_ss0;
    using Base::buffer_ss1;
    using Base::buffer_ss2;
    using Base::stress_tensor;
    using Base::heat_fluxes;
    using Base::umf;
    using Base::movef;

    physics_rtc(t_dcomp dcomp_info,
                numerics_rtc<Type> *numerics_instance)
        : physics_base<EnableViscous, Type>(dcomp_info)
    {
        // compile numerics kernels
        host::static_for<0, 3, 1>{}([&](auto nn)
        {
            if(numerics_instance->ndf[0][nn] == 1)
            {
                numerics_instance->template fill_buffer_compile<nn, 0>();
            }

            if(numerics_instance->ndf[1][nn] == 1)
            {
                numerics_instance->template fill_buffer_compile<nn, 1>();
            }

            numerics_instance->template deriv2d_compile<nn>(dcomp_info,
                                                            numerics_instance->ndf[0][nn],
                                                            numerics_instance->ndf[1][nn]);
        });

        // compile physics kernels
        compile_calc_fluxes_pre_compute();

        host::static_for<0, NumberOfVariables, 1>{}([&](auto m)
        {
            this->template compile_calc_fluxes_post_compute<m>();
        });
    }

    ~physics_rtc()
    {

    }

    void calc_fluxes(Type *qa,
                     Type *pressure,
                     Type *de,
                     Type *xim,
                     Type *etm,
                     Type *zem,
                     t_dcomp dcomp_info,
                     int mcd[2][3],
                     numerics_rtc<Type> *numerics_instance,
                     CUstream *streams)
    {
        std::string function_name  = "calc_fluxes";
        NVTX_RANGE(function_name.c_str());

        // Pre-Computations
        dispatch_calc_fluxes_pre_compute(qa,
                                         pressure,
                                         de,
                                         xim,
                                         etm,
                                         zem,
                                         dcomp_info.lmx,
                                         &streams[0]);

        numerics_instance->template fill_buffers<0, NumberOfVariables>(buffer,
                                                                       nrall,
                                                                       dcomp_info,
                                                                       streams);

        host::static_for<0, NumberOfVariables, 1>{}([&](auto m)
        {
            // Halo Exchange
            numerics_instance->mpigo(dcomp_info,
                                     mcd,
                                     m + 1,
                                     m,
                                     &streams[m],
                                     sync_function);

            // Derivatives
            unsigned int offset = m * dcomp_info.lmx * NumberOfSpatialDims;
            numerics_instance->template deriv2d<0>(buffer + offset,
                                                   buffer_deriv + offset,
                                                   dcomp_info,
                                                   m,
                                                   0,
                                                   &streams[m]);

            numerics_instance->template deriv2d<1>(buffer + offset,
                                                   buffer_deriv + offset,
                                                   dcomp_info,
                                                   m,
                                                   1,
                                                   &streams[m]);

            numerics_instance->template deriv2d<2>(buffer + offset,
                                                   buffer_deriv + offset,
                                                   dcomp_info,
                                                   m,
                                                   2,
                                                   &streams[m]);

            this->template dispatch_calc_fluxes_post_compute<m>(de,
                                                                dcomp_info.lmx,
                                                                &streams[m]);
        });

        for(int i=0; i<5; i++) check_cuda_driver( cuStreamSynchronize ( streams[i] ) ); 
    }

    private:
    void compile_calc_fluxes_pre_compute()
    {
        std::vector<std::string> kernel_opts;
        kernel_opts.push_back(std::string("-DDEBUG_INFO=1"));

        std::string kernel_name("calc_fluxes_pre_compute_kernel");
        std::string type_name = GetTypeName<Type>();

        kernel_name += std::string("<") +
            std::to_string(EnableViscous) + std::string(",") + type_name +
            std::string(">");

        std::vector<std::string> kernel_name_vec;
        kernel_name_vec.push_back(kernel_name);
        std::cout << "Compiling " << kernel_name_vec[0] << std::endl;
        calc_fluxes_pre_compute_compiler = new rt_compiler(std::string("cuda/kernels/physics.hpp"),
                                               std::string("calc_fluxes_pre_compute.cu"),
                                               kernel_name_vec,
                                               kernel_opts);
    }

    void dispatch_calc_fluxes_pre_compute(Type *qa,
                                          Type *pressure,
                                          Type *de,
                                          Type *xim,
                                          Type *etm,
                                          Type *zem,
                                          unsigned int size,
                                          CUstream *stream)
    {
        unsigned int blockSize = 256;
        unsigned int blockPerGrid = div_ceil(size, blockSize);

        CUfunction kernel = calc_fluxes_pre_compute_compiler->get_kernel(0);
        void *args[] = { &buffer, &qa, &pressure,
            &de, &xim, &etm, &zem, &stress_tensor, &heat_fluxes,
            &umf, &size };

        TIME_RTC(blockPerGrid, 1, 1,
                 blockSize, 1, 1,
                 0, *stream, false, kernel, args);
    }

    template<unsigned int VariableId>
    void compile_calc_fluxes_post_compute()
    {
        std::vector<std::string> kernel_opts;
        kernel_opts.push_back(std::string("-DDEBUG_INFO=1"));

        std::string kernel_name("calc_fluxes_post_compute_kernel");
        std::string type_name = GetTypeName<Type>();

        kernel_name += std::string("<") +
            std::to_string(VariableId) + std::string(",") + type_name +
            std::string(">");

        std::vector<std::string> kernel_name_vec;
        kernel_name_vec.push_back(kernel_name);
        std::cout << "Compiling " << kernel_name_vec[0] << std::endl;
        calc_fluxes_post_compute_compiler[VariableId] =
            new rt_compiler(std::string("cuda/kernels/physics.hpp"),
                            std::string("calc_fluxes_post_compute_kernel.cu"),
                            kernel_name_vec,
                            kernel_opts);
    }

    template<unsigned int VariableId>
    void dispatch_calc_fluxes_post_compute(Type *de,
                                           unsigned int size,
                                           CUstream *stream)
    {
        unsigned int blockSize = 256;
        unsigned int blockPerGrid = div_ceil(size, blockSize);

        CUfunction kernel = calc_fluxes_post_compute_compiler[VariableId]->get_kernel(0);
        void *args[] = { &de, &buffer, &size };

        TIME_RTC(blockPerGrid, 1, 1,
                 blockSize, 1, 1,
                 0, *stream, true, kernel, args);
    }

    void compile_calc_viscous_shear_stress_init()
    {

    }

    void dispatch_calc_viscous_shear_stress_init()
    {

    }

    template<unsigned int VariableId>
    void compile_calc_viscous_shear_stress_post_compute()
    {

    }

    template<unsigned int VariableId>
    void dispatch_calc_viscous_shear_stress_post_compute()
    {

    }

    void compile_calc_viscous_shear_stress_final()
    {

    }

    void dispatch_calc_viscous_shear_stress_final()
    {

    }

    void compile_calc_time_step()
    {

    }

    void dispatch_calc_time_step()
    {

    }

    rt_compiler * calc_fluxes_pre_compute_compiler;
    rt_compiler * calc_fluxes_post_compute_compiler[NumberOfVariables];
    rt_compiler * calc_viscous_shear_stress_init_compiler;
    rt_compiler * calc_viscous_shear_stress_post_compute_compiler[NumberOfVariables];
    rt_compiler * calc_viscous_shear_stress_final_compiler;
    rt_compiler * calc_time_step_compiler;
};

#endif
