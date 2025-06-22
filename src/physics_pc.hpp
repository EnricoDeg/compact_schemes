/*
 * @file physics_pc.hpp
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

#ifndef CANARD_PHYSICS_PC_HPP
#define CANARD_PHYSICS_PC_HPP

#include "common/data_types.hpp"
#include "common/parameters.hpp"
#include "common/utils.hpp"
#include "common/nvtx_utils.hpp"

#include "cuda/check.hpp"
#include "host/functional.hpp"

#include "cuda/physics.hpp"
#include "cuda/common.hpp"

#include "mpi/data_types.hpp"

#include "numerics_pc.hpp"

#include "physics_base.hpp"

auto sync_function = [] (cudaStream_t *stream) {
    check_cuda( cudaStreamSynchronize(*stream) );
};

template<bool EnableViscous, typename Type>
struct physics : public physics_base<EnableViscous, Type>
{
    using Base = physics_base<EnableViscous, Type>;
    using Base::buffer;
    using Base::buffer_deriv;
    using Base::buffer_ss0;
    using Base::buffer_ss1;
    using Base::buffer_ss2;
    using Base::stress_tensor;
    using Base::heat_fluxes;
    using Base::movef;

    physics(t_dcomp dcomp_info) : physics_base<EnableViscous, Type>(dcomp_info)
    {

    }

    ~physics()
    {

    }

    void init()
    {

    }

    void calc_fluxes(Type *qa, Type *pressure,
                     Type *de,
                     Type *xim, Type *etm, Type *zem,
                     t_dcomp dcomp_info,
                     t_point<Type> umf,
                     Type h_1,
                     unsigned int ndf[2][3],
                     int mcd[2][3],
                     numerics_pc<Type> *numerics_instance,
                     cudaStream_t *streams)
    {
        std::string function_name  = "calc_fluxes";
        NVTX_RANGE(function_name.c_str());

        // Pre-Computations
        calc_fluxes_pre_compute<EnableViscous>(buffer,
            qa, pressure, de, xim, etm, zem, stress_tensor, heat_fluxes,
            umf, dcomp_info.lmx);

        numerics_instance->template fill_buffers<0, NumberOfVariables>(buffer,
                                                                       nrall,
                                                                       dcomp_info,
                                                                       ndf,
                                                                       streams);

        host::static_for<0, NumberOfVariables, 1>{}([&](auto m)
        {
            // Halo Exchange
            numerics_instance->mpigo(dcomp_info,
                                     ndf,
                                     mcd,
                                     m + 1,
                                     m,
                                     &streams[m],
                                     sync_function);

            // Derivatives
            unsigned int offset = m * dcomp_info.lmx * NumberOfSpatialDims;
            numerics_instance->template deriv2d<0>(buffer + offset,
                                                   buffer_deriv + offset,
                                                   h_1,
                                                   ndf[0][0],
                                                   ndf[1][0],
                                                   dcomp_info,
                                                   m,
                                                   0,
                                                   &streams[m]);

            numerics_instance->template deriv2d<1>(buffer + offset,
                                                   buffer_deriv + offset,
                                                   h_1,
                                                   ndf[0][1],
                                                   ndf[1][1],
                                                   dcomp_info,
                                                   m,
                                                   1,
                                                   &streams[m]);

            numerics_instance->template deriv2d<2>(buffer + offset,
                                                   buffer_deriv + offset,
                                                   h_1,
                                                   ndf[0][2],
                                                   ndf[1][2],
                                                   dcomp_info,
                                                   m,
                                                   2,
                                                   &streams[m]);
            calc_fluxes_post_compute<m>(de,
                                        buffer_deriv,
                                        dcomp_info,
                                        &streams[m]);
        });

        for(int i=0; i<5; i++) cudaStreamSynchronize(streams[i]);
    }

    void calc_viscous_shear_stress(Type *de,
                                   Type *ssk,
                                   Type *xim, Type *etm, Type *zem,
                                   Type *yaco,
                                   t_dcomp dcomp_info,
                                   Type h_1,
                                   unsigned int ndf[2][3],
                                   int mcd[2][3],
                                   numerics_pc<Type> *numerics_instance,
                                   cudaStream_t *streams)
    {
        std::string function_name  = "calc_viscous_shear_stress";
        NVTX_RANGE(function_name.c_str());

        // Fill buffer with de
        calc_viscous_shear_stress_init(buffer, de, dcomp_info);

        // fill MPI buffer
        numerics_instance->template fill_buffers<1, NumberOfVariables>(buffer,
                                                                       nrall,
                                                                       dcomp_info,
                                                                       ndf,
                                                                       streams);

        host::static_for<1, NumberOfVariables, 1>{}([&](auto m)
        {
            // Halo Exchange
            numerics_instance->mpigo(dcomp_info,
                                     ndf,
                                     mcd,
                                     m + 1,
                                     m,
                                     &streams[m],
                                     sync_function);

            // Derivatives
            unsigned int offset = m * dcomp_info.lmx * NumberOfSpatialDims;
            numerics_instance->template deriv2d<2>(buffer + offset,
                                                   buffer_deriv + offset,
                                                   h_1,
                                                   ndf[0][0],
                                                   ndf[1][0],
                                                   dcomp_info,
                                                   m,
                                                   0,
                                                   &streams[m]);

            numerics_instance->template deriv2d<1>(buffer + offset,
                                                   buffer_deriv + offset,
                                                   h_1,
                                                   ndf[0][1],
                                                   ndf[1][1],
                                                   dcomp_info,
                                                   m,
                                                   0,
                                                   &streams[m]);

            numerics_instance->template deriv2d<0>(buffer + offset,
                                                   buffer_deriv + offset,
                                                   h_1,
                                                   ndf[0][2],
                                                   ndf[1][2],
                                                   dcomp_info,
                                                   m,
                                                   0,
                                                   &streams[m]);

            calc_viscous_shear_stress_post_compute<m>(
                    stress_tensor, heat_fluxes,
                    buffer_ss0, buffer_ss1, buffer_ss2,
                    buffer,
                    xim, etm, zem,
                    dcomp_info,
                    &streams[m]);
        });

        host::static_for<1, NumberOfVariables, 1>{}([&](auto i)
        {
            check_cuda(cudaStreamSynchronize(streams[i]));
        });

        calc_viscous_shear_stress_final(de, buffer_deriv, ssk, yaco,
            buffer_ss0, buffer_ss1, buffer_ss2, stress_tensor, heat_fluxes, dcomp_info);
    }

    void calc_time_step(Type * xim,
                        Type * etm,
                        Type * zem,
                        Type * de,
                        Type * yaco,
                        Type * ssk,
                        t_point<Type> umf,
                        Type   cfl,
                        Type * dte,
                        unsigned int size)
    {
        std::string function_name  = "calc_time_step";
        NVTX_RANGE(function_name.c_str());

        Type * max_value = (Type*)malloc(2 * sizeof(Type));
        Type * max_value_local = (Type*)malloc(2 * sizeof(Type));

        calc_time_step_dispatch(xim, etm, zem, de, yaco, ssk, umf, size, max_value_local);

        MPI_Datatype mpi_type = mpi_get_type<Type>();
        MPI_Allreduce(&max_value_local[0], &max_value[0], 2, mpi_type, MPI_MAX, MPI_COMM_WORLD);

        max_value[0] = cfl / max_value[0];
        max_value[1] = 0.5 / max_value[1];
        *dte = min(max_value[0], max_value[1]);

        free(max_value);
        free(max_value_local);
    }
};

#endif
