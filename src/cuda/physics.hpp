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

#ifndef CANARD_CUDA_PHYSICS_HPP
#define CANARD_CUDA_PHYSICS_HPP

#include "common.hpp"
#include "common/utils.hpp"

#include "cuda/kernels/physics.hpp"
#include "cuda/check.hpp"
#include "cuda/dispatch.hpp"
#include "cuda/common.hpp"

template<bool EnableViscous, typename Type>
void calc_fluxes_pre_compute(Type *buffer,
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
    unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(size, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGrid);

    TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
        CANARD_KERNEL_NAME(calc_fluxes_pre_compute_kernel<EnableViscous>),
        buffer, qa, pressure, de, xim, etm, zem, stress_tensor, heat_fluxes, umf, size);
}

template<
unsigned int VariableId,
typename Type>
void calc_fluxes_post_compute(Type *de,
                              Type *buffer,
                              t_dcomp dcomp_info,
                              cudaStream_t *stream)
{
    unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(dcomp_info.lmx, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGrid);

    TIME(blocksPerGrid, threadsPerBlock, 0, *stream, true,
        CANARD_KERNEL_NAME(calc_fluxes_post_compute_kernel<VariableId>),
        de, buffer, dcomp_info.lmx);
}

template<typename Type>
void calc_viscous_shear_stress_init(Type *buffer,
                                    Type *de,
                                    t_dcomp dcomp_info)
{
    unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(dcomp_info.lmx, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGrid);

    TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
        CANARD_KERNEL_NAME(calc_viscous_shear_stress_init_kernel),
        buffer, de, dcomp_info.lmx);
}

template<
unsigned int VariableId,
typename Type>
void calc_viscous_shear_stress_post_compute(
                                            t_stress_tensor<Type> * stress_tensor,
                                            t_heat_fluxes<Type> * heat_fluxes,
                                            Type *buffer_ss0,
                                            Type *buffer_ss1,
                                            Type *buffer_ss2,
                                            Type *buffer,
                                            Type *xim,
                                            Type *etm,
                                            Type *zem,
                                            t_dcomp dcomp_info,
                                            cudaStream_t *stream)
{
    unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(dcomp_info.lmx, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGrid);

    TIME(blocksPerGrid, threadsPerBlock, 0, *stream, true,
        CANARD_KERNEL_NAME(calc_viscous_shear_stress_post_compute_kernel<VariableId>),
        stress_tensor, heat_fluxes, buffer_ss0, buffer_ss1, buffer_ss2,
        buffer, xim, etm, zem, dcomp_info.lmx);
}

template<typename Type>
void calc_viscous_shear_stress_final(Type *de,
                                     Type *buffer,
                                     Type *ssk,
                                     Type *yaco,
                                     Type *buffer_ss0,
                                     Type *buffer_ss1,
                                     Type *buffer_ss2,
                                     t_stress_tensor<Type> * stress_tensor,
                                     t_heat_fluxes<Type> * heat_fluxes,
                                     t_dcomp dcomp_info)
{
    unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(dcomp_info.lmx, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGrid);

    TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
        CANARD_KERNEL_NAME(calc_viscous_shear_stress_final_kernel),
        de, buffer, ssk, yaco, buffer_ss0, buffer_ss1, buffer_ss2,
        stress_tensor, heat_fluxes, dcomp_info.lmx);
}

template<typename Type>
void calc_time_step_dispatch(Type * xim,
                             Type * etm,
                             Type * zem,
                             Type * de,
                             Type * yaco,
                             Type * ssk,
                             t_point<Type> umf,
                             unsigned int size,
                             Type * max_value_local)
{
    static constexpr unsigned int blockSize = 256;
    unsigned int blockPerGrid = div_ceil(size, blockSize);
    dim3 threadsPerBlock(blockSize, 1);
    dim3 blocksPerGrid(blockPerGrid);

    Type *d_max_value_local = allocate_cuda<Type>(2);

    TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
        CANARD_KERNEL_NAME(calc_time_step_kernel<blockSize>),
        xim, etm, zem, de, yaco, ssk, umf, d_max_value_local, size);

    memcpy_cuda_d2h(max_value_local, d_max_value_local, 2);
    free_cuda(d_max_value_local);
}

#endif
