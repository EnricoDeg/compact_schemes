/*
 * @file test_physics_serial.cu
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

#include <iostream>

#include <cuda.h>

#include "test_utils.hpp"

#include "common/parameters.hpp"
#include "common/data_types.hpp"
#include "cuda/common.hpp"
#include "cuda/physics.hpp"

void calc_fluxes_pre_compute_cpu(float *buffer,
                                 float *qa,
                                 float *pressure,
                                 float *de,
                                 float *xim,
                                 float *etm,
                                 float *zem,
                                 t_stress_tensor<float> * stress_tensor,
                                 t_heat_fluxes<float> * heat_fluxes,
                                 t_point<float> umf,
                                 unsigned int size)
{
    float ss1, ss2, ss3;

    float *de2 = de + size;
    float *de3 = de + 2 * size;
    float *de4 = de + 3 * size;

    for(unsigned int i = 0; i < size; ++i)
    {
        buffer[i]            = de2[i] + umf.x;
        buffer[i + size]     = de3[i] + umf.y;
        buffer[i + 2 * size] = de4[i] + umf.z;

        ss1 = xim[i]            * buffer[i] +
              xim[i + size]     * buffer[i + size] +
              xim[i + 2 * size] * buffer[i + 2 * size];
        ss2 = etm[i]            * buffer[i] +
              etm[i + size]     * buffer[i + size] +
              etm[i + 2 * size] * buffer[i + 2 * size];
        ss3 = zem[i]            * buffer[i] +
              zem[i + size]     * buffer[i + size] +
              zem[i + 2 * size] * buffer[i + 2 * size];

        buffer[i]            = qa[i] * ss1;
        buffer[i + size]     = qa[i] * ss2;
        buffer[i + 2 * size] = qa[i] * ss3;

        buffer[i +            NumberOfSpatialDims * size] =
            qa[i + size] * ss1 + xim[i] * pressure[i];
        buffer[i + size     + NumberOfSpatialDims * size] =
            qa[i + size] * ss2 + etm[i] * pressure[i];
        buffer[i + 2 * size + NumberOfSpatialDims * size] =
            qa[i + size] * ss3 + zem[i] * pressure[i];

        buffer[i +            NumberOfSpatialDims * size] -=
            (xim[i]            * stress_tensor->xx[i] +
             xim[i + size]     * stress_tensor->xy[i] +
             xim[i + 2 * size] * stress_tensor->zx[i]);
        buffer[i + size     + NumberOfSpatialDims * size] -=
            (etm[i]            * stress_tensor->xx[i] +
             etm[i + size]     * stress_tensor->xy[i] +
             etm[i + 2 * size] * stress_tensor->zx[i]);
        buffer[i + 2 * size + NumberOfSpatialDims * size] -=
            (zem[i]            * stress_tensor->xx[i] +
             zem[i + size]     * stress_tensor->xy[i] +
             zem[i + 2 * size] * stress_tensor->zx[i]);

        buffer[i +            2 * NumberOfSpatialDims * size] =
            qa[i + 2 * size] * ss1 + xim[i + size] * pressure[i];
        buffer[i + size     + 2 * NumberOfSpatialDims * size] =
            qa[i + 2 * size] * ss2 + etm[i + size] * pressure[i];
        buffer[i + 2 * size + 2 * NumberOfSpatialDims * size] =
            qa[i + 2 * size] * ss3 + zem[i + size] * pressure[i];

        buffer[i +            2 * NumberOfSpatialDims * size] -=
            (xim[i]            * stress_tensor->xy[i] +
             xim[i + size]     * stress_tensor->yy[i] +
             xim[i + 2 * size] * stress_tensor->yz[i]);
        buffer[i + size     + 2 * NumberOfSpatialDims * size] -=
            (etm[i]            * stress_tensor->xy[i] +
             etm[i + size]     * stress_tensor->yy[i] +
             etm[i + 2 * size] * stress_tensor->yz[i]);
        buffer[i + 2 * size + 2 * NumberOfSpatialDims * size] -=
            (zem[i]            * stress_tensor->xy[i] +
             zem[i + size]     * stress_tensor->yy[i] +
             zem[i + 2 * size] * stress_tensor->yz[i]);

        buffer[i +            3 * NumberOfSpatialDims * size] =
            qa[i + 3 * size] * ss1 + xim[i + 2 * size] * pressure[i];
        buffer[i + size     + 3 * NumberOfSpatialDims * size] =
            qa[i + 3 * size] * ss2 + etm[i + 2 * size] * pressure[i];
        buffer[i + 2 * size + 3 * NumberOfSpatialDims * size] =
            qa[i + 3 * size] * ss3 + zem[i + 2 * size] * pressure[i];

        buffer[i +            3 * NumberOfSpatialDims * size] -=
            (xim[i]            * stress_tensor->zx[i] +
             xim[i + size]     * stress_tensor->yz[i] +
             xim[i + 2 * size] * stress_tensor->zz[i]);
        buffer[i + size     + 3 * NumberOfSpatialDims * size] -=
            (etm[i]            * stress_tensor->zx[i] +
             etm[i + size]     * stress_tensor->yz[i] +
             etm[i + 2 * size] * stress_tensor->zz[i]);
        buffer[i + 2 * size + 3 * NumberOfSpatialDims * size] -=
            (zem[i]            * stress_tensor->zx[i] +
             zem[i + size]     * stress_tensor->yz[i] +
             zem[i + 2 * size] * stress_tensor->zz[i]);

        de[i + 4 * size] = qa[i + 4 * size] + pressure[i];
        buffer[i +            4 * NumberOfSpatialDims * size] =
            de[i + 4 * size] * ss1 - pressure[i] *
            (umf.x * xim[i] + umf.y * xim[i + size] + umf.z * xim[i + 2 * size]);
        buffer[i + size     + 4 * NumberOfSpatialDims * size] =
            de[i + 4 * size] * ss2 - pressure[i] *
            (umf.x * etm[i] + umf.y * etm[i + size] + umf.z * etm[i + 2 * size]);
        buffer[i + 2 * size + 4 * NumberOfSpatialDims * size] =
            de[i + 4 * size] * ss3 - pressure[i] *
            (umf.x * zem[i] + umf.y * zem[i + size] + umf.z * zem[i + 2 * size]);

        buffer[i +            4 * NumberOfSpatialDims * size] -=
            (xim[i]            * heat_fluxes->xx[i] +
             xim[i + size]     * heat_fluxes->yy[i] +
             xim[i + 2 * size] * heat_fluxes->zz[i]);
        buffer[i + size     + 4 * NumberOfSpatialDims * size] -=
            (etm[i]            * heat_fluxes->xx[i] +
             etm[i + size]     * heat_fluxes->yy[i] +
             etm[i + 2 * size] * heat_fluxes->zz[i]);
        buffer[i + 2 * size + 4 * NumberOfSpatialDims * size] -=
            (zem[i]            * heat_fluxes->xx[i] +
             zem[i + size]     * heat_fluxes->yy[i] +
             zem[i + 2 * size] * heat_fluxes->zz[i]);
    }
}

TEST(test_physics_serial, calc_fluxes_pre_compute)
{
    // Subdomain info
    t_dcomp dcomp_info;
    dcomp_info.lxi = 1024;
    dcomp_info.let = 1;
    dcomp_info.lze = 1;
    dcomp_info.lmx = dcomp_info.lxi * dcomp_info.let * dcomp_info.lze;

    // cpu solution
    float *buffer     = (float *)malloc(NumberOfSpatialDims * NumberOfVariables * dcomp_info.lmx * sizeof(float));
    float *buffer_out = (float *)malloc(NumberOfSpatialDims * NumberOfVariables * dcomp_info.lmx * sizeof(float));

    float *qa       = allocate_and_fill_random(NumberOfVariables   * dcomp_info.lmx, 0.0f, 1.0f);
    float *pressure = allocate_and_fill_random(dcomp_info.lmx, 0.0f, 1.0f);
    float *xim      = allocate_and_fill_random(NumberOfSpatialDims * dcomp_info.lmx, 0.0f, 1.0f);
    float *etm      = allocate_and_fill_random(NumberOfSpatialDims * dcomp_info.lmx, 0.0f, 1.0f);
    float *zem      = allocate_and_fill_random(NumberOfSpatialDims * dcomp_info.lmx, 0.0f, 1.0f);
    float *de       = allocate_and_fill_random(NumberOfVariables   * dcomp_info.lmx, 0.0f, 1.0f);
    t_stress_tensor<float> stress_tensor;
    stress_tensor.xx = allocate_and_fill_random(dcomp_info.lmx, 0.1f, 1.0f);
    stress_tensor.yy = allocate_and_fill_random(dcomp_info.lmx, 0.1f, 1.0f);
    stress_tensor.zz = allocate_and_fill_random(dcomp_info.lmx, 0.1f, 1.0f);
    stress_tensor.xy = allocate_and_fill_random(dcomp_info.lmx, 0.1f, 1.0f);
    stress_tensor.yz = allocate_and_fill_random(dcomp_info.lmx, 0.1f, 1.0f);
    stress_tensor.zx = allocate_and_fill_random(dcomp_info.lmx, 0.1f, 1.0f);
    t_heat_fluxes<float> heat_fluxes;
    heat_fluxes.xx = allocate_and_fill_random(dcomp_info.lmx, 0.1f, 1.0f);
    heat_fluxes.yy = allocate_and_fill_random(dcomp_info.lmx, 0.1f, 1.0f);
    heat_fluxes.zz = allocate_and_fill_random(dcomp_info.lmx, 0.1f, 1.0f);

    t_point<float> umf = {.x = 0.3, .y = 0.1, .z = 0.1 };

    calc_fluxes_pre_compute_cpu(buffer,
                                qa,
                                pressure,
                                de,
                                xim,
                                etm,
                                zem,
                                &stress_tensor,
                                &heat_fluxes,
                                umf,
                                dcomp_info.lmx);

    // gpu fields
    float * d_buffer = allocate_cuda<float>(NumberOfSpatialDims * NumberOfVariables * dcomp_info.lmx);

    float * d_qa = allocate_cuda<float>(5 * dcomp_info.lmx);
    memcpy_cuda_h2d(d_qa, qa, 5 * dcomp_info.lmx);

    float * d_pressure = allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(d_pressure, pressure, dcomp_info.lmx);

    float * d_xim = allocate_cuda<float>(3 * dcomp_info.lmx);
    memcpy_cuda_h2d(d_xim, xim, 3 * dcomp_info.lmx);

    float * d_etm = allocate_cuda<float>(3 * dcomp_info.lmx);
    memcpy_cuda_h2d(d_etm, etm, 3 * dcomp_info.lmx);

    float * d_zem = allocate_cuda<float>(3 * dcomp_info.lmx);
    memcpy_cuda_h2d(d_zem, zem, 3 * dcomp_info.lmx);

    float * d_de = allocate_cuda<float>(5 * dcomp_info.lmx);
    memcpy_cuda_h2d(d_de, de, 5 * dcomp_info.lmx);

    t_stress_tensor<float> * d_stress_tensor = allocate_cuda<t_stress_tensor<float>>(1);

    float *tmp = allocate_cuda<float>(dcomp_info.lmx);

    memcpy_cuda_h2d(tmp, stress_tensor.xx, dcomp_info.lmx);
    memcpy_cuda_h2d(&d_stress_tensor->xx, &tmp, 1);

    tmp =  allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(tmp, stress_tensor.yy, dcomp_info.lmx);
    memcpy_cuda_h2d(&d_stress_tensor->yy, &tmp, 1);

    tmp =  allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(tmp, stress_tensor.zz, dcomp_info.lmx);
    memcpy_cuda_h2d(&d_stress_tensor->zz, &tmp, 1);

    tmp =  allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(tmp, stress_tensor.xy, dcomp_info.lmx);
    memcpy_cuda_h2d(&d_stress_tensor->xy, &tmp, 1);

    tmp =  allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(tmp, stress_tensor.yz, dcomp_info.lmx);
    memcpy_cuda_h2d(&d_stress_tensor->yz, &tmp, 1);

    tmp =  allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(tmp, stress_tensor.zx, dcomp_info.lmx);
    memcpy_cuda_h2d(&d_stress_tensor->zx, &tmp, 1);

    t_heat_fluxes<float> * d_heat_fluxes = allocate_cuda<t_heat_fluxes<float>>(1);

    tmp =  allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(tmp, heat_fluxes.xx, dcomp_info.lmx);
    memcpy_cuda_h2d(&d_heat_fluxes->xx, &tmp, 1);

    tmp =  allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(tmp, heat_fluxes.yy, dcomp_info.lmx);
    memcpy_cuda_h2d(&d_heat_fluxes->yy, &tmp, 1);

    tmp =  allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(tmp, heat_fluxes.zz, dcomp_info.lmx);
    memcpy_cuda_h2d(&d_heat_fluxes->zz, &tmp, 1);

    calc_fluxes_pre_compute<true>(d_buffer,
                                  d_qa,
                                  d_pressure,
                                  d_de,
                                  d_xim,
                                  d_etm,
                                  d_zem,
                                  d_stress_tensor,
                                  d_heat_fluxes,
                                  umf,
                                  dcomp_info.lmx);

    memcpy_cuda_d2h(buffer_out, d_buffer, NumberOfSpatialDims * NumberOfVariables * dcomp_info.lmx);

    for(unsigned int i = 0; i < NumberOfSpatialDims * NumberOfVariables * dcomp_info.lmx; ++i)
    {
        ASSERT_TRUE(std::abs(buffer_out[i] - buffer[i]) / buffer[i] < 1e-3);
    }

    free(buffer);
    free(buffer_out);
    free(qa);
    free(pressure);
    free(xim);
    free(etm);
    free(zem);
    free(de);
    free(stress_tensor.xx);
    free(stress_tensor.yy);
    free(stress_tensor.zz);
    free(stress_tensor.xy);
    free(stress_tensor.yz);
    free(stress_tensor.zx);
    free(heat_fluxes.xx);
    free(heat_fluxes.yy);
    free(heat_fluxes.zz);

    free_cuda(d_buffer);
    free_cuda(d_qa);
    free_cuda(d_pressure);
    free_cuda(d_xim);
    free_cuda(d_etm);
    free_cuda(d_zem);
    free_cuda(d_de);

    memcpy_cuda_d2h(&tmp, &d_stress_tensor->xx, 1);
    free_cuda(tmp);

    memcpy_cuda_d2h(&tmp, &d_stress_tensor->yy, 1);
    free_cuda(tmp);

    memcpy_cuda_d2h(&tmp, &d_stress_tensor->zz, 1);
    free_cuda(tmp);

    memcpy_cuda_d2h(&tmp, &d_stress_tensor->xy, 1);
    free_cuda(tmp);

    memcpy_cuda_d2h(&tmp, &d_stress_tensor->yz, 1);
    free_cuda(tmp);

    memcpy_cuda_d2h(&tmp, &d_stress_tensor->zx, 1);
    free_cuda(tmp);

    free_cuda(d_stress_tensor);

    memcpy_cuda_d2h(&tmp, &d_heat_fluxes->xx, 1);
    free_cuda(tmp);

    memcpy_cuda_d2h(&tmp, &d_heat_fluxes->yy, 1);
    free_cuda(tmp);

    memcpy_cuda_d2h(&tmp, &d_heat_fluxes->zz, 1);
    free_cuda(tmp);

    free_cuda(d_heat_fluxes);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
