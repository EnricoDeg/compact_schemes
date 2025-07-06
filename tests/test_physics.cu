/*
 * @file test_physics.cu
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

#include <cmath>
#include <iostream>

#include <cuda.h>

#include "test_utils_mpi.hpp"
#include "test_utils.hpp"

#include "mpi/check.hpp"

#include "cuda/common.hpp"

#include "physics_pc.hpp"

float maxval(float *data, unsigned int size)
{
    float res = data[0];
    for(unsigned int i = 1; i< size; ++i)
    {
        if(data[i] > res)
        {
            res = data[i];
        }
    }
    return res;
}

float calc_time_step_cpu(float * xim,
                         float * etm,
                         float * zem,
                         float * de,
                         float * yaco,
                         float * ssk,
                         t_point<float> umf,
                         float   cfl,
                         unsigned int size)
{
    float *tmp = (float *)malloc(size * sizeof(tmp));
    float rr1, rr2, ssi, res_local, fctr;
    float ra0, ra1;
    for(unsigned int i = 0; i < size; ++i)
    {
        rr1 = xim[i] * xim[i] + xim[i + size] * xim[i + size] +
              xim[i + 2 * size] * xim[i + 2 * size] + etm[i] * etm[i] +
              etm[i + size] * etm[i + size] + etm[i + 2 * size] * etm[i + 2 * size] +
              zem[i] * zem[i] + zem[i + size] * zem[i + size] +
              zem[i + 2 * size] * zem[i + 2 * size];
        rr2 = std::abs( xim[i]            * ( de[i + size]     + umf.x )   +
                        xim[i + size]     * ( de[i + 2 * size] + umf.y )   +
                        xim[i + 2 * size] * ( de[i + 3 * size] + umf.z ) ) +
              std::abs( etm[i]            * ( de[i + size]     + umf.x )   +
                        etm[i + size]     * ( de[i + 2 * size] + umf.y )   +
                        etm[i + 2 * size] * ( de[i + 3 * size] + umf.z ) ) +
              std::abs( zem[i]            * ( de[i + size]     + umf.x )   +
                        zem[i + size]     * ( de[i + 2 * size] + umf.y )   +
                        zem[i + 2 * size] * ( de[i + 3 * size] + umf.z ) );
        ssi = std::abs(yaco[i]);
        tmp[i] = ( std::sqrt( de[i + 4 * size] * rr1 ) + rr2 ) * ssi;
    }

    res_local = maxval(tmp, size);

    MPI_Datatype mpi_type = mpi_get_type<float>();
    MPI_Allreduce(&res_local, &fctr, 1, mpi_type, MPI_MAX, MPI_COMM_WORLD);

    ra0 = cfl / fctr;
    ra1 = ra0;

    for(unsigned int i = 0; i < size; ++i)
    {
        rr1 = xim[i] * xim[i] + xim[i + size] * xim[i + size] +
              xim[i + 2 * size] * xim[i + 2 * size] + etm[i] * etm[i] +
              etm[i + size] * etm[i + size] + etm[i + 2 * size] * etm[i + 2 * size] +
              zem[i] * zem[i] + zem[i + size] * zem[i + size] +
              zem[i + 2 * size] * zem[i + 2 * size];
        ssi = std::abs(yaco[i]);
        tmp[i] = de[i] * ssk[i] * rr1 * ssi * ssi;
    }
    res_local = maxval(tmp, size);

    MPI_Allreduce(&res_local, &fctr, 1, mpi_type, MPI_MAX, MPI_COMM_WORLD);

    ra1 = 0.5f / fctr;

    free(tmp);

    return std::min(ra0, ra1);
}

TEST(test_physics, calc_time_step)
{
    // Get the number of processes
    int world_size;
    check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // Get the rank of the process
    int world_rank;
    check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    // Subdomain info
    t_dcomp dcomp_info;
    dcomp_info.lxi = 256;
    dcomp_info.let = 128;
    dcomp_info.lze = 64;
    dcomp_info.lmx = dcomp_info.lxi * dcomp_info.let * dcomp_info.lze;

    // cpu solution
    float *xim  = allocate_and_fill_random(3 * dcomp_info.lmx, 0.0f, 1.0f);
    float *etm  = allocate_and_fill_random(3 * dcomp_info.lmx, 0.0f, 1.0f);
    float *zem  = allocate_and_fill_random(3 * dcomp_info.lmx, 0.0f, 1.0f);
    float *de   = allocate_and_fill_random(5 * dcomp_info.lmx, 0.0f, 1.0f);
    float *yaco = allocate_and_fill_random(1 * dcomp_info.lmx, 0.0f, 1.0f);
    float *ssk  = allocate_and_fill_random(1 * dcomp_info.lmx, 0.0f, 1.0f);

    t_point<float> umf = {.x = 0.3, .y = 0.1, .z = 0.1 };
    float cfl = 0.95;

    float res = calc_time_step_cpu(xim, etm, zem, de, yaco, ssk, umf, cfl, dcomp_info.lmx);

    // physics instance init
    auto physics_instance = physics<false, float>(dcomp_info);
    physics_instance.umf = umf;

    // gpu fields
    float * d_xim = allocate_cuda<float>(3 * dcomp_info.lmx);
    memcpy_cuda_h2d(d_xim, xim, 3 * dcomp_info.lmx);

    float * d_etm = allocate_cuda<float>(3 * dcomp_info.lmx);
    memcpy_cuda_h2d(d_etm, etm, 3 * dcomp_info.lmx);

    float * d_zem = allocate_cuda<float>(3 * dcomp_info.lmx);
    memcpy_cuda_h2d(d_zem, zem, 3 * dcomp_info.lmx);

    float * d_de = allocate_cuda<float>(5 * dcomp_info.lmx);
    memcpy_cuda_h2d(d_de, de, 5 * dcomp_info.lmx);

    float * d_yaco = allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(d_yaco, yaco, dcomp_info.lmx);

    float * d_ssk = allocate_cuda<float>(dcomp_info.lmx);
    memcpy_cuda_h2d(d_ssk, ssk, dcomp_info.lmx);

    float dte = 0.0f;
    physics_instance.calc_time_step(d_xim, d_etm, d_zem,
        d_de, d_yaco, d_ssk,
        cfl, &dte,
        dcomp_info.lmx);

    std::cout << world_rank << ": " << res << " (CPU)" << std::endl;
    std::cout << world_rank << ": " << dte << " (GPU)" << std::endl;
    std::cout << world_rank << ": " << std::abs(res - dte) / res << " (rel diff)" << std::endl;
    ASSERT_TRUE(std::abs(res - dte) / res < 1e-6);

    free(xim);
    free(etm);
    free(zem);
    free(de);
    free(yaco);

    free_cuda(d_xim);
    free_cuda(d_etm);
    free_cuda(d_zem);
    free_cuda(d_yaco);
    free_cuda(d_ssk);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
