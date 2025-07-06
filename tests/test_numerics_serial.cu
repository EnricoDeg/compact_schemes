/*
 * @file test_numerics_serial.cu
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
#include "numerics_rtc.hpp"

template<unsigned int Axis>
void test_deriv1d_rct()
{
    // Subdomain info
    t_dcomp dcomp_info;
    dcomp_info.lxi = 256;
    dcomp_info.let = 128;
    dcomp_info.lze = 64;
    dcomp_info.lmx = dcomp_info.lxi * dcomp_info.let * dcomp_info.lze;

    // cpu solution
    float *infield = (float *)malloc(dcomp_info.lmx * sizeof(float));
    float *d_infield;
    check_cuda( cudaMalloc(&d_infield, dcomp_info.lmx * sizeof(float)) );

    float *outfield = (float *)malloc(dcomp_info.lmx * sizeof(float));
    float *d_outfield;
    check_cuda( cudaMalloc(&d_outfield, dcomp_info.lmx * sizeof(float)) );

    for(unsigned int i = 0; i < dcomp_info.lmx; ++i)
    {
        infield[i] = i;
    }
    check_cuda( cudaMemcpy(d_infield, infield,
        dcomp_info.lmx * sizeof(float), cudaMemcpyHostToDevice));

    int nstart = 0;
    int nend   = 0;
    int nbc[2][3];
    for(unsigned int nn = 0; nn < NumberOfSpatialDims; ++nn)
    {
        for(unsigned int ip = 0; ip < NumberOfFaces; ++ip)
        {
            nbc[ip][nn] = BC_NON_REFLECTIVE;
        }
    }

    auto numerics_instance = numerics_rtc<float>(dcomp_info, nbc);

    numerics_instance.template deriv1d_compile<Axis>(dcomp_info, nstart, nend);

    constexpr unsigned int NStreams = 1;
    CUstream streams[NStreams];
    for(unsigned int i = 0; i < NStreams; ++i)
    {
       check_cuda_driver(cuStreamCreate ( &streams[i], CU_STREAM_NON_BLOCKING ));
    }

    // setup derivatives
    numerics_instance.deriv_setup();

    for(unsigned int i = 0; i < 10; ++i)
    {
        numerics_instance.template deriv1d<Axis>(d_infield,
            d_outfield,
            dcomp_info,
            0,
            streams);
    }

    for(unsigned int i = 0; i < NStreams; ++i)
    {
        check_cuda_driver( cuStreamDestroy ( streams[i] ));
    }

    check_cuda( cudaMemcpy(outfield, d_outfield,
        dcomp_info.lmx * sizeof(float), cudaMemcpyDeviceToHost) );

    float solution;
    if constexpr(Axis == 0)
    {
        solution = infield[1] - infield[0];
    }
    else if constexpr(Axis == 1)
    {
        solution = infield[dcomp_info.lxi] - infield[0];
    }
    else if constexpr(Axis == 2)
    {
        solution = infield[dcomp_info.lxi * dcomp_info.let] - infield[0];
    }
    for(unsigned int i = 0; i < dcomp_info.lmx; ++i)
    {
        ASSERT_TRUE(std::abs(outfield[i] - solution) / solution < 1e-6);
        // if(std::abs(outfield[i] - solution) / solution > 1e-6)
        // {
        //     std::cout << std::abs(outfield[i] - solution) / solution << std::endl;
        //     std::cout << i << ": out = " << outfield[i] << " -- ref = " << solution << std::endl;
        //     exit(1);
        // }
    }

    free(infield);
    free(outfield);


    free_cuda(d_infield);
    free_cuda(d_outfield);
}

TEST(test_numerics_serial, deriv1d)
{
    test_deriv1d_rct<0>();
    test_deriv1d_rct<1>();
    test_deriv1d_rct<2>();
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
