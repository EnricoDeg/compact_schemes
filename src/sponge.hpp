/*
 * @file sponge.hpp
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

#ifndef CANARD_SPONGE_HPP
#define CANARD_SPONGE_HPP

#include "common/parameters.hpp"
#include "common/nvtx_utils.hpp"

#include "cuda/common.hpp"
#include "cuda/kernels/sponge.hpp"

#include "grid.hpp"

template<typename Type>
struct sponge
{

    sponge()
    {

    }

    void read_config(YAML::Node& sponge_yaml)
    {
        YAML::Node szco_node = sponge_yaml[0]["szco"];
        szco = szco_node.as<float>();
    }

    void up(grid<Type>& grid_instance, unsigned int lmx)
    {
        Type * tmp = (Type *)malloc(3 * lmx * sizeof(Type));
        int ll = -1;
        Type ra2 = 0.0;
        Type tmpa = pi / grid_instance.szth0;
        Type tmpb = pi / grid_instance.szth1;
        Type ra0, ra1, ra3;

        for(int i = 0; i < lmx; ++i)
        {
            ra3 = ra2 * grid_instance.patch[1][i];
            ra0 = tmpa * (grid_instance.patch[0][i] -
                (ra3 - grid_instance.doml0 + grid_instance.szth0));
            ra1 = tmpb * (ra3 + grid_instance.doml1 -
                grid_instance.szth1 - grid_instance.patch[0][i]);
            tmp[i] = szco * 0.5 * (2.0 +
                std::cos(std::max(std::min(ra0, static_cast<Type>(pi)), static_cast<Type>(0.0))) +
                std::cos(std::max(std::min(ra1, static_cast<Type>(pi)), static_cast<Type>(0.0))));
            tmp[i + lmx] = szco * 0.5 * (1.0 +
                std::cos(std::max(std::min(ra0, static_cast<Type>(pi)), static_cast<Type>(0.0))));
            if(tmp[i] > sml)
            {
                ll++;
                tmp[i + 2 * lmx] = i + sml;
            }
        }
        lsz = ll;
        if(lsz != -1)
        {
            lcsz = allocate_cuda<int>(lsz + 1);
            asz  = allocate_cuda<Type>(lsz + 1);
            bsz  = allocate_cuda<Type>(lsz + 1);
            int  *lcsz_h = (int  *)malloc((lsz + 1) * sizeof(int));
            Type *asz_h  = (Type *)malloc((lsz + 1) * sizeof(Type));
            Type *bsz_h  = (Type *)malloc((lsz + 1) * sizeof(Type));
            Type *yaco_h = (Type *)malloc(lmx * sizeof(Type));
            memcpy_cuda_d2h(yaco_h, grid_instance.yaco, lmx);

            int l;
            for(int ll = 0; ll <= lsz; ++ll)
            {
                l = static_cast<int>(tmp[ll + 2 * lmx]);
                lcsz_h[ll] = l;
                asz_h[ll]  = tmp[l] / yaco_h[l];
                bsz_h[ll]  = tmp[l + lmx] / yaco_h[l];
            }

            memcpy_cuda_h2d(lcsz, lcsz_h, lsz + 1);
            memcpy_cuda_h2d(asz , asz_h , lsz + 1);
            memcpy_cuda_h2d(bsz , bsz_h , lsz + 1);

            free(yaco_h);
            free(lcsz_h);
            free(asz_h);
            free(bsz_h);
        }

        free(tmp);
    }

    ~sponge()
    {
        if(lsz != -1)
        {
            free_cuda(lcsz);
            free_cuda(asz);
            free_cuda(bsz);
        }
    }

    void go(Type *qa, Type *de, unsigned int lmx)
    {
        std::string function_name  = "spongego";
        NVTX_RANGE(function_name.c_str());
        unsigned int blockSize = 256;
        unsigned int blockPerGrid = div_ceil(lsz + 1, blockSize);
        dim3 threadsPerBlock(blockSize, 1);
        dim3 blocksPerGrid(blockPerGrid);

        TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
            CANARD_KERNEL_NAME(spongego_kernel),
            qa, de, asz, bsz, lcsz, lsz, lmx);
    }

    int lsz;
    Type szco;
    Type *asz, *bsz;
    int *lcsz;
};

#endif
