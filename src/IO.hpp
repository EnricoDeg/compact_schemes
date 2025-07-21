/*
 * @file IO.hpp
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

#ifndef CANARD_IO_HPP
#define CANARD_IO_HPP

#include <cinttypes>
#include <utility>
#include <iostream>
#include <fstream>
#include <vector>

#include "common/parameters.hpp"
#include "cuda/IO.hpp"
#include "mpi/check.hpp"

#include "domdcomp.hpp"
#include "grid.hpp"

float SwapEnd2(float& var)
{
    float tmp = var;
    char* varArray = reinterpret_cast<char*>(&tmp);
    for(long i = 0; i < static_cast<long>(sizeof(var)/2); i++)
    {
        std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
    }
    return tmp;
}

struct IOwriter
{
    IOwriter(int nvars_,
             const domdcomp& domdcomp_instance,
             std::vector<std::string>& variable_names_)
        : nvars(nvars_)
    {
        dtsum  = 0.0;
        size   = domdcomp_instance.lmx + 1;
        qb     = allocate_cuda<float>(nvars * size);
        d_vart = allocate_cuda<float>((nvars + NumberOfSpatialDims) * size);
        vart   = (float *)malloc((nvars + NumberOfSpatialDims) * size * sizeof(float));
        for(unsigned int i = 0; i < variable_names_.size(); ++i)
        {
            variable_names.push_back(variable_names_[i]);
        }
        int mpro;
        check_mpi( MPI_Comm_size(MPI_COMM_WORLD, &mpro) );
        mpro--;
        lpos = (int *)malloc((mpro + 1) * sizeof(int));

        int mp;
        for(unsigned int mm = 0; mm <= domdcomp_instance.nblocks; ++mm)
        {
            lpos[domdcomp_instance.mo[mm]] = 0;
            for(unsigned int i = 1; i <= domdcomp_instance.nbpc[0][mm] - 1; ++i)
            {
                mp       = domdcomp_instance.mo[mm] + i;
                lpos[mp] = lpos[mp-1] + domdcomp_instance.lxim[mp-1] + 1;
            }
            int jp = domdcomp_instance.nbpc[0][mm];
            for(unsigned int j = 1; j <= domdcomp_instance.nbpc[1][mm] - 1; ++j)
            {
                for(unsigned int i = 0; i <= domdcomp_instance.nbpc[0][mm] - 1; ++i)
                {
                    mp       = domdcomp_instance.mo[mm] + j * jp + i;
                    lpos[mp] = lpos[mp-jp] + ( domdcomp_instance.lximb[mm] + 1 ) *
                                             ( domdcomp_instance.letm[mp-jp] + 1 );
                }
            }
            int kp = domdcomp_instance.nbpc[0][mm] * domdcomp_instance.nbpc[1][mm];
            for(unsigned int k = 1; k <= domdcomp_instance.nbpc[2][mm] - 1; ++k)
            {
                for(unsigned int j = 0; j <= domdcomp_instance.nbpc[1][mm] - 1; ++j)
                {
                    for(unsigned int i = 0; i <= domdcomp_instance.nbpc[0][mm] - 1; ++i)
                    {
                        mp       = domdcomp_instance.mo[mm] + k * kp + j * jp + i;
                        lpos[mp] = lpos[mp-kp] + ( domdcomp_instance.lximb[mm] + 1 ) *
                                                 ( domdcomp_instance.letmb[mm] + 1 ) *
                                                 ( domdcomp_instance.lzem[mp-kp] + 1 );
                    }
                }
            }
        }
    }
    ~IOwriter()
    {
        free(lpos);
        free(vart);
        free_cuda(qb);
        free_cuda(d_vart);
    }

    void SwapEnd(float& var)
    {
        char* varArray = reinterpret_cast<char*>(&var);
        for(long i = 0; i < static_cast<long>(sizeof(var)/2); i++)
        {
            std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
        }
    }

    void fill_buffer(float *qo,
                     float *qa,
                     t_patch<float> *patch,
                     const float& dt,
                     t_point<float> umf,
                     bool is_output_step)
    {
        dtsum += dt;
        float fctr = 0.5 * dt;
        fill_IO_buffer(d_vart,
                       qb,
                       qo,
                       qa,
                       patch,
                       fctr,
                       dtsum,
                       umf,
                       is_output_step,
                       size,
                       file_number);
    }

    void reset_buffer()
    {
        reset_IO_buffer(qb, size);
    }

    void go_vtk(const domdcomp& domdcomp_instance,
                float *data)
    {
        std::ofstream vtkstream;
        std::string filename = "output_data" + std::to_string(file_number) + ".vtk";
        vtkstream.open(filename, std::ios::out | std::ios::app | std::ios::binary);
        if (vtkstream) {
            int block_points = (domdcomp_instance.lxio + 1) *
                (domdcomp_instance.leto + 1) *
                (domdcomp_instance.lzeo + 1);

            // header
            vtkstream << "# vtk DataFile Version 2.0" << "\n";
            vtkstream << "Output compact schemes" << "\n";
            vtkstream << "BINARY" << "\n";

            // grid
            vtkstream << "DATASET STRUCTURED_GRID" << std::endl;
            vtkstream << "DIMENSIONS " << domdcomp_instance.lxio + 1 << " "
                << domdcomp_instance.leto + 1 << " "
                << domdcomp_instance.lzeo + 1 << std::endl;
            vtkstream << "POINTS " << block_points << " float" << std::endl;
            for (unsigned int i = 0; i < block_points; ++i) {
                SwapEnd(data[i]);
                vtkstream.write((char*)&data[i], sizeof(float));
                SwapEnd(data[i + block_points]);
                vtkstream.write((char*)&data[i + block_points], sizeof(float));
                SwapEnd(data[i + 2 * block_points]);
                vtkstream.write((char*)&data[i + 2 * block_points], sizeof(float));
            }

            // data
            vtkstream << "POINT_DATA " << block_points << std::endl;
            int nvars_total = nvars + NumberOfSpatialDims;
            for(unsigned int var = NumberOfSpatialDims; var < nvars_total; ++var)
            {
                vtkstream << "SCALARS " << variable_names[var] << " float 1\n";
                vtkstream << "LOOKUP_TABLE default\n";
                for(unsigned int i = 0; i < block_points; ++i)
                {
                    SwapEnd(data[i + var * block_points]);
                    vtkstream.write((char*)&data[i + var * block_points], sizeof(float));
                }
            }

            // close file
            vtkstream.close();
        } else {
            std::cout << "ERROR opening vtk file" << std::endl;
            exit(1);
        }
        file_number++;
    }

    void go(const domdcomp& domdcomp_instance,
            const grid<float>& grid_instance,
            bool is_output_step)
    {
        if(!is_output_step)
        {
            return;
        }
        int nvars_total = nvars + NumberOfSpatialDims;
        memcpy_cuda_d2h(vart, d_vart, nvars_total * size);

        int myid;
        check_mpi( MPI_Comm_rank(MPI_COMM_WORLD, &myid) );

        int ltomb = (domdcomp_instance.lxio + 1) *
                    (domdcomp_instance.leto + 1) *
                    (domdcomp_instance.lzeo + 1);

        int llmb = nvars_total * ltomb - 1;
        float *vara = (float *)malloc((llmb + 1) * sizeof(float));
        float *varb = (float *)malloc((llmb + 1) * sizeof(float));
        int lje = -1;
        int ljs = lje + 1;
        lje = ljs + nvars_total * (domdcomp_instance.lmx + 1) - 1;
        if ( myid == domdcomp_instance.mo[domdcomp_instance.mb] )
        {
            int mps = domdcomp_instance.mo[domdcomp_instance.mb];
            int mpe = mps + domdcomp_instance.nbpc[0][domdcomp_instance.mb] *
                        domdcomp_instance.nbpc[1][domdcomp_instance.mb] *
                        domdcomp_instance.nbpc[2][domdcomp_instance.mb] - 1;
            int lis = 0;
            int lie = nvars_total * ( domdcomp_instance.lmx + 1 ) - 1;
            for(unsigned int i = 0; i <= lie-lis; ++i)
            {
                vara[lis + i] = vart[ljs + i];
            }

            for(unsigned int mp = mps + 1; mp <= mpe; ++mp)
            {
                lis = lie + 1;
                lie = lis + nvars_total * ( domdcomp_instance.lxim[mp] + 1 ) *
                                          ( domdcomp_instance.letm[mp] + 1 ) *
                                          ( domdcomp_instance.lzem[mp] + 1 ) - 1;
                int lmpi = lie - lis + 1;
                int itag = 1;
                check_mpi(MPI_Recv(vara + lis, lmpi, MPI_FLOAT,
                    mp, itag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
            lis = 0;
            for(unsigned int mp = mps; mp <= mpe; ++mp)
            {
                for(unsigned int m = 0; m < nvars_total; ++m)
                {
                    for(unsigned int k = 0; k <= domdcomp_instance.lzem[mp]; ++k)
                    {
                        for(unsigned int j = 0; j <= domdcomp_instance.letm[mp]; ++j)
                        {
                            ljs = lpos[mp] + m * ltomb + k * ( domdcomp_instance.leto + 1 ) *
                                                                     ( domdcomp_instance.lxio + 1 ) + j *
                                                                     ( domdcomp_instance.lxio + 1 );
                            for(unsigned int i = 0; i <= domdcomp_instance.lxim[mp]; ++i)
                            {
                                varb[ljs + i] = vara[lis + i];
                            }
                            lis = lis + domdcomp_instance.lxim[mp] + 1;
                        }
                    }
                }
            }
            free(vara);

            // write_output_file_mb
            if(domdcomp_instance.mb == 0)
            {
                go_vtk(domdcomp_instance, varb);
            }
        }
        else
        {
            int lmpi = lje - ljs + 1;
            int itag = 1;
            check_mpi(MPI_Send(vart + ljs, lmpi, MPI_FLOAT,
                domdcomp_instance.mo[domdcomp_instance.mb], itag, MPI_COMM_WORLD));
            free(vara);
        }
        dtsum = 0.0;
        free(varb);
        reset_buffer();
    }
    int nvars;
    unsigned int size;
    int *lpos;
    int file_number = 0;
    float dtsum = 0.0;
    float *qb;
    float *d_vart;
    float *vart;
    std::vector<std::string> variable_names;
};

void helper_vtk(const domdcomp& domdcomp_instance,
                grid<float>& grid_instance,
                int noutput,
                std::string filename,
                float *data)
{
    std::ofstream vtkstream;
    vtkstream.open(filename, std::ios::out | std::ios::app | std::ios::binary);
    if (vtkstream) {
        int block_points = (domdcomp_instance.lxio + 1) *
            (domdcomp_instance.leto + 1) *
            (domdcomp_instance.lzeo + 1);

        // header
        vtkstream << "# vtk DataFile Version 2.0" << "\n";
        vtkstream << "Output compact schemes" << "\n";
        vtkstream << "BINARY" << "\n";

        // grid
        vtkstream << "DATASET STRUCTURED_GRID" << std::endl;
        vtkstream << "DIMENSIONS " << domdcomp_instance.lxio + 1 << " "
            << domdcomp_instance.leto + 1 << " "
            << domdcomp_instance.lzeo + 1 << std::endl;
        vtkstream << "POINTS " << block_points << " float" << std::endl;
        float tmp;
        for (unsigned int i = 0; i < block_points; ++i) {
            tmp = SwapEnd2(grid_instance.patch[0][i]);
            vtkstream.write((char*)&tmp, sizeof(float));
            tmp = SwapEnd2(grid_instance.patch[1][i]);
            vtkstream.write((char*)&tmp, sizeof(float));
            tmp = SwapEnd2(grid_instance.patch[2][i]);
            vtkstream.write((char*)&tmp, sizeof(float));
        }

        // data
        std::cout << "block points = " << block_points << std::endl;
        vtkstream << "POINT_DATA " << block_points << std::endl;
        for(unsigned int var = 0; var < noutput; ++var)
        {
            vtkstream << "SCALARS test float 1\n";
            vtkstream << "LOOKUP_TABLE default\n";
            for(unsigned int i = 0; i < block_points; ++i)
            {
                tmp = SwapEnd2(data[i + var * block_points]);
                vtkstream.write((char*)&tmp, sizeof(float));
            }
        }

        // close file
        vtkstream.close();
    } else {
        std::cout << "ERROR opening vtk file" << std::endl;
        exit(1);
    }
}

#endif
