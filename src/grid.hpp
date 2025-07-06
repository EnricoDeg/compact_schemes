/*
 * @file grid.hpp
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

#ifndef CANARD_GRID_HPP
#define CANARD_GRID_HPP

#include <cassert>

#include "yaml-cpp/yaml.h"

#include "common/data_types.hpp"
#include "common/parameters.hpp"

#include "mpi/check.hpp"

#include "cuda/common.hpp"

#include "domdcomp.hpp"

int indx3(int i, int j, int k, int nn, int lxi, int let) {
    assert(nn < 3);
    if(nn == 0)
    {
        return (k*(let+1)+j)*(lxi+1)+i;
    }
    else if(nn == 1)
    {
        return (j*(let+1)+i)*(lxi+1)+k;
    }
    else
    {
        return (i*(let+1)+k)*(lxi+1)+j;
    }
}

template<typename Type>
struct grid
{
    grid(const domdcomp& domdcomp_instance)
    {
        size_t nelements = domdcomp_instance.lmx;

        xim = allocate_cuda<Type>(NumberOfSpatialDims * nelements);
        etm = allocate_cuda<Type>(NumberOfSpatialDims * nelements);
        zem = allocate_cuda<Type>(NumberOfSpatialDims * nelements);
    }

    void read_config(YAML::Node& grid_yaml)
    {
        YAML::Node doml0_node = grid_yaml[0]["doml0"];
        doml0 = doml0_node.as<float>();

        YAML::Node doml1_node = grid_yaml[1]["doml1"];
        doml1 = doml1_node.as<float>();

        YAML::Node domh_node = grid_yaml[2]["domh"];
        domh = domh_node.as<float>();

        YAML::Node span_node = grid_yaml[3]["span"];
        span = span_node.as<float>();
    }

    void generate(const domdcomp& domdcomp_instance)
    {
        patch[0] = (Type *)malloc((domdcomp_instance.lmx + 1) * sizeof(Type));
        patch[1] = (Type *)malloc((domdcomp_instance.lmx + 1) * sizeof(Type));
        patch[2] = (Type *)malloc((domdcomp_instance.lmx + 1) * sizeof(Type));

        int myid;
        check_mpi( MPI_Comm_rank(MPI_COMM_WORLD, &myid) );
        // master process in block generate full block grid and then send the partition
        // to the processes in the same block (and to itself)
        if (myid == domdcomp_instance.mo[domdcomp_instance.mb]) {
            int lxi0 = domdcomp_instance.lximb[domdcomp_instance.mb];
            int let0 = domdcomp_instance.letmb[domdcomp_instance.mb];
            int lze0 = domdcomp_instance.lzemb[domdcomp_instance.mb];
            Type *grid_points_global[NumberOfSpatialDims];
            int np = (lxi0 + 1) * (let0 + 1) * (lze0 + 1) - 1;
            grid_points_global[0] = (Type *)malloc((np+1)*sizeof(Type));
            grid_points_global[1] = (Type *)malloc((np+1)*sizeof(Type));
            grid_points_global[2] = (Type *)malloc((np+1)*sizeof(Type));

            // generate full block grid
            for (int k = 0; k <= lze0; ++k) {
                for (int j = 0; j <= let0; ++j) {
                    for (int i = 0; i <= lxi0; ++i) {
                        Type ra0 = (doml1 + doml0) / lxi0;
                        Type ra1 = 2.0 * domh / let0;
                        grid_points_global[0][i+j*(lxi0+1)+k*(lxi0+1)*(let0+1)] = -doml0 + i * ra0;
                        grid_points_global[1][i+j*(lxi0+1)+k*(lxi0+1)*(let0+1)] = -domh  + j * ra1;
                        grid_points_global[2][i+j*(lxi0+1)+k*(lxi0+1)*(let0+1)] = span * ((Type)(lze0 - k) / lze0 - 0.5);
                    }
                }
            }

            // send partition
            int id_end;
            if (domdcomp_instance.mb == domdcomp_instance.nblocks) {
                int mpro;
                check_mpi( MPI_Comm_size(MPI_COMM_WORLD, &mpro) );
                id_end = mpro - 1;
            } else {
                id_end = domdcomp_instance.mo[domdcomp_instance.mb + 1];
            }

            for (int id = myid; id <= id_end; ++id) {

                int lmx_id = ( domdcomp_instance.lxim[id] + 1 ) *
                            ( domdcomp_instance.letm[id] + 1 ) *
                            ( domdcomp_instance.lzem[id] + 1 ) - 1;
                int *lio = (int *)malloc((domdcomp_instance.letm[id] + 1) *
                                        (domdcomp_instance.lzem[id] + 1) * sizeof(int));

                // compute lio for each receiving process
                for (int k = 0; k <= domdcomp_instance.lzem[id]; k++) {
                    int kp = k * (domdcomp_instance.leto + 1) * (domdcomp_instance.lxio + 1);
                    for (int j = 0; j <= domdcomp_instance.letm[id]; j++) {
                        int jp = j * (domdcomp_instance.lxio+1);
                        lio[j + k * (domdcomp_instance.letm[id] + 1)] = jp + kp;
                    }
                }

                // fill buffer
                Type *buffer_x = (Type *)malloc((lmx_id + 1) * sizeof(Type));
                Type *buffer_y = (Type *)malloc((lmx_id + 1) * sizeof(Type));
                Type *buffer_z = (Type *)malloc((lmx_id + 1) * sizeof(Type));
                
                int lp = domdcomp_instance.lpos[id];
                for (int k = 0; k <= domdcomp_instance.lzem[id]; ++k) {
                    for (int j = 0; j <= domdcomp_instance.letm[id]; ++j) {
                        int lq = lp + lio[j + k * (domdcomp_instance.letm[id] + 1)];
                        for (int i = 0; i <= domdcomp_instance.lxim[id]; ++i) {
                            int l = indx3(i, j, k, 0,
                                domdcomp_instance.lxim[id], domdcomp_instance.letm[id]);
                            buffer_x[l] = grid_points_global[0][lq+i];
                            buffer_y[l] = grid_points_global[1][lq+i];
                            buffer_z[l] = grid_points_global[2][lq+i];
                        }
                    }
                }

                // send each coordinate
                if (id == myid) {
                    for (int k = 0; k <= domdcomp_instance.lzem[id]; ++k) {
                        for (int j = 0; j <= domdcomp_instance.letm[id]; ++j) {
                            for (int i = 0; i <= domdcomp_instance.lxim[id]; ++i) {
                                int l = indx3(i, j, k, 0,
                                    domdcomp_instance.lxim[id], domdcomp_instance.letm[id]);
                                patch[0][l] = buffer_x[l];
                                patch[1][l] = buffer_y[l];
                                patch[2][l] = buffer_z[l];
                            }
                        }
                    }
                } else {
                    check_mpi(MPI_Send(buffer_x, lmx_id+1, MPI_FLOAT, id, id, MPI_COMM_WORLD));
                    check_mpi(MPI_Send(buffer_y, lmx_id+1, MPI_FLOAT, id, id, MPI_COMM_WORLD));
                    check_mpi(MPI_Send(buffer_z, lmx_id+1, MPI_FLOAT, id, id, MPI_COMM_WORLD));
                }

                free(buffer_x);
                free(buffer_y);
                free(buffer_z);
                free(lio);
            }

            free(grid_points_global[0]);
            free(grid_points_global[1]);
            free(grid_points_global[2]);
        } else {
            // recv partition
            check_mpi( MPI_Recv(patch[0], domdcomp_instance.lmx+1, 
                            MPI_FLOAT, domdcomp_instance.mo[domdcomp_instance.mb],
                            myid, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE) );

            check_mpi( MPI_Recv(patch[1], domdcomp_instance.lmx+1, 
                            MPI_FLOAT, domdcomp_instance.mo[domdcomp_instance.mb],
                            myid, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE) );

            check_mpi( MPI_Recv(patch[2], domdcomp_instance.lmx+1, 
                            MPI_FLOAT, domdcomp_instance.mo[domdcomp_instance.mb],
                            myid, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE) );
        }
    }

    ~grid()
    {
        free_cuda(xim);
        free_cuda(etm);
        free_cuda(zem);
    }

    Type span;
    Type doml0;
    Type doml1;
    Type domh;
    Type *patch[NumberOfSpatialDims];
    Type *xim;
    Type *etm;
    Type *zem;
};

#endif
