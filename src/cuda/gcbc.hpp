/*
 * @file gcbc.hpp
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

#ifndef CANARD_CUDA_GCBC_HPP
#define CANARD_CUDA_GCBC_HPP

#include "cuda/kernels/gcbc.hpp"
#include "cuda/check.hpp"
#include "cuda/dispatch.hpp"

template<typename Type>
struct gcbc_dispatch
{
    gcbc_dispatch(Type * cm_, Type * drva_, Type * drvb_,
         Type * qa_, Type * de_, Type * pressure_,
         Type * yaco_, Type * sbcc_,
         t_point<Type> umf_, t_point<Type> dudtmf_,
         t_dcomp dcomp_info_)
        : cm{cm_},
          drva{drva_},
          drvb{drvb_},
          qa{qa_},
          de{de_},
          pressure{pressure_},
          yaco{yaco_},
          sbcc{sbcc_},
          umf{umf_},
          dudtmf{dudtmf_},
          dcomp_info{dcomp_info_}
    {
    }

    void reset_buffer_pointer(Type * drva_, Type * cm_)
    {
        drva = drva_;
        cm = cm_;
    }

    void reset_buffer_pointer(Type * drva_, Type * drvb_, Type * cm_)
    {
        drva = drva_;
        drvb = drvb_;
        cm = cm_;
    }

    template<unsigned int Axis>
    void setup(unsigned int face_id, unsigned int face_offset, unsigned int flag)
    {
        unsigned int blockSize;
        unsigned int blockPerGrid;
        if(Axis == 0)
        {
            blockSize = dcomp_info.let;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 1)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 2)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.let;
        }
        dim3 threadsPerBlock(blockSize);
        dim3 blocksPerGrid(blockPerGrid);
        TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
            CANARD_KERNEL_NAME(gcbc_setup_kernel<Axis>),
            cm, drva, qa, de, pressure, yaco, umf, face_id, face_offset, flag,
            dcomp_info);
    }

    template<unsigned int Axis>
    void update_non_reflective(unsigned int face_id,
                               unsigned int face_offset)
    {
        unsigned int blockSize;
        unsigned int blockPerGrid;
        if(Axis == 0)
        {
            blockSize = dcomp_info.let;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 1)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 2)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.let;
        }
        dim3 threadsPerBlock(blockSize);
        dim3 blocksPerGrid(blockPerGrid);
        TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
            CANARD_KERNEL_NAME(gcbc_update_non_reflective_kernel<Axis>),
            cm, drva, qa, de, pressure, sbcc, umf, face_id, face_offset, dcomp_info);
    }

    template<unsigned int Axis>
    void update_wall(unsigned int face_id,
                     unsigned int face_offset,
                     Type dtwi)
    {
        unsigned int blockSize;
        unsigned int blockPerGrid;
        if(Axis == 0)
        {
            blockSize = dcomp_info.let;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 1)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 2)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.let;
        }
        dim3 threadsPerBlock(blockSize);
        dim3 blocksPerGrid(blockPerGrid);
        TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
            CANARD_KERNEL_NAME(gcbc_update_wall_kernel<Axis>),
            cm, drva, qa, de, pressure, sbcc, umf, dudtmf,
            face_id, face_offset, dtwi, dcomp_info);
    }

    template<unsigned int Axis>
    void update_inter_curv(unsigned int face_id,
                           unsigned int face_offset)
    {
        unsigned int blockSize;
        unsigned int blockPerGrid;
        if(Axis == 0)
        {
            blockSize = dcomp_info.let;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 1)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 2)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.let;
        }
        dim3 threadsPerBlock(blockSize);
        dim3 blocksPerGrid(blockPerGrid);
        TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
            CANARD_KERNEL_NAME(gcbc_update_inter_curv_kernel<Axis>),
            cm, drva, drvb, qa, de, pressure, sbcc, umf, dudtmf,
            face_id, face_offset, dcomp_info);
    }

    Type * cm;
    Type * drva;
    Type * drvb;
    Type * qa;
    Type * de;
    Type * pressure;
    Type * yaco;
    Type * sbcc;
    t_point<Type> umf;
    t_point<Type> dudtmf;
    t_dcomp dcomp_info;
};

template<typename Type, typename TypeIndex>
struct wall_bc_dispatch
{
    wall_bc_dispatch(Type * qa_, TypeIndex * npex_, t_point<Type> umf_, t_dcomp dcomp_info_)
        : qa{qa_},
          npex{npex_},
          umf{umf_},
          dcomp_info{dcomp_info_}
    {
    }

    template<unsigned int Axis>
    void apply_inviscid(unsigned int face_offset)
    {
        unsigned int blockSize;
        unsigned int blockPerGrid;
        if(Axis == 0)
        {
            blockSize = dcomp_info.let;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 1)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 2)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.let;
        }
        dim3 threadsPerBlock(blockSize);
        dim3 blocksPerGrid(blockPerGrid);
        TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
            CANARD_KERNEL_NAME(wall_inviscid_kernel<Axis>),
            qa, npex, face_offset, dcomp_info);
    }

    template<unsigned int Axis>
    void apply_viscous(unsigned int face_offset)
    {
        unsigned int blockSize;
        unsigned int blockPerGrid;
        if(Axis == 0)
        {
            blockSize = dcomp_info.let;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 1)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.lze;
        }
        else if(Axis == 2)
        {
            blockSize = dcomp_info.lxi;
            blockPerGrid = dcomp_info.let;
        }
        dim3 threadsPerBlock(blockSize);
        dim3 blocksPerGrid(blockPerGrid);
        TIME(blocksPerGrid, threadsPerBlock, 0, 0, false,
            CANARD_KERNEL_NAME(wall_viscous_kernel<Axis>),
            qa, umf, npex, face_offset, dcomp_info);
    }

    Type * qa;
    TypeIndex * npex;
    t_point<Type> umf;
    t_dcomp dcomp_info;
};

#endif
