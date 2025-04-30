/*
 * @file transforms.hpp
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

#ifndef CANARD_TRANSFORMS_HPP
#define CANARD_TRANSFORMS_HPP

#include "common/data_types.hpp"

#include "cuda/kernels/definitions.hpp"

CANARD_DEVICE unsigned int get_thread_local_idx()
{
    return threadIdx.x;
}

CANARD_DEVICE unsigned int get_thread_global_idx()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

CANARD_HOST_DEVICE CANARD_FORCE_INLINE unsigned int get_index(unsigned int block_stridex,
                                                              unsigned int block_stridey,
                                                              unsigned int thread_stridex)
{
    return blockIdx.y * block_stridey + blockIdx.x * block_stridex + threadIdx.x * thread_stridex;
}

template<unsigned int Axis>
CANARD_HOST_DEVICE CANARD_FORCE_INLINE unsigned int get_block_stridex(t_dcomp dcomp_info)
{
    static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");
    if constexpr(Axis == 0)
    {
        return dcomp_info.lxi;
    }
    else if constexpr(Axis == 1)
    {
        return 1;
    }
    else if constexpr(Axis == 2)
    {
        return 1;
    }
}

template<unsigned int Axis>
CANARD_HOST_DEVICE CANARD_FORCE_INLINE unsigned int get_block_stridey(t_dcomp dcomp_info)
{
    static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");
    if constexpr(Axis == 0)
    {
        return dcomp_info.let;
    }
    else if constexpr(Axis == 1)
    {
        return dcomp_info.let * dcomp_info.lxi;
    }
    else if constexpr(Axis == 2)
    {
        return dcomp_info.lxi;
    }
}

template<unsigned int Axis>
CANARD_HOST_DEVICE CANARD_FORCE_INLINE unsigned int get_thread_stride(t_dcomp dcomp_info)
{
    static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");
    if constexpr(Axis == 0)
    {
        return 1;
    }
    else if constexpr(Axis == 1)
    {
        return dcomp_info.lxi;
    }
    else if constexpr(Axis == 2)
    {
        return dcomp_info.lxi * dcomp_info.let;
    }
}

template<unsigned int Axis>
CANARD_HOST_DEVICE CANARD_FORCE_INLINE unsigned int get_face_stride(t_dcomp dcomp_info)
{
    static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");
    if constexpr(Axis == 0)
    {
        return dcomp_info.let;
    }
    else if constexpr(Axis == 1)
    {
        return dcomp_info.lxi;
    }
    else if constexpr(Axis == 2)
    {
        return dcomp_info.lxi;
    }
}

template<unsigned int Axis>
CANARD_HOST_DEVICE CANARD_FORCE_INLINE unsigned int get_face_size(t_dcomp dcomp_info)
{
    static_assert(Axis >= 0 && Axis < 3, "Axis index must be 0, 1, or 2");
    if constexpr(Axis == 0)
    {
        return dcomp_info.let * dcomp_info.lze;
    }
    else if constexpr(Axis == 1)
    {
        return dcomp_info.lxi * dcomp_info.lze;
    }
    else if constexpr(Axis == 2)
    {
        return dcomp_info.lxi * dcomp_info.let;
    }
}

#endif
