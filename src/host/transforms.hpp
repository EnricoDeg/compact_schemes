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

#ifndef CANARD_HOST_TRANSFORMS_HPP
#define CANARD_HOST_TRANSFORMS_HPP

#include <cstdlib>

#include "common/data_types.hpp"

namespace host{

template<unsigned int Axis>
inline int get_dimension(t_dcomp dcomp_info)
{
    if constexpr(Axis == 0)
    {
        return dcomp_info.lxi;
    }
    else if constexpr(Axis == 1)
    {
        return dcomp_info.let;
    }
    else if constexpr(Axis == 2)
    {
        return dcomp_info.lze;
    }
}

template<unsigned int Axis, unsigned int Stencil>
size_t get_face_size(t_dcomp dcomp_info)
{
    if constexpr(Axis == 0)
    {
        return Stencil * dcomp_info.let * dcomp_info.lze;
    }
    else if constexpr(Axis == 1)
    {
        return Stencil * dcomp_info.lxi * dcomp_info.lze;
    }
    else if constexpr(Axis == 2)
    {
        return Stencil * dcomp_info.lxi * dcomp_info.let;
    }
}

}

#endif
