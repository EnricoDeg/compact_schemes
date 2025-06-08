/*
 * @file numerics_rtc.hpp
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

#ifndef CANARD_KERNELS_NUMERICS_RTC_HPP
#define CANARD_KERNELS_NUMERICS_RTC_HPP

#include "cuda/kernels/numerics_device_rtc.hpp"

template<
unsigned int BlockSize,
unsigned int Axis,
typename Type>
CANARD_GLOBAL void deriv_kernel_1d(Type *infield, Type *outfield, Type * recv, Type * pbci,
                                   Type *drva,
                                   t_dcomp dcomp_info,
                                   unsigned int variable_id)
{
    CANARD_SHMEM Type sdata[5 * BlockSize];

    Type * sx, *srhs, *sa, *sb, *sc;

    sx   = sdata;
    srhs = sdata + BlockSize;
    sa   = sdata + 2 * BlockSize;
    sb   = sdata + 3 * BlockSize;
    sc   = sdata + 4 * BlockSize;

    deriv_kernel_1d_impl<Axis, BlockSize>(infield,
                                          outfield,
                                          recv,
                                          pbci,
                                          drva,
                                          dcomp_info,
                                          variable_id,
                                          sx,
                                          srhs,
                                          sa,
                                          sb,
                                          sc);
}

#endif
