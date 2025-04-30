/*
 * @file definitions.hpp
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

#ifndef CANARD_DEFINITIONS_HPP
#define CANARD_DEFINITIONS_HPP

#include <cuda.h>

#define FULL_WARP_MASK 0xffffffff

#define CANARD_GLOBAL __global__
#define CANARD_DEVICE __device__
#define CANARD_RESTRICT __restrict__
#define CANARD_HOST __host__
#define CANARD_HOST_DEVICE __host__ __device__
#define CANARD_SHMEM __shared__
#define CANARD_FORCE_INLINE __forceinline__
#define CANARD_LAUNCH_BOUNDS(N) __launch_bounds__(N)

#define CANARD_WARPSIZE 32

#define CANARD_UNROLL _Pragma("unroll")
#define CANARD_NO_UNROLL _Pragma("nounroll")

#endif
