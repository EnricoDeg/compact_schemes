/*
 * @file check_rtc.hpp
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
#ifndef CANARD_CUDA_CHECK_RTC_HPP
#define CANARD_CUDA_CHECK_RTC_HPP

#include <cuda.h>
#include <nvrtc.h>
#include <iostream>

inline void check_cuda_rtc(nvrtcResult error) {
    if ( error != NVRTC_SUCCESS ) {
        std::cout << "NVRTC error: " << nvrtcGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void check_cuda_driver(CUresult error) {
    if ( error != CUDA_SUCCESS ) {
        const char *msg;
        cuGetErrorName(error, &msg);
        std::cout << "CUDA driver error: " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif
