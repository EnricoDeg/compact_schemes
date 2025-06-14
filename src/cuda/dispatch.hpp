/*
 * @file dispatch.hpp
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

#ifndef CANARD_DISPATCH_HPP
#define CANARD_DISPATCH_HPP

#include "cuda/check.hpp"

#define TIME(blocksPerGrid, threadsPerBlock, shmem, stream, async, func, args ...)               \
  do {                                                                                           \
    func<<< blocksPerGrid, threadsPerBlock, shmem, stream >>>(args);                             \
    if (!async) {                                                                                \
        check_cuda( cudaPeekAtLastError() ) ;                                                    \
        check_cuda( cudaStreamSynchronize(stream) );                                             \
    }                                                                                            \
  } while(0)

#define TIME_RTC(blocksPerGridX, blocksPerGridY, blocksPerGridZ,              \
                 threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ,        \
                 shmem, stream, async, func, args)                            \
  do {                                                                        \
    check_cuda_driver( cuLaunchKernel(func,                                   \
                        blocksPerGridX, blocksPerGridY, blocksPerGridZ,       \
                        threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, \
                        shmem, stream,                                        \
                        args, 0) );                                           \
    if (!async) {                                                             \
        check_cuda_driver( cuStreamSynchronize ( stream ) );                  \
    }                                                                         \
  } while(0)

#define CANARD_KERNEL_NAME(...) __VA_ARGS__

#endif
