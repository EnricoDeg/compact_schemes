/*
 * @file templateShMem.hpp
 *
 * @copyright Copyright (C) 2024 Enrico Degregori <enrico.degregori@gmail.com>
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

#ifndef TEMPLATESHMEM_HPP
#define TEMPLATESHMEM_HPP

#include "cuda/kernels/definitions.hpp"

/** @brief Wrapper class for templatized dynamic shared memory arrays.
  * 
  * This struct uses template specialization on the type \a T to declare
  * a differently named dynamic shared memory array for each type
  * (\code extern __shared__ T s_type[] \endcode).
  * 
  * Currently there are specializations for the following types:
  * \c int, \c uint, \c char, \c uchar, \c short, \c ushort, \c long, 
  * \c unsigned long, \c bool, \c float, and \c double. One can also specialize it
  * for user defined types.
  */
template <typename T>
struct SharedMemory
{
    CANARD_DEVICE T* getPointer() {
        // Ensure that we won't compile any un-specialized types
        extern CANARD_DEVICE void Error_UnsupportedType();
        Error_UnsupportedType();
        return (T*)0;
    }
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double

template <>
struct SharedMemory <int> {
    CANARD_DEVICE int* getPointer() {
        extern CANARD_SHMEM int s_int[];
        return s_int;
    }
};

template <>
struct SharedMemory <unsigned int> {
    CANARD_DEVICE unsigned int* getPointer() {
        extern CANARD_SHMEM unsigned int s_uint[];
        return s_uint;
    }
};

template <>
struct SharedMemory <char> {
    CANARD_DEVICE char* getPointer() {
        extern CANARD_SHMEM char s_char[];
        return s_char;
    }
};

template <>
struct SharedMemory <unsigned char> {
    CANARD_DEVICE unsigned char* getPointer() {
        extern CANARD_SHMEM unsigned char s_uchar[];
        return s_uchar;
    }
};

template <>
struct SharedMemory <short> {
    CANARD_DEVICE short* getPointer() {
        extern CANARD_SHMEM short s_short[];
        return s_short;
    }
};

template <>
struct SharedMemory <unsigned short> {
    CANARD_DEVICE unsigned short* getPointer() {
        extern CANARD_SHMEM unsigned short s_ushort[];
        return s_ushort;
    }
};

template <>
struct SharedMemory <long> {
    CANARD_DEVICE long* getPointer() {
        extern CANARD_SHMEM long s_long[];
        return s_long;
    }
};

template <>
struct SharedMemory <unsigned long> {
    CANARD_DEVICE unsigned long* getPointer() {
        extern CANARD_SHMEM unsigned long s_ulong[];
        return s_ulong;
    }
};

template <>
struct SharedMemory <long long> {
    CANARD_DEVICE long long* getPointer() {
        extern CANARD_SHMEM long long s_longlong[];
        return s_longlong;
    }
};

template <>
struct SharedMemory <unsigned long long> {
    CANARD_DEVICE unsigned long long* getPointer() {
        extern CANARD_SHMEM unsigned long long s_ulonglong[];
        return s_ulonglong;
    }
};

template <>
struct SharedMemory <bool> {
    CANARD_DEVICE bool* getPointer() {
        extern CANARD_SHMEM bool s_bool[];
        return s_bool;
    }
};

template <>
struct SharedMemory <float> {
    CANARD_DEVICE float* getPointer() {
        extern CANARD_SHMEM float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double> {
    CANARD_DEVICE double* getPointer() {
        extern CANARD_SHMEM double s_double[];
        return s_double;
    }
};

template <>
struct SharedMemory <uchar4> {
    CANARD_DEVICE uchar4* getPointer() {
        extern CANARD_SHMEM uchar4 s_uchar4[];
        return s_uchar4;
    }
};

#endif