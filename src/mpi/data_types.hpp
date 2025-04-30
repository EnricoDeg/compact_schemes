/*
 * @file data_types.hpp
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

#ifndef CANARD_MPI_DATA_TYPES_HPP
#define CANARD_MPI_DATA_TYPES_HPP

#include <mpi.h>

template <typename T>
constexpr MPI_Datatype mpi_get_type() noexcept {
    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
      
    if constexpr (std::is_same_v<T, char>) {
        mpi_type = MPI_CHAR;
    } else if constexpr (std::is_same_v<T, signed char>) {
        mpi_type = MPI_SIGNED_CHAR;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        mpi_type = MPI_UNSIGNED_CHAR;
    } else if constexpr (std::is_same_v<T, wchar_t>) {
        mpi_type = MPI_WCHAR;
    } else if constexpr (std::is_same_v<T, signed short>) {
        mpi_type = MPI_SHORT;
    } else if constexpr (std::is_same_v<T, unsigned short>) {
        mpi_type = MPI_UNSIGNED_SHORT;
    } else if constexpr (std::is_same_v<T, signed int>) {
        mpi_type = MPI_INT;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        mpi_type = MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<T, signed long int>) {
        mpi_type = MPI_LONG;
    } else if constexpr (std::is_same_v<T, unsigned long int>) {
        mpi_type = MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<T, signed long long int>) {
        mpi_type = MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<T, unsigned long long int>) {
        mpi_type = MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same_v<T, float>) {
        mpi_type = MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        mpi_type = MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, long double>) {
        mpi_type = MPI_LONG_DOUBLE;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        mpi_type = MPI_INT8_T;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
        mpi_type = MPI_INT16_T;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        mpi_type = MPI_INT32_T;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        mpi_type = MPI_INT64_T;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        mpi_type = MPI_UINT8_T;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        mpi_type = MPI_UINT16_T;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        mpi_type = MPI_UINT32_T;
    } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        mpi_type = MPI_UINT64_T;
    } else if constexpr (std::is_same_v<T, bool>) {
        mpi_type = MPI_C_BOOL;
    }   else if constexpr (std::is_same_v<T, std::complex<float>>) {
        mpi_type = MPI_C_COMPLEX;
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        mpi_type = MPI_C_DOUBLE_COMPLEX;
    } else if constexpr (std::is_same_v<T, std::complex<long double>>) {
        mpi_type = MPI_C_LONG_DOUBLE_COMPLEX;
    }
    assert(mpi_type != MPI_DATATYPE_NULL);
    return mpi_type;
}

#endif
