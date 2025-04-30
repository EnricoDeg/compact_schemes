/*
 * @file check.hpp
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

#ifndef CANARD_MPI_CHECK_HPP
#define CANARD_MPI_CHECK_HPP

#include <mpi.h>
#include <iostream>

inline void check_mpi(int error_code)
{
    int rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    char error_string[MPI_MAX_ERROR_STRING];
    int length_of_error_string, error_class;

    if (error_code != MPI_SUCCESS) {
        MPI_Comm_rank(comm, &rank);
        MPI_Error_class(error_code, &error_class);
        MPI_Error_string(error_class, error_string, &length_of_error_string);
        std::cout << rank << ":MPI error: " << error_string << std::endl;
        MPI_Error_string(error_code, error_string, &length_of_error_string);
        std::cout << rank << ":MPI error: " << error_string << std::endl;
        MPI_Abort(comm, error_code);
    }
}

#endif
