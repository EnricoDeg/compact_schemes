/*
 * @file exchange.hpp
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

#ifndef CANARD_MPI_EXCHANGE_HPP
#define CANARD_MPI_EXCHANGE_HPP

#include "mpi/check.hpp"
#include "mpi/data_types.hpp"

template<typename Type>
struct exchange
{
    exchange()
    {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        requests = (MPI_Request *)malloc(world_size * sizeof(MPI_Request));
        number_of_requests = 0;
        mpi_type = mpi_get_type<Type>();
    }

    void reset_buffer_pointer(Type * send_buffer_, Type * recv_buffer_)
    {
        send_buffer = send_buffer_;
        recv_buffer = recv_buffer_;
    }

    void trigger(int count, int offset, int pair_process, int tag_send, int tag_recv)
    {
        check_mpi(MPI_Isend(send_buffer + offset,
                            count,
                            mpi_type,
                            pair_process,
                            tag_send,
                            MPI_COMM_WORLD,
                            &requests[number_of_requests]));
        number_of_requests++;
        check_mpi(MPI_Irecv(recv_buffer + offset,
                            count,
                            mpi_type,
                            pair_process,
                            tag_recv,
                            MPI_COMM_WORLD,
                            &requests[number_of_requests]));
        number_of_requests++;
    }

    void reset()
    {
        if (number_of_requests != 0)
        {
            check_mpi(MPI_Waitall(number_of_requests, requests, MPI_STATUSES_IGNORE));
        }
    }

    ~exchange()
    {
        free(requests);
    }

    Type * send_buffer;
    Type * recv_buffer;
    MPI_Request *requests;
    int number_of_requests;
    MPI_Datatype mpi_type;
};

#endif
