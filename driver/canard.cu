#include <stdio.h>
#include <cuda.h>
#include <mpi.h>

#include "numerics.hpp"

// Main program
int main()
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    static constexpr unsigned int ax = 0;
    unsigned int variable_id = 0;

    // Subdomain info
    t_dcomp dcomp_info;
    dcomp_info.lxi = 128;
    dcomp_info.let = 256;
    dcomp_info.lze = 64;
    dcomp_info.lmx = dcomp_info.lxi * dcomp_info.let * dcomp_info.lze;

    // Spatial distance
    float h = 1.0;
    float h_1 = 1.0 / h;

    // Number of bytes to allocate for fields
    size_t bytes = dcomp_info.lmx*sizeof(float);

    // Allocate memory for input and output fields on host
    float *infield = (float*)malloc(bytes);
    float *outfield = (float*)malloc(bytes);

    // Coefficients for halo exchange
    float *d_pbco, *d_pbci;
    cudaMalloc(&d_pbco, 2 * lmd * sizeof(float));
    cudaMalloc(&d_pbci, 2 * lmd * sizeof(float));
    deriv_setup(d_pbco, d_pbci);

    MPI_Barrier(MPI_COMM_WORLD);

    // Send buffer
    float * send0 = (float *)malloc(2 * 2 * dcomp_info.let * dcomp_info.lze * sizeof(float));
    float * send1 = (float *)malloc(2 * 2 * dcomp_info.lxi * dcomp_info.lze * sizeof(float));
    float *d_send0, *d_send1, *d_send2;
    cudaMalloc(&d_send0, 2 * 2 * dcomp_info.let * dcomp_info.lze * sizeof(float));
    cudaMalloc(&d_send1, 2 * 2 * dcomp_info.lxi * dcomp_info.lze * sizeof(float));
    cudaMalloc(&d_send2, 2 * 2 * dcomp_info.lxi * dcomp_info.let * sizeof(float));

    float *d_send[3];
    d_send[0] = d_send0;
    d_send[1] = d_send1;
    d_send[2] = d_send2;

    // Recv buffer
    float * recv0 = (float *)malloc(2 * 2 * dcomp_info.let * dcomp_info.lze * sizeof(float));
    float * recv1 = (float *)malloc(2 * 2 * dcomp_info.lxi * dcomp_info.lze * sizeof(float));
    float *d_recv0, *d_recv1, *d_recv2;
    cudaMalloc(&d_recv0, 2 * 2 * dcomp_info.let * dcomp_info.lze * sizeof(float));
    cudaMalloc(&d_recv1, 2 * 2 * dcomp_info.lxi * dcomp_info.lze * sizeof(float));
    cudaMalloc(&d_recv2, 2 * 2 * dcomp_info.lxi * dcomp_info.let * sizeof(float));

    float *d_recv[3];
    d_recv[0] = d_recv0;
    d_recv[1] = d_recv1;
    d_recv[2] = d_recv2;

    // drva
    float *d_drva0, *d_drva1, *d_drva2;
    cudaMalloc(&d_drva0, 2 * NumberOfVariables * dcomp_info.let * dcomp_info.lze * sizeof(float));
    cudaMalloc(&d_drva1, 2 * NumberOfVariables * dcomp_info.lxi * dcomp_info.lze * sizeof(float));
    cudaMalloc(&d_drva2, 2 * NumberOfVariables * dcomp_info.lxi * dcomp_info.let * sizeof(float));

    float *d_drva[3];
    d_drva[0] = d_drva0;
    d_drva[1] = d_drva1;
    d_drva[2] = d_drva2;

    // Halo exchange info for each face

    // ndf is 1 if halo exchange is needed, otherwise is 0
    unsigned int ndf[2][3];
    for(unsigned int ip = 0; ip < 2; ++ip)
    {
        for(unsigned int nn = 0; nn < 3; ++nn)
        {
            ndf[ip][nn] = 0;
        }
    }
    if(world_rank == 0)
    {
        ndf[1][ax] = 1;
    }
    else if(world_rank == 1)
    {
        ndf[0][ax] = 1;
    }

    // mcd indicates the pair process for each face. If there is no
    // halol exchange on a face, it is set to -1
    int mcd[2][3];
    for(unsigned int ip = 0; ip < 2; ++ip)
    {
        for(unsigned int nn = 0; nn < 3; ++nn)
        {
            mcd[ip][nn] = -1;
        }
    }

    if(world_rank == 0)
    {
        mcd[1][ax] = 1;
    }
    else if(world_rank == 1)
    {
        mcd[0][ax] = 0;
    }

    // Allocate memory for input and output fields on device
    float *d_infield, *d_outfield;
    cudaMalloc(&d_infield, bytes);
    cudaMalloc(&d_outfield, bytes);

    // Fill host input field
    if(ax == 0)
    {
        for(unsigned int k = 0; k < dcomp_info.lze; ++k)
        {
            for(unsigned int j = 0; j < dcomp_info.let; ++j)
            {
                for(unsigned int i = 0; i < dcomp_info.lxi; ++i)
                {
                    infield[i + j * dcomp_info.lxi + k * dcomp_info.lxi * dcomp_info.let] =
                        i + world_rank * dcomp_info.lxi +
                        j * 2 * dcomp_info.lxi +
                        k * 2 * dcomp_info.lxi * dcomp_info.let;
                }
            }
        }
    }
    else if(ax == 1)
    {
        for(unsigned int k = 0; k < dcomp_info.lze; ++k)
        {
            for(unsigned int j = 0; j < dcomp_info.let; ++j)
            {
                for(unsigned int i = 0; i < dcomp_info.lxi; ++i)
                {
                    infield[i + j * dcomp_info.lxi + k * dcomp_info.lxi * dcomp_info.let] =
                        world_rank * dcomp_info.lxi * dcomp_info.let +
                        i +
                        j * dcomp_info.lxi +
                        k * dcomp_info.lxi * 2 * dcomp_info.let;
                }
            }
        }
    }

    // Copy data from host to device (input and output field)
    cudaMemcpy(d_infield, infield, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outfield, outfield, bytes, cudaMemcpyHostToDevice);

    // halo exchange
    mpigo(d_infield, d_send, d_recv, d_pbco, dcomp_info, ndf, mcd);

    // Compute derivative
    deriv<ax>(d_infield, d_outfield, d_recv[ax], d_pbci,
              d_drva[ax], h_1, ndf[0][ax], ndf[1][ax], dcomp_info, variable_id);

    // Copy data from device to host
    cudaMemcpy(send0, d_send0, 2 * 2 * dcomp_info.let * dcomp_info.lze * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(recv0, d_recv0, 2 * 2 * dcomp_info.let * dcomp_info.lze * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(send1, d_send1, 2 * 2 * dcomp_info.lxi * dcomp_info.lze * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(recv1, d_recv1, 2 * 2 * dcomp_info.lxi * dcomp_info.lze * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(outfield, d_outfield, bytes, cudaMemcpyDeviceToHost);

    if(world_rank == 0)
    {
        std::cout << world_rank << ":outfield:[123] = " << outfield[123] << std::endl;
        std::cout << world_rank << ":outfield:[124] = " << outfield[124] << std::endl;
        std::cout << world_rank << ":outfield:[125] = " << outfield[125] << std::endl;
        std::cout << world_rank << ":outfield:[126] = " << outfield[126] << std::endl;
        std::cout << world_rank << ":outfield:[127] = " << outfield[127] << std::endl;
    }
    else
    {
        std::cout << world_rank << ":outfield[0] = " << outfield[0] << std::endl;
        std::cout << world_rank << ":outfield[1] = " << outfield[1] << std::endl;
        std::cout << world_rank << ":outfield[2] = " << outfield[2] << std::endl;
        std::cout << world_rank << ":outfield[3] = " << outfield[3] << std::endl;
        std::cout << world_rank << ":outfield[4] = " << outfield[4] << std::endl;
    }

    // Free CPU memory
    free(infield);
    free(outfield);
    free(send0);
    free(send1);
    free(recv0);
    free(recv1);

    // Free GPU memory
    cudaFree(d_infield);
    cudaFree(d_outfield);
    cudaFree(d_pbco);
    cudaFree(d_pbci);
    cudaFree(d_send0);
    cudaFree(d_send1);
    cudaFree(d_send2);
    cudaFree(d_recv0);
    cudaFree(d_recv1);
    cudaFree(d_recv2);

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}
