#include <stdio.h>
#include <cuda.h>
#include <mpi.h>

#include "common/parameters.hpp"
#include "common/data_types.hpp"
#include "cuda/check.hpp"
#include "cuda/driver.hpp"
#include "mpi/check.hpp"
#include "mpi/driver.hpp"
#include "numerics_rtc.hpp"

// Main program
int main()
{
    mpi_driver mpi_driver_instance{};
    std::cout << "mpi driver initialized\n";
    cuda_driver cuda_driver_instance{};
    std::cout << "cuda driver initialized\n";

    // Get the number of processes
    int world_size;
    check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // Get the rank of the process
    int world_rank;
    check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    static constexpr unsigned int ax = 1;

    // Subdomain info
    t_dcomp dcomp_info;
    dcomp_info.lxi = 1024;
    dcomp_info.let = 64;
    dcomp_info.lze = 64;
    dcomp_info.lmx = dcomp_info.lxi * dcomp_info.let * dcomp_info.lze;

    float *infield = (float *)malloc(dcomp_info.lmx * sizeof(float));
    float *d_infield;
    check_cuda( cudaMalloc(&d_infield, dcomp_info.lmx * sizeof(float)) );

    float *outfield = (float *)malloc(dcomp_info.lmx * sizeof(float));
    float *d_outfield;
    check_cuda( cudaMalloc(&d_outfield, dcomp_info.lmx * sizeof(float)) );

    for(unsigned int i = 0; i < dcomp_info.lmx; ++i)
    {
        infield[i] = i;
    }
    check_cuda( cudaMemcpy(d_infield, infield,
        dcomp_info.lmx * sizeof(float), cudaMemcpyHostToDevice));

    int nstart = 0;
    int nend   = 0;

    auto numerics_instance = numerics_rtc<float>(dcomp_info);

    numerics_instance.template deriv1d_compile<ax>(dcomp_info, nstart, nend);

    std::cout << "compilation done\n";

    constexpr unsigned int NStreams = 1;
    CUstream streams[NStreams];
    for(unsigned int i = 0; i < NStreams; ++i)
    {
       check_cuda_driver(cuStreamCreate ( &streams[i], CU_STREAM_NON_BLOCKING ));
    }

    // setup derivatives
    numerics_instance.deriv_setup();

    check_mpi(MPI_Barrier(MPI_COMM_WORLD));

    for(unsigned int i = 0; i < 10; ++i)
    {
        numerics_instance.template deriv1d<ax>(d_infield,
            d_outfield,
            dcomp_info,
            0,
            streams);
    }

    for(unsigned int i = 0; i < NStreams; ++i)
    {
        check_cuda_driver( cuStreamDestroy ( streams[i] ));
    }

    check_cuda( cudaMemcpy(outfield, d_outfield,
        dcomp_info.lmx * sizeof(float), cudaMemcpyDeviceToHost) );

    std::cout << outfield[0] << std::endl;
    std::cout << outfield[1] << std::endl;
    std::cout << outfield[2] << std::endl;
    std::cout << outfield[3] << std::endl;
    std::cout << outfield[4] << std::endl;
    std::cout << outfield[5] << std::endl;
    std::cout << outfield[6] << std::endl;
    std::cout << outfield[7] << std::endl;

    std::cout << "Done\n";

    return 0;
}
