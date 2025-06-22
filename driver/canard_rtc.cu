#include <stdio.h>
#include <cuda.h>
#include <mpi.h>

#include "common/parameters.hpp"
#include "common/data_types.hpp"
#include "cuda/check.hpp"
#include "cuda/driver.hpp"
#include "mpi/check.hpp"
#include "mpi/driver.hpp"
#include "grid.hpp"
#include "numerics_rtc.hpp"
#include "physics_rtc.hpp"

// Main program
int main()
{
    mpi_driver mpi_driver_instance{};
    cuda_driver cuda_driver_instance{};

    // Get the number of processes
    int world_size;
    check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // Get the rank of the process
    int world_rank;
    check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    static constexpr unsigned int ax = 0;

    // Subdomain info
    t_dcomp dcomp_info;
    dcomp_info.lxi = 1024;
    dcomp_info.let = 64;
    dcomp_info.lze = 64;
    dcomp_info.lmx = dcomp_info.lxi * dcomp_info.let * dcomp_info.lze;

    // qa
    float * d_qa;
    cudaMalloc(&d_qa, NumberOfSpatialDims * dcomp_info.lmx * sizeof(float));

    // de
    float * d_de;
    cudaMalloc(&d_de, NumberOfSpatialDims * dcomp_info.lmx * sizeof(float));

    // pressure
    float * d_pressure;
    cudaMalloc(&d_pressure, dcomp_info.lmx * sizeof(float));

    // umf
    t_point<float> umf = {.x = 0.3, .y = 0.0, .z = 0.0 };

    unsigned int ndf[2][3];
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 3; ++j)
            ndf[i][j] = 0;
    if(world_rank == 0)
    {
        ndf[1][ax] = 1;
    }
    else if(world_rank == 1)
    {
        ndf[0][ax] = 1;
    }

    int mcd[2][3];
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 3; ++j)
            mcd[i][j] = -1;

    if(world_rank == 0)
    {
        mcd[1][ax] = 1;
    }
    else if(world_rank == 1)
    {
        mcd[0][ax] = 0;
    }

    auto grid_instance = grid<float>(dcomp_info);

    auto numerics_instance = numerics_rtc<float>(dcomp_info);

    auto physics_instance  = physics_rtc<true, float>(dcomp_info, ndf, &numerics_instance);

    // setup derivatives
    numerics_instance.deriv_setup();

    constexpr unsigned int NStreams = 5;
    CUstream streams[NStreams];
    for(unsigned int i = 0; i < NStreams; ++i)
    {
       check_cuda_driver(cuStreamCreate ( &streams[i], CU_STREAM_NON_BLOCKING ));
    }

    check_mpi(MPI_Barrier(MPI_COMM_WORLD));

    // compute fluxes
    for(unsigned int i = 0; i < 10; ++i)
    {
        physics_instance.calc_fluxes(d_qa,
                                    d_pressure,
                                    d_de,
                                    grid_instance.xim,
                                    grid_instance.etm,
                                    grid_instance.zem,
                                    dcomp_info,
                                    umf,
                                    ndf,
                                    mcd,
                                    &numerics_instance,
                                    &streams[0]);
    }

    for(unsigned int i = 0; i < NStreams; ++i)
    {
        check_cuda_driver( cuStreamDestroy ( streams[i] ));
    }

    // check_cuda( cudaMemcpy(outfield, d_outfield,
    //     dcomp_info.lmx * sizeof(float), cudaMemcpyDeviceToHost) );

    // float solution;
    // if constexpr(ax == 0)
    // {
    //     solution = infield[1] - infield[0];
    // }
    // else if constexpr(ax == 1)
    // {
    //     solution = infield[dcomp_info.lxi] - infield[0];
    // }
    // else if constexpr(ax == 2)
    // {
    //     solution = infield[dcomp_info.lxi * dcomp_info.let] - infield[0];
    // }
    // for(unsigned int i = 0; i < dcomp_info.lmx; ++i)
    // {
    //     if(std::abs(outfield[i] - solution) / solution > 1e-6)
    //     {
    //         std::cout << std::abs(outfield[i] - solution) / solution << std::endl;
    //         std::cout << i << ": out = " << outfield[i] << " -- ref = " << solution << std::endl;
    //         exit(1);
    //     }
    // }

    return 0;
}
