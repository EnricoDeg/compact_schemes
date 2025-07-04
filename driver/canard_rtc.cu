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
#include "domdcomp.hpp"
#include "IO.hpp"

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

    // domain decomposition
    auto domdcomp_instance = domdcomp(0);
    domdcomp_instance.read_config();
    domdcomp_instance.go();
    domdcomp_instance.show();

    static constexpr unsigned int ax = 0;

    // Subdomain info
    t_dcomp dcomp_info;
    dcomp_info.lxi = domdcomp_instance.lxi + 1;
    dcomp_info.let = domdcomp_instance.let + 1;
    dcomp_info.lze = domdcomp_instance.lze + 1;
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

    // generate grid
    auto grid_instance = grid<float>(domdcomp_instance);
    grid_instance.read_config();
    grid_instance.generate(domdcomp_instance);

    auto numerics_instance = numerics_rtc<float>(dcomp_info);

    auto physics_instance  = physics_rtc<true, float>(dcomp_info, ndf, &numerics_instance);

    std::vector<std::string> variable_names{"x", "y", "z", "density", "u", "v", "w", "p"};
    auto io_instance = IOwriter(5, domdcomp_instance, variable_names);
    // io_instance.go(domdcomp_instance, grid_instance, data);

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
                                    domdcomp_instance.mcd,
                                    &numerics_instance,
                                    &streams[0]);
    }

    for(unsigned int i = 0; i < NStreams; ++i)
    {
        check_cuda_driver( cuStreamDestroy ( streams[i] ));
    }

    return 0;
}
