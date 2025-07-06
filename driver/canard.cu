#include <stdio.h>
#include <cuda.h>
#include <mpi.h>
#include "yaml-cpp/yaml.h"

#include "common/parameters.hpp"
#include "domdcomp.hpp"
#include "IO.hpp"
#include "numerics_pc.hpp"
#include "gcbc.hpp"
#include "grid.hpp"
#include "physics_pc.hpp"

// Main program
int main()
{
    // Initialize the MPI environment
    check_mpi(MPI_Init(NULL, NULL));

    // Get the number of processes
    int world_size;
    check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // Get the rank of the process
    int world_rank;
    check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    // Yaml Configuration
    const char* env_p;
    if (!(env_p = getenv("CANARD_ROOT")))
    {
        std::cout << "Error: CANARD_ROOT=" << env_p << '\n';
        exit(EXIT_FAILURE);
    }
    std::string project_root_path(env_p);

    YAML::Node config = YAML::LoadFile(project_root_path+std::string("/config.yaml"));
    YAML::Node driver_yaml = config["driver"];
    const int nblocks = driver_yaml[0]["nblocks"].as<int>();

    // domain decomposition
    YAML::Node domdcomp_yaml = config["domdcomp"];
    auto domdcomp_instance = domdcomp(nblocks);
    domdcomp_instance.read_config(domdcomp_yaml);
    domdcomp_instance.go();
    domdcomp_instance.show();

    static constexpr unsigned int ax = 0;

    // Subdomain info
    t_dcomp dcomp_info;
    dcomp_info.lxi = domdcomp_instance.lxi + 1;
    dcomp_info.let = domdcomp_instance.let + 1;
    dcomp_info.lze = domdcomp_instance.lze + 1;
    dcomp_info.lmx = dcomp_info.lxi * dcomp_info.let * dcomp_info.lze;

    // Spatial distance
    float h = 1.0;
    float h_1 = 1.0 / h;

    // cm
    float *d_cm0, *d_cm1, *d_cm2;
    cudaMalloc(&d_cm0, 2 * NumberOfSpatialDims * dcomp_info.let * dcomp_info.lze * sizeof(float));
    cudaMalloc(&d_cm1, 2 * NumberOfSpatialDims * dcomp_info.lxi * dcomp_info.lze * sizeof(float));
    cudaMalloc(&d_cm2, 2 * NumberOfSpatialDims * dcomp_info.lxi * dcomp_info.let * sizeof(float));

    float *d_cm[3];
    d_cm[0] = d_cm0;
    d_cm[1] = d_cm1;
    d_cm[2] = d_cm2;

    // qa
    float * d_qa;
    cudaMalloc(&d_qa, NumberOfSpatialDims * dcomp_info.lmx * sizeof(float));

    // de
    float * d_de;
    cudaMalloc(&d_de, NumberOfSpatialDims * dcomp_info.lmx * sizeof(float));

    // pressure
    float * d_pressure;
    cudaMalloc(&d_pressure, dcomp_info.lmx * sizeof(float));

    // yaco
    float *d_yaco;
    cudaMalloc(&d_yaco, dcomp_info.lmx * sizeof(float));

    // ss
    float *d_ss;
    cudaMalloc(&d_ss, dcomp_info.lmx * sizeof(float));

    // npex
    int * d_npex;
    cudaMalloc(&d_npex, dcomp_info.lmx * sizeof(int));

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

    // generate grid
    YAML::Node grid_yaml = config["grid"];
    auto grid_instance = grid<float>(domdcomp_instance);
    grid_instance.read_config(grid_yaml);
    grid_instance.generate(domdcomp_instance);

    YAML::Node physics_yaml = config["physics"];
    auto physics_instance = physics<true, float>(dcomp_info);
    physics_instance.read_config(physics_yaml);

    auto numerics_instance = numerics_pc<float>(dcomp_info);

    std::vector<std::string> variable_names{"x", "y", "z", "density", "u", "v", "w", "p"};
    auto io_instance = IOwriter(5, domdcomp_instance, variable_names);
    // io_instance.go(domdcomp_instance, grid_instance, data);

    // setup derivatives
    numerics_instance.deriv_setup();

    cudaStream_t stream[5];
    for(int i=0; i<5; i++) cudaStreamCreate(&stream[i]);

    size_t n = 0;
    size_t ndt = 0;
    float dt = 0.1f;
    float dts = 0.0f;
    float dte = 0.0f;
    float timo = 0.0f;
    float dtsum = 0.0f;
    float tmax = 1.0;
    float cfl = 0.95f;
    int nout;
    float res;
    int ndati = -1;
    float dtk, dtko;
    int ndata = 2;
    bool output_enabled = false;

    physics_instance.init();

    check_mpi(MPI_Barrier(MPI_COMM_WORLD));

    do{
        std::cout << "Time step = " << n << std::endl;
        for(int nk = 0; nk < nkrk; ++nk)
        {

            // move frame velocity and acceleration before time advancing
            dtko = dt * min( max( nk - 2, 0 ), 1 ) / ( nkrk - nk + 3 );
            dtk  = dt * min( nk - 1, 1 ) / ( nkrk - nk + 2 );
            physics_instance.movef(dtko, dtk, timo);

            // temporary storage of primitive variables and pressure


            // compute time step size and output time
            if(nk == 1)
            {
                if(n % 10 == 1)
                {
                    ndt = n;
                    dts = dte;
                    physics_instance.calc_time_step(grid_instance.xim,
                                                    grid_instance.etm,
                                                    grid_instance.zem,
                                                    d_de,
                                                    d_yaco,
                                                    d_ss,
                                                    cfl,
                                                    &dte,
                                                    dcomp_info.lmx);
                }
                // dt = dts + (dte - dts) *
                //     std::sin(0.05f * pi * (n - ndt)) *
                //     std::sin(0.05f * pi * (n - ndt));

                nout = 0;
                res = (ndati + 1) * tmax / ndata;
                if((timo - res) * (timo + dt - res) <= 0.0f)
                {
                    nout = 1;
                    ndati++;
                }
            }

            // compute viscous shear stress
            physics_instance.calc_viscous_shear_stress(d_de, d_ss,
               grid_instance.xim, grid_instance.etm, grid_instance.zem,
               d_yaco, dcomp_info, h_1, ndf, domdcomp_instance.mcd, &numerics_instance, &stream[0]);

            // compute fluxes
            physics_instance.calc_fluxes(d_qa, d_pressure, d_de,
                                         grid_instance.xim, grid_instance.etm, grid_instance.zem,
                                         dcomp_info,
                                         h_1, ndf, domdcomp_instance.mcd, &numerics_instance, &stream[0]);

            float dtwi = 1 / dt;

            // GCBC
            // auto gcbc_instance = gcbc<float, int>(dcomp_info);
            // gcbc_go(numerics_instance.drva_buffer, d_cm, gcbc_instance.drvb,
            //         d_qa, d_de, d_pressure, d_yaco, gcbc_instance.sbcc,
            //         umf, dudtmf, dcomp_info, dtwi,
            //         domdcomp_instance.nbc, mcd);

            // sponge condition

            // update conservative variables
            dtko = dt * min(nk-1, 1) / (nkrk - nk + 2);
            dtk  = dt / (nkrk - nk + 1);
            physics_instance.movef(dtko, dtk, timo);

            // wall temperature / velocity condition

            // wall_condition_update(d_qa, d_npex, umf, dcomp_info, domdcomp_instance.nbc);
        }

        // advance in time
        n++;
        timo += dt;

        // record intermediate results
        // if(output_enabled)
        // {
        //     if(timo > (-tmax) / ndata)
        //     {
        //         dtsum += dt;
        //         if(nout == 1)
        //         {

        //         }
        //     }
        // }

    } while(timo < tmax && (dt != 0.0f || n <= 2));

    for(int i=0; i<5; i++) cudaStreamDestroy(stream[i]);

    // Finalize the MPI environment.
    check_mpi(MPI_Finalize());

    return 0;
}
