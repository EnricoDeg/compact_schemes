#include <stdio.h>
#include <cuda.h>
#include <mpi.h>
#include "yaml-cpp/yaml.h"

#include "common/parameters.hpp"
#include "cuda/common.hpp"
#include "cuda/general.hpp"
#include "domdcomp.hpp"
#include "IO.hpp"
#include "numerics_pc.hpp"
#include "gcbc.hpp"
#include "grid.hpp"
#include "physics_pc.hpp"
#include "sponge.hpp"

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
    const int nblocks      = driver_yaml[0]["nblocks"].as<int>();
    const int nts          = driver_yaml[1]["nts"].as<int>();
    bool output_enabled    = driver_yaml[2]["output_enabled"].as<bool>();

    // domain decomposition
    YAML::Node domdcomp_yaml = config["domdcomp"];
    auto domdcomp_instance = domdcomp(nblocks);
    domdcomp_instance.read_config(domdcomp_yaml);
    domdcomp_instance.go();
    domdcomp_instance.show();

    // Subdomain info
    t_dcomp dcomp_info;
    dcomp_info.lxi = domdcomp_instance.lxi + 1;
    dcomp_info.let = domdcomp_instance.let + 1;
    dcomp_info.lze = domdcomp_instance.lze + 1;
    dcomp_info.lmx = dcomp_info.lxi * dcomp_info.let * dcomp_info.lze;

    // qa
    float * d_qa = allocate_cuda<float>(NumberOfVariables *
        dcomp_info.lmx);

    // qo
    float * d_qo = allocate_cuda<float>(NumberOfVariables *
        dcomp_info.lmx);

    // de
    float * d_de = allocate_cuda<float>(NumberOfVariables *
        dcomp_info.lmx);

    // pressure
    float * d_pressure;
    cudaMalloc(&d_pressure, dcomp_info.lmx * sizeof(float));

    // ss
    float *d_ss;
    cudaMalloc(&d_ss, dcomp_info.lmx * sizeof(float));

    // generate grid
    YAML::Node grid_yaml = config["grid"];
    auto grid_instance = grid<float>(domdcomp_instance);
    grid_instance.read_config(grid_yaml);
    grid_instance.generate(domdcomp_instance);

    YAML::Node physics_yaml = config["physics"];
    auto physics_instance = physics<true, float>(dcomp_info);
    physics_instance.read_config(physics_yaml);

    auto numerics_instance = numerics_pc<float>(dcomp_info, domdcomp_instance.nbc);

    std::vector<std::string> variable_names{"x", "y", "z",
        "density", "u", "v", "w", "p"};
    auto io_instance = IOwriter(5, domdcomp_instance, variable_names);

    // setup derivatives
    numerics_instance.deriv_setup();

    // compute grid metrics (needs numerics instance to be initialized)
    grid_instance.calc_metrics(domdcomp_instance,
                           dcomp_info,
                           domdcomp_instance.mcd,
                           &numerics_instance);

    // sponge
    YAML::Node sponge_yaml = config["sponge"];
    auto sponge_instance = sponge<float>();
    sponge_instance.read_config(sponge_yaml);
    sponge_instance.up(grid_instance, dcomp_info.lmx);

    // gcbc
    auto gcbc_instance = gcbc<float, int>(domdcomp_instance, grid_instance.yaco);

    cudaStream_t stream[5];
    for(int i=0; i<5; i++) cudaStreamCreate(&stream[i]);

    size_t n, ndt;
    float dt, dts, dte, timo;
    if(nts == 0)
    {
        n = 0;
        ndt = 0;
        dt = 0.01f;
        dts = 0.0f;
        dte = 0.0f;
        timo = 0.0f;
        physics_instance.init(d_qa, dcomp_info.lmx, &grid_instance);
    }
    else
    {
        // read restart file
    }

    float tmax = 0.1;
    float cfl = 0.95f;
    bool nout;
    float res;
    int ndati = -1;
    float dtk, dtko;
    int ndata = 10;

    check_mpi(MPI_Barrier(MPI_COMM_WORLD));

    do{
        if(world_rank == 0)
        {
            std::cout << "Time step = " << n << std::endl;
        }

        init_main_loop(d_qa, d_qo, dcomp_info.lmx);

        for(int nk = 0; nk < nkrk; ++nk)
        {

            // move frame velocity and acceleration before time advancing
            dtko = dt * min( max( nk - 2, 0 ), 1 ) / ( nkrk - nk + 3 );
            dtk  = dt * min( nk - 1, 1 ) / ( nkrk - nk + 2 );
            physics_instance.movef(dtko, dtk, timo);

            // temporary storage of primitive variables and pressure
            init_runge_kutta(d_de,
                             d_qa,
                             d_pressure,
                             d_ss,
                             physics_instance.srefp1dre,
                             physics_instance.srefoo,
                             dcomp_info.lmx);

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
                                                    grid_instance.yaco,
                                                    d_ss,
                                                    cfl,
                                                    &dte,
                                                    dcomp_info.lmx);
                }
                // dt = dts + (dte - dts) *
                //     std::sin(0.05f * pi * (n - ndt)) *
                //     std::sin(0.05f * pi * (n - ndt));

                nout = false;
                res = (ndati + 1) * tmax / ndata;
                if((timo - res) * (timo + dt - res) <= 0.0f)
                {
                    nout = true;
                    ndati++;
                }
            }

            // compute viscous shear stress
            physics_instance.calc_viscous_shear_stress(d_de,
                                                       d_ss,
                                                       grid_instance.xim,
                                                       grid_instance.etm,
                                                       grid_instance.zem,
                                                       grid_instance.yaco,
                                                       dcomp_info,
                                                       domdcomp_instance.mcd,
                                                       &numerics_instance,
                                                       &stream[0]);

            // compute fluxes
            physics_instance.calc_fluxes(d_qa,
                                         d_pressure,
                                         d_de,
                                         grid_instance.xim,
                                         grid_instance.etm,
                                         grid_instance.zem,
                                         dcomp_info,
                                         domdcomp_instance.mcd,
                                         &numerics_instance,
                                         &stream[0]);

            // GCBC
            // gcbc_instance.go(numerics_instance.drva_buffer,
            //                  grid_instance.cm,
            //                  d_qa,
            //                  d_de,
            //                  d_pressure,
            //                  grid_instance.yaco,
            //                  physics_instance.umf,
            //                  physics_instance.dudtmf,
            //                  dcomp_info,
            //                  1.0 / dt,
            //                  domdcomp_instance.nbc,
            //                  domdcomp_instance.mcd);

            // sponge condition
            // sponge_instance.go(d_qa, d_de, dcomp_info.lmx);

            // update conservative variables
            dtko = dt * min(nk-1, 1) / (nkrk - nk + 2);
            dtk  = dt / (nkrk - nk + 1);
            physics_instance.movef(dtko, dtk, timo);

            // updating conservative variables
            update_conservative_variables(d_qa,
                                          d_qo,
                                          d_de,
                                          grid_instance.yaco,
                                          dtk,
                                          dcomp_info.lmx);

            // wall temperature / velocity condition
            // gcbc_instance.wall_condition_go(d_qa,
            //                                 physics_instance.umf,
            //                                 dcomp_info,
            //                                 domdcomp_instance.nbc);
        }

        // advance in time
        n++;
        timo += dt;

        // record intermediate results
        if(output_enabled)
        {
            if(timo > (-tmax) / ndata)
            {
                if(nout && world_rank == 0)
                    std::cout << "writing output n = " << n << "\n";
                io_instance.fill_buffer(d_qo,
                    d_qa, grid_instance.d_patch, dt, physics_instance.umf, nout);
                io_instance.go(domdcomp_instance, grid_instance, nout);
            }
        }

    } while(timo < tmax && (dt != 0.0f || n <= 2));

    for(int i=0; i<5; i++) cudaStreamDestroy(stream[i]);

    // free memory
    free_cuda(d_qa);
    free_cuda(d_qo);
    free_cuda(d_de);

    // Finalize the MPI environment.
    check_mpi(MPI_Finalize());

    return 0;
}
