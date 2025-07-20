#include <stdio.h>
#include <cuda.h>
#include <mpi.h>
#include "yaml-cpp/yaml.h"

#include "common/parameters.hpp"
#include "cuda/general.hpp"
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
    const int nts = driver_yaml[1]["nts"].as<int>();
    bool output_enabled = driver_yaml[2]["output_enabled"].as<bool>();

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

    // cm
    // float *d_cm0, *d_cm1, *d_cm2;
    // cudaMalloc(&d_cm0, 2 * NumberOfSpatialDims * dcomp_info.let * dcomp_info.lze * sizeof(float));
    // cudaMalloc(&d_cm1, 2 * NumberOfSpatialDims * dcomp_info.lxi * dcomp_info.lze * sizeof(float));
    // cudaMalloc(&d_cm2, 2 * NumberOfSpatialDims * dcomp_info.lxi * dcomp_info.let * sizeof(float));

    // float *d_cm[3];
    // d_cm[0] = d_cm0;
    // d_cm[1] = d_cm1;
    // d_cm[2] = d_cm2;

    // qa
    float * d_qa;
    cudaMalloc(&d_qa, NumberOfVariables * dcomp_info.lmx * sizeof(float));

    // qo
    float * d_qo;
    cudaMalloc(&d_qo, NumberOfVariables * dcomp_info.lmx * sizeof(float));

    // de
    float * d_de;
    cudaMalloc(&d_de, NumberOfVariables * dcomp_info.lmx * sizeof(float));

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
    // int * d_npex;
    // cudaMalloc(&d_npex, dcomp_info.lmx * sizeof(int));

    // generate grid
    YAML::Node grid_yaml = config["grid"];
    auto grid_instance = grid<float>(domdcomp_instance);
    grid_instance.read_config(grid_yaml);
    grid_instance.generate(domdcomp_instance);

    YAML::Node physics_yaml = config["physics"];
    auto physics_instance = physics<true, float>(dcomp_info);
    physics_instance.read_config(physics_yaml);

    auto numerics_instance = numerics_pc<float>(dcomp_info, domdcomp_instance.nbc);

    std::vector<std::string> variable_names{"x", "y", "z", "density", "u", "v", "w", "p"};
    auto io_instance = IOwriter(5, domdcomp_instance, variable_names);
    // io_instance.go(domdcomp_instance, grid_instance, data);

    // setup derivatives
    numerics_instance.deriv_setup();

    cudaStream_t stream[5];
    for(int i=0; i<5; i++) cudaStreamCreate(&stream[i]);

    size_t n, ndt;
    float dt, dts, dte, timo;
    if(nts == 0)
    {
        n = 0;
        ndt = 0;
        dt = 0.1f;
        dts = 0.0f;
        dte = 0.0f;
        timo = 0.0f;
        physics_instance.init(d_qa, dcomp_info.lmx, &grid_instance);
    }
    else
    {
        // read restart file
    }

    float tmax = 1.0;
    float cfl = 0.95f;
    bool nout;
    float res;
    int ndati = -1;
    float dtk, dtko;
    int ndata = 2;

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
                                                    d_yaco,
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
            physics_instance.calc_viscous_shear_stress(d_de, d_ss,
               grid_instance.xim, grid_instance.etm, grid_instance.zem,
               d_yaco, dcomp_info, domdcomp_instance.mcd, &numerics_instance, &stream[0]);

            // compute fluxes
            physics_instance.calc_fluxes(d_qa, d_pressure, d_de,
                                         grid_instance.xim, grid_instance.etm, grid_instance.zem,
                                         dcomp_info,
                                         domdcomp_instance.mcd, &numerics_instance, &stream[0]);

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
        if(output_enabled)
        {
            if(timo > (-tmax) / ndata)
            {
                io_instance.fill_buffer(d_qo,
                    d_qa, grid_instance.d_patch, dt, physics_instance.umf, nout);
                io_instance.go(domdcomp_instance, grid_instance, nout);
            }
        }

    } while(timo < tmax && (dt != 0.0f || n <= 2));

    for(int i=0; i<5; i++) cudaStreamDestroy(stream[i]);

    // Finalize the MPI environment.
    check_mpi(MPI_Finalize());

    return 0;
}
