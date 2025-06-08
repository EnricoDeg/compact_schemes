/*
 * @file compiler.hpp
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

#ifndef CANARD_CUDA_COMPILER_HPP
#define CANARD_CUDA_COMPILER_HPP

#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "cuda/check_rtc.hpp"

template <typename T>
std::string GetTypeName()
{
    if constexpr (std::is_same<T, float>::value == true)
    {
        return std::string("float");
    }
}

void compile_program(nvrtcProgram prog, const char *opts[], int num_opts)
{
    nvrtcResult compileResult = nvrtcCompileProgram(prog,
                                                    num_opts,
                                                    opts);

    // Obtain compilation log from the program.
    size_t logSize;
    check_cuda_rtc( nvrtcGetProgramLogSize(prog, &logSize) );
    char *log = new char[logSize];
    check_cuda_rtc( nvrtcGetProgramLog(prog, log) );
    delete[] log;
    if (compileResult != NVRTC_SUCCESS) {
        exit(1);
    }
}

std::string readFile(std::string file, std::string path)
{
    std::string myText;
    std::string str;
    // read file
    std::ifstream MyReadFile(path+file);
    while (std::getline (MyReadFile, str)) {
        myText += str;
        myText.push_back('\n');
    }

    // Close the file
    MyReadFile.close();

    return myText;
}

class rt_compiler
{
    public:
    rt_compiler(std::string filename,
                std::string program_name,
                std::vector<std::string> kernel_name_vec,
                std::vector<std::string> kernel_opts)
    {
        const char* env_p;
        if (!(env_p = getenv("CANARD_ROOT")))
        {
            std::cout << "Error: CANARD_ROOT=" << env_p << '\n';
            exit(EXIT_FAILURE);
        }
        std::string project_root_path(env_p);

        std::string myText = readFile(std::string("/")+filename, project_root_path);
        check_cuda_rtc( nvrtcCreateProgram(&prog,
                                           myText.c_str(),
                                           program_name.c_str(),
                                           0,
                                           NULL,
                                           NULL) );

        // add kernel name expressions to NVRTC. Note this must be done before
        // the program is compiled.
        for(unsigned int i = 0; i < kernel_name_vec.size(); ++i)
        {
            check_cuda_rtc( nvrtcAddNameExpression(prog, kernel_name_vec[i].c_str()) );
        }

        std::string include_dirs = std::string("-I")+project_root_path;
        // std::string cuda_include_dirs0 = std::string("-I/sw/spack-levante/nvhpc-23.9-xpxqeo/Linux_x86_64/23.9/cuda/12.2/include");
        // std::string cuda_include_dirs1 = std::string("-I/sw/spack-levante/nvhpc-23.9-xpxqeo/Linux_x86_64/23.9/math_libs/12.2/include");
        // std::string cuda_include_dirs2 = std::string("-I/sw/spack-levante/nvhpc-23.9-xpxqeo/Linux_x86_64/23.9/comm_libs/12.2/nccl/include");
        // std::string cuda_include_dirs3 = std::string("-I/sw/spack-levante/nvhpc-23.9-xpxqeo/Linux_x86_64/23.9/comm_libs/12.2/nvshmem/include");
        // std::string cuda_include_dirs4 = std::string("-I/sw/spack-levante/nvhpc-23.9-xpxqeo/Linux_x86_64/23.9/cuda/12.2/bin/../targets/x86_64-linux/include");
        // std::string cuda_include_dirs5 = std::string("-I/sw/spack-levante/gcc-11.2.0-bcn7mb/include");
        // std::string cuda_include_dirs6 = std::string("-I/sw/spack-levante/gcc-11.2.0-bcn7mb/lib/gcc/x86_64-pc-linux-gnu/11.2.0/include");
        // std::string cuda_include_dirs7 = std::string("-I/sw/spack-levante/gcc-11.2.0-bcn7mb/lib/gcc/x86_64-pc-linux-gnu/11.2.0/include-fixed");
        // std::string cuda_include_dirs8 = std::string("-I/usr/include");



        const char *opts[kernel_opts.size() + 4];
        opts[0] = "-std=c++17";
        opts[1] = "--gpu-architecture=compute_80";
        opts[2] = "-default-device";
        opts[3] = include_dirs.c_str();
        // opts[4] = cuda_include_dirs0.c_str();
        // opts[5] = cuda_include_dirs1.c_str();
        // opts[6] = cuda_include_dirs2.c_str();
        // opts[7] = cuda_include_dirs3.c_str();
        // opts[8] = cuda_include_dirs4.c_str();
        // opts[9] = cuda_include_dirs5.c_str();
        // opts[10] = cuda_include_dirs6.c_str();
        // opts[11] = cuda_include_dirs7.c_str();
        // opts[12] = cuda_include_dirs8.c_str();

        for(unsigned int i = 0; i < kernel_opts.size(); ++i)
        {
            opts[4+i] = kernel_opts[i].c_str();
        }

        // compile program
        compile_program(prog, opts, kernel_opts.size() + 4);

        // Obtain PTX from the program.
        size_t ptxSize;
        check_cuda_rtc( nvrtcGetPTXSize(prog, &ptxSize) );
        char *ptx = new char[ptxSize];
        check_cuda_rtc( nvrtcGetPTX(prog, ptx) );

        // Load module
        check_cuda_driver( cuModuleLoadDataEx(&module, ptx, 0, 0, 0) );

        // for each of the kernel name expressions previously provided to NVRTC,
        // extract the lowered name for corresponding __global__ function,
        // and launch it.

        for(unsigned int i = 0; i < kernel_name_vec.size(); ++i)
        {
            const char *name;
            CUfunction kernel;
            // note: this call must be made after NVRTC program has been
            // compiled and before it has been destroyed.
            check_cuda_rtc( nvrtcGetLoweredName(prog,
                                                kernel_name_vec[i].c_str(),
                                                &name));

            // get kernel function
            check_cuda_driver( cuModuleGetFunction(&kernel, module, name) );
            names.push_back(std::string(name));
            kernels.push_back(kernel);
        }
        delete[] ptx;
    }

    ~rt_compiler()
    {
        check_cuda_driver( cuModuleUnload(module) );
        check_cuda_rtc( nvrtcDestroyProgram(&prog) );
    }

    CUfunction get_kernel(unsigned int i)
    {
        return kernels[i];
    }

    std::string get_name(unsigned int i)
    {
        return names[i];
    }

    private:
    nvrtcProgram prog;
    CUmodule module;
    std::vector<CUfunction> kernels;
    std::vector<std::string> names;
};

#endif
