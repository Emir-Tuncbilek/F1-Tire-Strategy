//
// Created by Emir Tuncbilek on 7/28/24.
//

#ifndef F1_STRATEGIES_GPUFUNCTIONS_H
#define F1_STRATEGIES_GPUFUNCTIONS_H

#include <OpenCL/opencl.h>

#include <iostream>
#include <string>
#include <vector>

class GPUFunctions {
public:
    GPUFunctions() = default;

    void init();
    virtual bool attachKernel(const std::string& path) = 0;
protected:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;
    std::string source;

    void readFile(const std::string& path);
    bool buildKernel(const std::string& kernelName);
    void cleanUp();
    static static bool checkError(cl_int error, const std::string& message);
};

class GPUMatrixMultiplier : public GPUFunctions {
public:
    bool attachKernel(const std::string &path) override;
    bool execute(const std::vector<float>& A,
                 const std::vector<float>& B,
                 std::vector<float>& result,
                 unsigned int M,
                 unsigned int N,
                 unsigned int K);
};

#endif //F1_STRATEGIES_GPUFUNCTIONS_H
