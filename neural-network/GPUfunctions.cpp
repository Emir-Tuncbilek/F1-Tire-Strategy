//
// Created by Emir Tuncbilek on 7/28/24.
//
#include <string>
#include <fstream>
#include <sstream>

#include "GPUfunctions.h"


void GPUFunctions::readFile(const std::string &path) {
    std::ifstream file(path);
    if (!file) {
        throw std::invalid_argument("File not found");
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    this->source = buffer.str();
}

void GPUFunctions::init() {
    cl_int ret;

    // Get the number of platforms
    cl_uint platformCount;
    ret = clGetPlatformIDs(0, nullptr, &platformCount);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to get platform count. Error code: " << ret << std::endl;
        return;
    }

    std::vector<cl_platform_id> platforms(platformCount);
    ret = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to get platforms. Error code: " << ret << std::endl;
        return;
    }

    for (cl_uint i = 0; i < platformCount; i ++) {
        cl_uint deviceCount;
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        if (ret != CL_SUCCESS) {
            std::cerr << "Failed to get device count for platform " << i << ". Error code: " << ret << std::endl;
            continue;
        }

        std::vector<cl_device_id> devices(deviceCount);
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
        if (ret != CL_SUCCESS) {
            std::cerr << "Failed to get devices for platform " << i << ". Error code: " << ret << std::endl;
            continue;
        }

        std::cout << "Platform " << i << " has " << deviceCount << " device(s)." << std::endl;
        for (cl_uint j = 0; j < deviceCount; ++j) {
            char deviceName[128];
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            if (ret != CL_SUCCESS) {
                std::cerr << "Failed to get device name for device " << j << ". Error code: " << ret << std::endl;
                continue;
            }
            std::cout << "  Device " << j << ": " << deviceName << std::endl;

            cl_uint computeUnits;
            ret = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
            if (ret != CL_SUCCESS) {
                std::cerr << "Failed to get compute units for device " << j << ". Error code: " << ret << std::endl;
                continue;
            }
            std::cout << "    Number of compute units: " << computeUnits << std::endl;

            // Choose the first available device
            this->platform = platforms[i];
            this->device = devices[j];
            break;
        }
        // if (this->device) break;
    }

    if (!this->device) {
        std::cerr << "No suitable device found." << std::endl;
        return;
    }

    cl_int err;
    this->context = clCreateContext(nullptr, 1, &this->device, nullptr, nullptr, &err);
    if (!checkError(err, "Failed to create context")) return;

    this->commandQueue = clCreateCommandQueue(this->context, this->device, 0, &err);
    if (!checkError(err, "Failed to create command queue")) return;
}

void GPUFunctions::cleanUp() {
    if (this->kernel) clReleaseKernel(kernel);
    if (this->program) clReleaseProgram(program);
    if (this->commandQueue) clReleaseCommandQueue(commandQueue);
    if (this->context) clReleaseContext(context);
    this->commandQueue = nullptr;
    this->context = nullptr;
    this->program = nullptr;
    this->kernel = nullptr;
}

bool GPUFunctions::checkError(cl_int error, const std::string &message) {
    if (error != CL_SUCCESS) {
        std::cerr << message << ": " << error << std::endl;
        return false;
    }
    return true;
}

bool GPUFunctions::buildKernel(const std::string &kernelName) {

    cl_int err;
    const char* sourceCStr = this->source.c_str();
    size_t sourceSize = this->source.length();
    this->program = clCreateProgramWithSource(this->context, 1, &sourceCStr, &sourceSize, &err);
    if (!checkError(err, "Failed to create program")) return false;

    err = clBuildProgram(this->program, 1, &this->device, nullptr, nullptr, nullptr);
    if (!checkError(err, "Failed to build program")) return false;

    this->kernel = clCreateKernel(this->program, kernelName.c_str(), &err);
    if (!checkError(err, "Failed to create kernel")) return false;

    return true;
}

/* GPU Matrix Multiplier */
bool GPUMatrixMultiplier::execute(const std::vector<float>& A,
                                  const std::vector<float>& B,
                                  std::vector<float>& result,
                                  unsigned int M,
                                  unsigned int N,
                                  unsigned int K) {
    cl_int err;

    // Create buffers for matrices A, B, and C
    cl_mem bufferA = clCreateBuffer(this->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A.size() * sizeof(float), const_cast<float*>(A.data()), &err);
    if (!checkError(err, "Failed to create buffer for A")) return false;

    cl_mem bufferB = clCreateBuffer(this->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B.size() * sizeof(float), const_cast<float*>(B.data()), &err);
    if (!checkError(err, "Failed to create buffer for B")) return false;

    cl_mem resultBuffer = clCreateBuffer(this->context, CL_MEM_WRITE_ONLY, result.size() * sizeof(float), nullptr, &err);
    if (!checkError(err, "Failed to create buffer for result")) return false;

    // Set kernel arguments
    err = clSetKernelArg(this->kernel, 0, sizeof(cl_mem), &bufferA);
    if (!checkError(err, "Failed to set kernel arg 0")) return false;

    err = clSetKernelArg(this->kernel, 1, sizeof(cl_mem), &bufferB);
    if (!checkError(err, "Failed to set kernel arg 1")) return false;

    err = clSetKernelArg(this->kernel, 2, sizeof(cl_mem), &resultBuffer);
    if (!checkError(err, "Failed to set kernel arg 2")) return false;

    err = clSetKernelArg(this->kernel, 3, sizeof(unsigned int), &M);
    if (!checkError(err, "Failed to set kernel arg 3")) return false;

    err = clSetKernelArg(this->kernel, 4, sizeof(unsigned int), &N);
    if (!checkError(err, "Failed to set kernel arg 4")) return false;

    err = clSetKernelArg(this->kernel, 5, sizeof(unsigned int), &K);
    if (!checkError(err, "Failed to set kernel arg 5")) return false;

    // Determine global and local work sizes
    size_t localWorkSize[2] = { 16, 16 };
    size_t globalWorkSize[2] = {
            (N + localWorkSize[0] - 1) / localWorkSize[0] * localWorkSize[0],
            (M + localWorkSize[1] - 1) / localWorkSize[1] * localWorkSize[1]
    };

    err = clEnqueueNDRangeKernel(this->commandQueue, this->kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    if (!checkError(err, "Failed to enqueue kernel")) return false;

    // Read the output data from the device
    err = clEnqueueReadBuffer(this->commandQueue, resultBuffer, CL_TRUE, 0, result.size() * sizeof(float), result.data(), 0, nullptr, nullptr);
    if (!checkError(err, "Failed to read output buffer")) return false;

    // Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(resultBuffer);
    // this->cleanUp();
    return true;
}

bool GPUMatrixMultiplier::attachKernel(const std::string &path) {
    this->readFile(path);
    return this->buildKernel("matrixMultiply");
}