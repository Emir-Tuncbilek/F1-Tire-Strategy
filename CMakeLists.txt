cmake_minimum_required(VERSION 3.24)
project(F1_STRATEGIES)

set(CMAKE_CXX_STANDARD 20)

find_library(OpenCL_LIBRARY OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_definitions(USE_GPU=0) # set to 0 to use CPU computing and 1 for GPU computing


add_executable(F1_STRATEGIES main.cpp neural-network/layers.cpp neural-network/layers.h neural-network/activation-functions.cpp neural-network/activation-functions.h neural-network/matrix.cpp neural-network/matrix.h neural-network/loss-functions.cpp neural-network/loss-functions.h neural-network/model.cpp neural-network/model.h neural-network/test-and-gate.h neural-network/optimizers.cpp neural-network/optimizers.h neural-network/env.h neural-network/GPUfunctions.cpp neural-network/GPUfunctions.h data-interpretor/data-loader.cpp data-interpretor/data-loader.h neural-network/visitor.cpp neural-network/visitor.h)
add_executable(F1_STRATEGIES_RUN predict.cpp neural-network/layers.cpp neural-network/layers.h neural-network/activation-functions.cpp neural-network/activation-functions.h neural-network/matrix.cpp neural-network/matrix.h neural-network/loss-functions.cpp neural-network/loss-functions.h neural-network/model.cpp neural-network/model.h neural-network/test-and-gate.h neural-network/optimizers.cpp neural-network/optimizers.h neural-network/env.h neural-network/GPUfunctions.cpp neural-network/GPUfunctions.h data-interpretor/data-loader.cpp data-interpretor/data-loader.h neural-network/visitor.cpp neural-network/visitor.h)

target_link_libraries(F1_STRATEGIES ${OpenCL_LIBRARY})
target_link_libraries(F1_STRATEGIES_RUN ${OpenCL_LIBRARY})

