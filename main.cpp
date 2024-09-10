/*
 *
 * The goal is to be able to stream a live F1 race and decide on the best strategies for a given car.
 * The data should contain info on FP1, FP2, FP3, Qualifying and, when applicable, sprints, in order to
 * accurately determine an optimal race strategy for Sunday. The strategy algorithm has the purpose of
 * determining when a car should stop, what tires to bolt on, and how many stops are needed.
 *
 * */

#include <iostream>
#include <chrono>
#include <OpenCL/opencl.h>

#include "./neural-network/model.h"
#include "./neural-network/test-and-gate.h"
#include "./neural-network/visitor.h"
#include "./data-interpretor/data-loader.h"


// #undef USE_GPU


int main() {

    /* Test Spanish GP tyre decay rate */

    std::pair<std::vector<float>, size_t> Xdata = DataLoader::load("../x-preprocessed-data.csv");
    std::pair<std::vector<float>, size_t> Ydata = DataLoader::load("../y-preprocessed-data.csv");
    std::cout << "x size: " << Xdata.second << std::endl << "y size: " << Ydata.second << std::endl;
    std::cout << "x length: " << Xdata.first.size() / Xdata.second << std::endl << "y length: " << Ydata.first.size() / Ydata.second << std::endl;

    auto X = DataLoader::generateVectors(Xdata);
    auto Y = DataLoader::generateVectors(Ydata);

    Model TyreModel = Model(14, TanH(0.01), std::make_unique<MSE>(0.1));
    TyreModel.addLayer(TanH(0.01), 64);
    // TyreModel.addLayer(TanH(0.01), 128);
    TyreModel.addLayer(TanH(0.01), 64);
    TyreModel.addLayer(TanH(0.01), 9);
    TyreModel.addLayer(TanH(0.01), 3);

    TyreModel.selectOptimiser(std::make_unique<RMSPROP>(0.005));
    TyreModel.trainNetwork(X, Y, 100, 3);

    TyreModel.save("file.model");
    std::cout << "Model Saved!";
}
