//
// Created by Emir Tuncbilek on 7/19/24.
//

#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <string>
#include <memory>

#include "matrix.h"
#include "activation-functions.h"
#include "loss-functions.h"
#include "layers.h"
#include "optimizers.h"
#include "GPUfunctions.h"

class Visitor;

class Model {
public:

    static std::unique_ptr<Model> importModel(const std::string& filePath);

    Model(const size_t& numberOfInputs, const ActivationFunction& activation, std::unique_ptr<LossFunction> lossFunction);
    ~Model() = default;

    void trainNetwork(const std::vector<Matrix>& inputX,
                      const std::vector<Matrix>& inputY,
                      const int& epochs,
                      const size_t& batchSize);

    void addLayer(const ActivationFunction& f, const size_t& neuronCount);

    void selectOptimiser(std::unique_ptr<Optimizer> o);

    void save(const std::string& filePath);

    Matrix predict(const Matrix& input) { return this->inputLayer->forwardFeed(input); }

    std::shared_ptr<InputLayer> getInputLayer() const { return this->inputLayer; }

private:

    void trainStochastic(const Matrix& inputX, const Matrix& inputY, const Matrix& resultY);

    void trainBatch(const std::vector<Matrix>& inputsX, const std::vector<Matrix> &inputsY, const std::vector<Matrix> &resultsY);

    void backPropagate(const std::vector<Matrix>& targetY, const std::vector<Matrix>& predictedY, const std::vector<Matrix>& inputX, const int& layerNumber);

    void calculateDels(const Matrix& targetY, const Matrix& predictedY, const Matrix& inputX, const std::shared_ptr<Layer>& currentLayer) const;
    // attributes
    std::shared_ptr<InputLayer> inputLayer;
    std::unique_ptr<LossFunction> lossFunction;
    std::shared_ptr<GPUMatrixMultiplier> gpuMatrixMultiplier;
    // std::unique_ptr<Optimizer> optimizer;
    int lastEpochNumber;
};

#endif // MODEL_H
