//
// Created by Emir Tuncbilek on 7/15/24.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>

#include "./activation-functions.h"
#include "./loss-functions.h"
#include "./matrix.h"
#include "./optimizers.h"
#include "./GPUfunctions.h"

class OutputLayer;

class Layer : public std::enable_shared_from_this<Layer> {
public:
    Layer(const ActivationFunction& f,
          const size_t& neuronCount,
          const size_t& activationCount,
          const int& layerNumber);
    ~Layer() = default;

    Layer(const Layer& other) = default;

    void setMatrixMultiplier(const std::shared_ptr<GPUMatrixMultiplier>& f);

    Matrix output(const Matrix& input);

    Matrix forwardFeed(const Matrix& input);

    void addLayer(const ActivationFunction& activation, const size_t& neuronCount);

    Matrix forwardFeedUntilLayer(const Matrix& input, const int& layerNumber);

    bool isNextLayer() const { return this->hasNextLayer; }

    int getLayerNumber() const { return this->layerNumber; }

    Matrix getWeight() const { return *this->weights; }

    Matrix getBiases() const { return *this->biases; }

    std::shared_ptr<ActivationFunction> getActivation() { return this->activation; }

    void setWeights(const Matrix& weight) {
        this->neuronCount = weight.getRowSize();
        this->activationCount = this->layerNumber == 0 ? 0 : weight.getColumnSize();
        this->weights = std::make_unique<Matrix>(weight);

    }

    void setBiases(const Matrix& bias) { this->biases = std::make_unique<Matrix>(bias); }

    void setActivationFunction(std::shared_ptr<ActivationFunction> f) { this->activation = f; }

    std::shared_ptr<OutputLayer> getNextLayer() const { return this->nextLayer; }

    size_t getNeuronCount() const { return this->neuronCount; }

    void updateDels(const Matrix& newValues) {
        this->delValues = std::make_unique<Matrix>(newValues);
        this->delValues->setGPUMatrixMult(this->gpuMatrixMultFunction);
    }

    double getSumDels() const { return this->delValues->sum(); }

    Matrix& getDels() const { return *this->delValues; }

    double getColumnSumWeights(const size_t& column) const;

    double getActivationDerivative(double value) const { return this->activation->derivative(value); };

    virtual std::shared_ptr<Layer> getPreviousLayer() const = 0;

    void setOptimizer(std::unique_ptr<Optimizer> o);

    void gradientDescent(const std::vector<Matrix>& inputs);


protected:
    virtual void initialize() = 0;
    bool hasNextLayer;
    int layerNumber;
    size_t neuronCount;
    size_t activationCount;
    std::shared_ptr<ActivationFunction> activation;
    std::unique_ptr<Matrix> weights;            // an N x M matrix, where N is neuron count and M is activation count
    std::unique_ptr<Matrix> biases;             // an N x 1 matrix, where N is neuron count
    std::unique_ptr<Matrix> delValues;          // an N x 1 matrix, where N is neuron count
    std::shared_ptr<Layer> previousLayer;
    std::shared_ptr<OutputLayer> nextLayer;
    std::unique_ptr<Optimizer> optimizer;
    std::shared_ptr<GPUMatrixMultiplier> gpuMatrixMultFunction;
};

class InputLayer : public Layer {
public:
    InputLayer(const ActivationFunction& f, const size_t& neuronCount) :
            Layer(f, neuronCount, 0, 0) {
        this->initialize(); // function that initializes the weights and biases
    }
    ~InputLayer() = default;

    std::shared_ptr<Layer> getPreviousLayer() const override { return nullptr; }

protected:
    void initialize() override;
};

class OutputLayer : public Layer {
public:
    OutputLayer(const ActivationFunction& f,
                const size_t& neuronCount,
                const size_t& activationCount,
                const int& layerNumber) :
            Layer(f, neuronCount, activationCount, layerNumber) {
        this->initialize(); // function that initializes the weights and biases
    }

    ~OutputLayer() = default;

    void setPreviousLayer(const std::shared_ptr<Layer>& layer);

    std::shared_ptr<Layer> getPreviousLayer() const override { return this->previousLayer; }

protected:
    void initialize() override;
};

#endif
