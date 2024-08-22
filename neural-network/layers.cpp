//
// Created by Emir Tuncbilek on 7/15/24.
//

#include "./layers.h"
#include "./optimizers.h"

#define LARGEST_SIGNED_INT 0x7fffffff
#define SCALE_INIT_FACTOR 1

/* Generic Layer implementation */
Layer::Layer(const ActivationFunction &f, const size_t &neuronCount, const size_t &activationCount,
             const int &layerNumber) {
    this->activation = std::move(f.clone());
    this->layerNumber = layerNumber;
    this->neuronCount = neuronCount;
    this->activationCount = activationCount;
    std::vector<std::unique_ptr<std::vector<double>>> delVals(neuronCount);
    for (int i = 0; i < neuronCount; i ++) {
        delVals[i] = std::make_unique<std::vector<double>>(1, 0);
    }
    this->delValues = std::make_unique<Matrix>(std::move(delVals));
}

void Layer::setMatrixMultiplier(const std::shared_ptr<GPUMatrixMultiplier> &f) {
    this->gpuMatrixMultFunction = f;
    this->delValues->setGPUMatrixMult(this->gpuMatrixMultFunction);
    this->weights->setGPUMatrixMult(this->gpuMatrixMultFunction);
    this->biases->setGPUMatrixMult(this->gpuMatrixMultFunction);
    if (!this->hasNextLayer)
        return;
    std::shared_ptr<OutputLayer> currentLayer = this->nextLayer;
    while (currentLayer != nullptr) {
        currentLayer->gpuMatrixMultFunction = f;
        currentLayer->delValues->setGPUMatrixMult(this->gpuMatrixMultFunction);
        currentLayer->weights->setGPUMatrixMult(this->gpuMatrixMultFunction);
        currentLayer->biases->setGPUMatrixMult(this->gpuMatrixMultFunction);
        currentLayer = currentLayer->nextLayer;
    }

}

Matrix Layer::output(const Matrix &input) {
    return this->activation->function(*this->weights * input + *this->biases);
}

Matrix Layer::forwardFeed(const Matrix &input) {
    /*
     * layer number is the largest signed int possible (32 bits), practically no one should
     * ever get close to that number, even less so, pass it. This ensures no code duplication
     * and easy maintenance, by exploiting the fact that you would meet the last layer well
     * before meeting the limit. There is only one extra operation performed, but it shouldn't
     * lead to too much overhead.
     * */
    return this->forwardFeedUntilLayer(input, LARGEST_SIGNED_INT);
}

void Layer::addLayer(const ActivationFunction &activation, const size_t &neuronCount) {
    if (this->hasNextLayer) this->nextLayer->addLayer(activation, neuronCount);
    else {
        this->hasNextLayer = true;
        this->nextLayer = std::make_shared<OutputLayer>(activation,
                                                        neuronCount,
                                                        this->neuronCount,
                                                        this->layerNumber + 1 );

        this->nextLayer->setPreviousLayer(shared_from_this());
    }
}

double Layer::getColumnSumWeights(const size_t& column) const {
    return this->weights->getColumn(column).sum();
}

Matrix Layer::forwardFeedUntilLayer(const Matrix &input, const int& layerNumber) {
    if (this->hasNextLayer && this->layerNumber != layerNumber) return this->nextLayer->forwardFeedUntilLayer(this->output(input), layerNumber);
    else return this->output(input);
}

void Layer::setOptimizer(std::unique_ptr<Optimizer> o) {
    this->optimizer = o->clone();
}

void Layer::gradientDescent(const std::vector<Matrix>& inputs) {
    std::shared_ptr<Layer> currentLayer = shared_from_this();
    while (currentLayer->isNextLayer()) {
        currentLayer = currentLayer->nextLayer;
    }

    while (currentLayer->layerNumber > 0) {
        Matrix accumulatedGradients = Matrix::nullMatrix(currentLayer->weights->getRowSize(), currentLayer->weights->getColumnSize());
        Matrix accumulatedDels = Matrix::nullMatrix(currentLayer->biases->getRowSize(), 1);

        for (const Matrix& input : inputs) {
            const Matrix activations = this->forwardFeedUntilLayer(input, currentLayer->layerNumber - 1);
            const Matrix dels = currentLayer->delValues->clone();

            accumulatedGradients += dels * activations.transpose();
            accumulatedDels += dels;
        }

        const auto f = [currentLayer](double x) { return x * -currentLayer->optimizer->getLearningRate(); };

        currentLayer->optimizer->updateWeights(*currentLayer->weights, accumulatedGradients.map(f));
        currentLayer->optimizer->updateBiases(*currentLayer->biases, accumulatedDels.map(f));

        currentLayer = currentLayer->previousLayer;
    }
}

/* Input Layer */

void InputLayer::initialize() {
    /*
     * The input layer has all its weights set to '1' and all biases are null. Therefore,
     * the weight matrix is the identity matrix, and the biases is a null vector.
     * */
    this->weights = std::make_unique<Matrix>(Matrix::identity(this->neuronCount));
    this->biases = std::make_unique<Matrix>(Matrix::nullVector(this->neuronCount));
}

/* Output or Hidden Layer */

void OutputLayer::initialize() {
    /*
     * The output layer has all its weights set to randomly, as it is susceptible to change.
     * Therefore, the weight matrix is a random matrix, and the biases is a random vector.
     * */
    this->weights = std::make_unique<Matrix>(Matrix::randomMatrix(this->neuronCount, this->activationCount));
    this->biases = std::make_unique<Matrix>(Matrix::randomVector(this->neuronCount));
    *this->weights *= SCALE_INIT_FACTOR;
    *this->biases *= SCALE_INIT_FACTOR;
}

void OutputLayer::setPreviousLayer(const std::shared_ptr<Layer> &layer) { this->previousLayer = layer; }

