//
// Created by Emir Tuncbilek on 7/19/24.
//

#include "model.h"
#include "visitor.h"

/* Utility function */
std::vector<size_t> generateRandomIndices(size_t size) {
    std::vector<size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., size-1
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    return indices;
}

Model::Model(const size_t &numberOfInputs, const ActivationFunction &activation, std::unique_ptr<LossFunction> lossFunction) {
    this->inputLayer = std::make_unique<InputLayer>(activation, numberOfInputs);
    this->lastEpochNumber = -1;
    this->lossFunction = std::move(lossFunction);
    this->gpuMatrixMultiplier = std::make_shared<GPUMatrixMultiplier>();
    this->gpuMatrixMultiplier->init();
    if (!this->gpuMatrixMultiplier->attachKernel("../neural-network/gpu_kernel/matrix_mult.cl")) {
        throw std::runtime_error("Failed to attach kernel");
    }
    this->inputLayer->setMatrixMultiplier(this->gpuMatrixMultiplier);
}

void Model::addLayer(const ActivationFunction &f, const size_t &neuronCount) {
    this->inputLayer->addLayer(f, neuronCount);
    this->inputLayer->setMatrixMultiplier(this->gpuMatrixMultiplier);
}

void Model::trainNetwork(const std::vector<Matrix> &inputX, const std::vector<Matrix> &inputY,
                    const int &epochs, const size_t &batchSize) {
    if (inputX.size() != inputY.size())
        throw std::invalid_argument("InputX and InputY must be of the same length");

    double lossAtEpoch;
    if (batchSize == 1) {
        Matrix lastPrediction;
        for (int i = 0; i < epochs; i ++) {
            lossAtEpoch = 0.;
            for (int j = 0; j < inputY.size(); j ++) {
                lastPrediction = this->inputLayer->forwardFeed(inputX[j]);
                this->trainStochastic(inputX[j], inputY[j], lastPrediction);
                lossAtEpoch += this->lossFunction->loss(lastPrediction, inputY[j]);
            }
            std::cout << "Loss at Epoch " << i + 1<< " : " << lossAtEpoch / (double)inputY.size() << std::endl;
        }
    } else {
        for (int i = 0; i < epochs; i ++) {
            std::vector<size_t> indices = generateRandomIndices(inputX.size());
            lossAtEpoch = 0;
            for (size_t start = 0; start < indices.size(); start += batchSize) {
                std::vector<Matrix> batchInputsX;
                std::vector<Matrix> batchInputsY;
                std::vector<Matrix> results;

                for (size_t j = start; j < start + batchSize && j < indices.size(); j ++) {
                    batchInputsX.push_back(inputX[indices[j]]);
                    batchInputsY.push_back(inputY[indices[j]]);
                    results.push_back(this->inputLayer->forwardFeed(inputX[indices[j]]));
                }

                this->trainBatch(batchInputsX, batchInputsY, results);

                for (size_t j = 0; j < batchInputsX.size(); j ++) {
                    Matrix prediction = this->inputLayer->forwardFeed(batchInputsX[j]);
                    lossAtEpoch += this->lossFunction->loss(prediction, batchInputsY[j]);
                }
            }
            std::cout << "Loss at Epoch " << i + 1<< " : " << lossAtEpoch / static_cast<double>(inputY.size()) << std::endl;

        }
    }
}

void Model::trainStochastic(const Matrix &inputX, const Matrix &inputY, const Matrix &resultY) {
    this->backPropagate(std::vector<Matrix>(1, inputY), std::vector<Matrix>(1, resultY), std::vector<Matrix>(1, inputX), 0);
    this->inputLayer->gradientDescent(std::vector<Matrix>(1, inputX));
}

void Model::trainBatch(const std::vector<Matrix> &inputsX, const std::vector<Matrix> &inputsY, const std::vector<Matrix> &resultsY) {
    if (inputsX.size() != inputsY.size())
        throw std::invalid_argument("InputX and InputY must be of the same length");
    if (inputsX.size() != resultsY.size())
        throw std::invalid_argument("InputX and ResultsY must be of the same length");

    this->backPropagate(inputsY, resultsY, inputsX, 0);
    this->inputLayer->gradientDescent(inputsX);
}

void Model::selectOptimiser(std::unique_ptr<Optimizer> o) {
    // this->optimizer = std::move(o);
    std::shared_ptr<Layer> currentLayer = this->inputLayer;
    while (currentLayer->isNextLayer()) {
        currentLayer->setOptimizer(o->clone());
        currentLayer = currentLayer->getNextLayer();
    }
    currentLayer->setOptimizer(o->clone());
}

void Model::backPropagate(const std::vector<Matrix>& targetY, const std::vector<Matrix>& predictedY, const std::vector<Matrix>& inputX, const int& layerNumber) {

    std::shared_ptr<Layer> currentLayer = this->inputLayer;
    while (currentLayer->isNextLayer())
        currentLayer = currentLayer->getNextLayer();
    for (int i = 0; i < layerNumber; i ++)
        currentLayer = currentLayer->getPreviousLayer();

    Matrix accumulatedDels = Matrix::nullVector(currentLayer->getNeuronCount());
    for (int i = 0; i < inputX.size(); i ++) {
        this->calculateDels(targetY[i], predictedY[i], inputX[i], currentLayer);
        accumulatedDels += currentLayer->getDels();
    }

    if (inputX.size() > 1) {
        accumulatedDels *= 1.0 / (double)inputX.size();
        currentLayer->updateDels(accumulatedDels);
    }

    if (currentLayer->getPreviousLayer()) this->backPropagate(targetY, predictedY, inputX, layerNumber + 1);
    else return;
}

void Model::save(const std::string& filePath) {
    ExportVisitor visitor(filePath);
    visitor.doSomethingWithWeight(this);
    visitor.doSomethingWithBias(this);
    visitor.doSomethingWithActivations(this);
}

std::unique_ptr<Model> Model::importModel(const std::string &filePath) {
    std::unique_ptr<Model> m = std::make_unique<Model>(1, NoActivation(), std::make_unique<MSE>(0.01)) ;
    ImportVisitor visitor(filePath);
    visitor.doSomethingWithWeight(&*m);
    visitor.doSomethingWithBias(&*m);
    visitor.doSomethingWithActivations(&*m);
    return m;
}

void Model::calculateDels(const Matrix &targetY, const Matrix &predictedY, const Matrix &inputX,
                          const std::shared_ptr<Layer>& currentLayer) const {
    if (!currentLayer->isNextLayer()) {
        std::vector<std::unique_ptr<std::vector<double>>> newDelVals(currentLayer->getNeuronCount());
        double delVal;
        for (int i = 0; i < currentLayer->getNeuronCount(); i ++) {
            this->lossFunction->setWeightsSquaredSum(currentLayer->getWeight().toVector());
            delVal = this->lossFunction->derivative(predictedY[i][0], targetY[i][0]) *
                     currentLayer->getActivationDerivative(
                             this->inputLayer->forwardFeedUntilLayer(inputX, currentLayer->getLayerNumber())[i][0]
                     );
            newDelVals[i] = std::make_unique<std::vector<double>>(1, delVal);
        }
        currentLayer->updateDels(Matrix(std::move(newDelVals)));
    }
    else {
        double sumOfNextDels = currentLayer->getNextLayer()->getSumDels();
        std::vector<std::unique_ptr<std::vector<double>>> delValsForLayer(currentLayer->getNeuronCount());

        Matrix activations = this->inputLayer->forwardFeedUntilLayer(inputX, currentLayer->getLayerNumber());
        double columnSumOfWeights, delValue, derivative;
        for (int i = 0; i < currentLayer->getNeuronCount(); i ++) {
            columnSumOfWeights = currentLayer->getNextLayer()->getColumnSumWeights(i);
            delValue = sumOfNextDels * columnSumOfWeights;
            derivative = currentLayer->getActivationDerivative(activations[i][0]);
            delValue *= derivative;
            delValsForLayer[i] = std::make_unique<std::vector<double>>(1, delValue);
        }
        currentLayer->updateDels(Matrix(std::move(delValsForLayer)));
    }
}
