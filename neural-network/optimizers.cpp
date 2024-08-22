//
// Created by Emir Tuncbilek on 7/24/24.
//

#include "optimizers.h"

/* No optimization */
void NoOptimization::updateWeights(Matrix &weights, const Matrix &gradients) const {
    weights -= gradients.map([this](double x) { return x * this->learningRate; });
}

void NoOptimization::updateBiases(Matrix &biases, const Matrix &gradients) const {
    biases -= gradients.map([this](double x) { return x * -this->learningRate; });
}

std::unique_ptr<Optimizer> NoOptimization::clone() const {
    return std::make_unique<NoOptimization>(this->learningRate);
}

/* RMSPROP */
void RMSPROP::updateWeights(Matrix &weights, const Matrix &gradients) const {
    if (!this->weightGradCache.getRowSize()) {
        this->weightGradCache = Matrix::nullMatrix(gradients.getRowSize(), gradients.getColumnSize());
    }
    this->weightGradCache = this->weightGradCache.map(gradients, [this](double cache, double grad) {
        return RMSPROP_DECAY_RATE * cache + (1 - RMSPROP_DECAY_RATE) * grad * grad;
    });
    weights -= gradients.map(this->weightGradCache , [this](double grad, double cache) {
        return this->learningRate * grad / (std::sqrt(cache) + RMSPROP_EPSILON);
    });
}

void RMSPROP::updateBiases(Matrix &biases, const Matrix &gradients) const {
    if (!this->biasGradCache.getRowSize()) {
        this->biasGradCache = Matrix::nullVector(biases.getRowSize());
    }
    this->biasGradCache = this->biasGradCache .map(gradients, [this](double cache, double grad) {
        return RMSPROP_DECAY_RATE * cache + (1 - RMSPROP_DECAY_RATE) * grad * grad;
    });
    biases -= gradients.map(this->biasGradCache , [this](double grad, double cache) {
        return this->learningRate * grad / (std::sqrt(cache) + RMSPROP_EPSILON);
    });
}

std::unique_ptr<Optimizer> RMSPROP::clone() const {
    return std::make_unique<RMSPROP>(this->learningRate);
}

/* ADAM */
void ADAM::updateWeights(Matrix &weights, const Matrix &gradients) const {
    if (!m.getRowSize()) {
        m = Matrix::nullMatrix(weights.getRowSize(), weights.getColumnSize());
        v = Matrix::nullMatrix(weights.getRowSize(), weights.getColumnSize());
    }
    t++;
    m = m * ADAM_DECAY_RATE_1 + gradients * (1 - ADAM_DECAY_RATE_1);
    v = v * ADAM_DECAY_RATE_2 + gradients.map([](double x) { return x * x; }) * (1 - ADAM_DECAY_RATE_2);

    Matrix mHat = m.map([this](double x) { return x / (1 - std::pow(ADAM_DECAY_RATE_1, t)); });
    Matrix vHat = v.map([this](double x) { return x / (1 - std::pow(ADAM_DECAY_RATE_2, t)); });

    weights -= mHat.map(vHat, [this](double x, double cache) {
        return learningRate * x / (std::sqrt(cache) + ADAM_EPSILON);
    });
}

void ADAM::updateBiases(Matrix &biases, const Matrix &gradients) const {
    if (!mb.getRowSize()) {
        mb = Matrix::nullVector(biases.getRowSize());
        vb = Matrix::nullVector(biases.getRowSize());
    }
    t++;
    mb = mb * ADAM_DECAY_RATE_1 + gradients * (1 - ADAM_DECAY_RATE_1);
    vb = vb * ADAM_DECAY_RATE_2 + gradients.map([](double x) { return x * x; }) * (1 - ADAM_DECAY_RATE_2);


    Matrix mHat = mb.map([this](double x) { return x / (1 - std::pow(ADAM_DECAY_RATE_1, t)); });
    Matrix vHat = vb.map([this](double x) { return x / (1 - std::pow(ADAM_DECAY_RATE_2, t)); });

    biases -= mHat.map(vHat, [this](double x, double cache) {
        return learningRate * x / (std::sqrt(cache) + ADAM_EPSILON);
    });
}

std::unique_ptr<Optimizer> ADAM::clone() const {
    return std::make_unique<ADAM>(this->learningRate);
}

/* ADAGRAD */

void ADAGRAD::updateWeights(Matrix &weights, const Matrix &gradients) const {
    if (!this->weightCache.getRowSize())
        this->weightCache = Matrix::nullMatrix(weights.getRowSize(), weights.getColumnSize());

    this->weightCache += gradients.map([](const double& x) { return x * x; });

    weights -= gradients.map(this->weightCache, [this](const double& grad, const double& cache) {
        return this->learningRate * grad / (std::sqrt(cache) + ADAGRAD_EPSILON);
    });
}

void ADAGRAD::updateBiases(Matrix &biases, const Matrix &gradients) const {
    if (!this->biasCache.getRowSize())
        this->biasCache = Matrix::nullVector(biases.getRowSize());

    this->biasCache += gradients.map([](const double& x) { return x * x; });

    biases -= gradients.map(this->biasCache, [this](const double& grad, const double& cache) {
        return this->learningRate * grad / (std::sqrt(cache) + ADAGRAD_EPSILON);    });
}

std::unique_ptr<Optimizer> ADAGRAD::clone() const {
    return std::make_unique<ADAGRAD>(this->learningRate);
}

/* ADADelta */
void ADADelta::updateWeights(Matrix &weights, const Matrix &gradients) const {
    if (!this->weightGradientCache.getRowSize()) {
        this->weightGradientCache = Matrix::nullMatrix(weights.getRowSize(), weights.getColumnSize());
        this->weightUpdateCache = Matrix::nullMatrix(weights.getRowSize(), weights.getColumnSize());
    }

    this->weightGradientCache = this->weightGradientCache.map(gradients, [](double cache, double grad) {
        return (ADA_DELTA_DECAY_RATE * cache) + (1 - ADA_DELTA_DECAY_RATE) * grad * grad;
    });

    /* Roots Mean Squared (RMS) */
    Matrix rmsGradient = this->weightGradientCache.map([](const double& x){
        return std::sqrt(x + ADA_DELTA_EPSILON);
    });

    Matrix rmsUpdate = this->weightUpdateCache.map([](const double& x){
        return std::sqrt(x + ADA_DELTA_EPSILON);
    });

    Matrix update = rmsUpdate.map(rmsGradient, [](const double& update, const double& gradient) {
        return -update / gradient;
    });

    update = update.map(gradients, [](const double& update, const double& grad){
        return update * grad;
    });

    this->weightUpdateCache = this->weightUpdateCache.map(update, [](const double& cache, const double& update) {
        return (ADA_DELTA_DECAY_RATE * cache) + (1 - ADA_DELTA_DECAY_RATE) * update * update;
    });

    weights += update;
}

void ADADelta::updateBiases(Matrix &biases, const Matrix &gradients) const {
    if (!this->biasGradientCache.getRowSize()) {
        this->biasGradientCache = Matrix::nullVector(biases.getRowSize());
        this->biasUpdateCache = Matrix::nullVector(biases.getRowSize());
    }

    this->biasGradientCache = this->biasGradientCache.map(gradients, [](double cache, double grad) {
        return (ADA_DELTA_DECAY_RATE * cache) + (1 - ADA_DELTA_DECAY_RATE) * grad * grad;
    });

    /* Roots Mean Squared (RMS) */
    Matrix rmsGradient = this->biasGradientCache.map([](const double& x){
        return std::sqrt(x + ADA_DELTA_EPSILON);
    });

    Matrix rmsUpdate = this->biasUpdateCache.map([](const double& x){
        return std::sqrt(x + ADA_DELTA_EPSILON);
    });

    Matrix update = rmsUpdate.map(rmsGradient, [](const double& update, const double& gradient) {
        return -update / gradient;
    });

    update = update.map(gradients, [](const double& update, const double& grad){
        return update * grad;
    });

    this->biasUpdateCache = this->biasUpdateCache.map(update, [](const double& cache, const double& update) {
        return (ADA_DELTA_DECAY_RATE * cache) + (1 - ADA_DELTA_DECAY_RATE) * update * update;
    });

    biases += update;
}

std::unique_ptr<Optimizer> ADADelta::clone() const {
    return std::make_unique<ADADelta>(this->learningRate);
}