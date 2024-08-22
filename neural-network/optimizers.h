//
// Created by Emir Tuncbilek on 7/24/24.
//

#ifndef F1_STRATEGIES_OPTIMIZERS_H
#define F1_STRATEGIES_OPTIMIZERS_H

#include "./matrix.h"

/* RMSPROP MACROS */
#define RMSPROP_EPSILON 1e-8
#define RMSPROP_DECAY_RATE 0.9
/* ADAM MACROS */
#define ADAM_EPSILON 1e-8
#define ADAM_DECAY_RATE_1 0.9
#define ADAM_DECAY_RATE_2 0.999
/* ADAGRAD MACROS */
#define ADAGRAD_EPSILON 1e-8
/* ADADelta MACROS */
#define ADA_DELTA_EPSILON 1e-6
#define ADA_DELTA_DECAY_RATE 0.9


class Optimizer {
public:
    Optimizer(const double& _learningRate) : learningRate(_learningRate) {};
    ~Optimizer() = default;

    double& getLearningRate() { return this->learningRate; }

    virtual void updateWeights(Matrix& weights, const Matrix& gradients) const = 0;
    virtual void updateBiases(Matrix& biases, const Matrix& gradients) const = 0;
    virtual std::unique_ptr<Optimizer> clone() const = 0;
protected:
    double learningRate;
};

class NoOptimization : public Optimizer {
    /* A.K.A Stochastic gradient descent (SGD) */
public:
    explicit NoOptimization(const double& _learningRate): Optimizer(_learningRate) {};
    ~NoOptimization() = default;
    void updateWeights(Matrix &weights, const Matrix &gradients) const override;
    void updateBiases(Matrix &biases, const Matrix &gradients) const override;
    std::unique_ptr<Optimizer> clone() const override;

};

class RMSPROP : public Optimizer {
public:
    explicit RMSPROP(const double& _learningRate) : Optimizer(_learningRate) {}
    ~RMSPROP() = default;
    void updateWeights(Matrix &weights, const Matrix &gradients) const override;
    void updateBiases(Matrix &biases, const Matrix &gradients) const override;
    std::unique_ptr<Optimizer> clone() const override;

private:
    mutable Matrix weightGradCache, biasGradCache;
};

class ADAM : public Optimizer {
public:
    explicit ADAM(const double& _learningRate) : t(0), Optimizer(_learningRate) {};
    ~ADAM() = default;
    void updateWeights(Matrix &weights, const Matrix &gradients) const override;
    void updateBiases(Matrix &biases, const Matrix &gradients) const override;
    std::unique_ptr<Optimizer> clone() const override;

private:
    mutable int t;
    mutable Matrix m, v, mb, vb;
};

class ADAGRAD : public Optimizer {
public:
    explicit ADAGRAD(const double& _learningRate) : Optimizer(_learningRate) {}
    ~ADAGRAD() = default;
    void updateWeights(Matrix &weights, const Matrix &gradients) const override;
    void updateBiases(Matrix &biases, const Matrix &gradients) const override;
    std::unique_ptr<Optimizer> clone() const override;

private:
    mutable Matrix weightCache;
    mutable Matrix biasCache;
};

class ADADelta : public Optimizer {
public:
    ADADelta(const double& learningRate) : Optimizer(learningRate) {}
    ~ADADelta() = default;
    void updateWeights(Matrix &weights, const Matrix &gradients) const override;
    void updateBiases(Matrix &biases, const Matrix &gradients) const override;
    std::unique_ptr<Optimizer> clone() const override;

private:
    mutable Matrix weightGradientCache, weightUpdateCache, biasGradientCache, biasUpdateCache;
};

#endif //F1_STRATEGIES_OPTIMIZERS_H
