//
// Created by Emir Tuncbilek on 7/16/24.
//

#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <iostream>

#include "./matrix.h"



class LossFunction {
public:
    explicit LossFunction(const float& lambda): _lambda(lambda) {}
    ~LossFunction() = default;
    /* predicted and targetY are both vectors of size N */
    virtual double loss(const Matrix& predicted, const Matrix& targetY) = 0;
    virtual double derivative(const double& number, const double& targetY) = 0;
    /* L2 regularization method */
    double l2Penalty() const;
    void setWeightsSquaredSum(const std::vector<float>& weights);
protected:
    double squaredSumWeights;
    float _lambda;
};

/* Mean Square Error */
class MSE : public LossFunction {
public:
    explicit MSE(const float& lambda): LossFunction(lambda) {};
    ~MSE() = default;
    double loss(const Matrix &predicted, const Matrix &targetY) override;
    double derivative(const double &number, const double &targetY) override;
};

/* Mean Absolute Error */
class MAE : public LossFunction {
public:
    explicit MAE(const float& lambda): LossFunction(lambda) {};
    ~MAE() = default;
    double loss(const Matrix &predicted, const Matrix &targetY) override;
    double derivative(const double &number, const double &targetY) override;
};



#endif // LOSS_FUNCTIONS_H
