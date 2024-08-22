//
// Created by Emir Tuncbilek on 7/16/24.
//

#include "loss-functions.h"

double LossFunction::l2Penalty() const {
    return this->_lambda * this->squaredSumWeights;
}

void LossFunction::setWeightsSquaredSum(const std::vector<float>& weights) {
    this->squaredSumWeights = 0.;
    for (const auto &weight : weights) {
        this->squaredSumWeights += weight * weight;
    }
}


double MSE::loss(const Matrix &predicted, const Matrix &targetY) {
    if (predicted.getColumnSize() != 1 || targetY.getColumnSize() != 1) {
        throw std::invalid_argument("Predicted and Target must be vectors!");
    }
    const size_t N = predicted.getRowSize();
    auto square = [](double x) { return x * x; };
    const double sumSquaredDiff = (predicted - targetY).map(square).sum();
    return (sumSquaredDiff / (double)N);
}

double MSE::derivative(const double &number, const double &targetY) {
    return 2 * (targetY - number) + this->l2Penalty();
}

double MAE::loss(const Matrix &predicted, const Matrix &targetY) {
    if (predicted.getColumnSize() != 1 || targetY.getColumnSize() != 1) {
        throw std::invalid_argument("Predicted and Target must be vectors!");
    }
    const size_t N = predicted.getRowSize();
    auto square = [](double x) { return std::abs(x); };
    const double sumSquaredDiff = (predicted - targetY).map(square).sum();
    return (sumSquaredDiff / (double)N);
}

double MAE::derivative(const double &number, const double &targetY) {
    return ((targetY - number) / std::abs(targetY - number)) + this->l2Penalty();
}

