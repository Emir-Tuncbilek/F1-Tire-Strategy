//
// Created by Emir Tuncbilek on 7/16/24.
//

#include "activation-functions.h"

std::ostream& operator << (std::ostream& o, const ActivationFunction& f) {
    f.print(o);
    return o;
}


/* No Activation */
Matrix NoActivation::function(const Matrix &inputs) { return inputs; }

double NoActivation::derivative(const double &input) { return input; }

std::unique_ptr<ActivationFunction> NoActivation::clone() const {
    return std::unique_ptr<ActivationFunction>(std::make_unique<NoActivation>(*this).release());
}

void NoActivation::print(std::ostream& o) const {
    o << "NoActivation";
}

/* Rectified linear (ReLU) */
Matrix ReLU::function(const Matrix &inputs) {
    auto f = [](double x) { return std::max(0.0, x); };
    return inputs.map(f);
}

double ReLU::derivative(const double &input) {
    return (double)(input > 0);
}

std::unique_ptr<ActivationFunction> ReLU::clone() const {
    return std::unique_ptr<ActivationFunction>(std::make_unique<ReLU>(*this).release());
}

void ReLU::print(std::ostream& o) const {
    o << "ReLU";
}

/* Leaky Rectified Linear (Leaky ReLU) */
Matrix LeakyReLU::function(const Matrix &inputs) {
    auto f = [this](double x) { return x >= 0 ? x : this->_alpha * x; };
    return inputs.map(f);
}

double LeakyReLU::derivative(const double &input) {
    return input < 0 ? this->_alpha : (double) input != 0.;
}

std::unique_ptr<ActivationFunction> LeakyReLU::clone() const {
    return std::unique_ptr<ActivationFunction>(std::make_unique<LeakyReLU>(*this).release());
}

void LeakyReLU::print(std::ostream& o) const {
    o << "LeakyReLU, alpha = " << this->_alpha;
}

/* Exponential Linear Unit (ELU) */
Matrix ELU::function(const Matrix &inputs) {
    auto f = [this](double x) { return x >= 0 ? x : this->_alpha * (std::exp(x) - 1); };
    return inputs.map(f);
}

double ELU::derivative(const double &input) {
    return input < 0 ? this->_alpha * std::exp(input) : 1.;
}

std::unique_ptr<ActivationFunction> ELU::clone() const {
    return std::unique_ptr<ActivationFunction>(std::make_unique<ELU>(*this).release());
}

void ELU::print(std::ostream& o) const {
    o << "ELU, alpha = " << this->_alpha;
}

/* Tanh */
Matrix TanH::function(const Matrix &inputs) {
    auto f = [](double x) { return std::tanh(x); };
    return inputs.map(f);
}

double TanH::derivative(const double &input) {
    const double tanh = (std::exp(input) - std::exp(-input)) / (std::exp(input) + std::exp(-input));
    return 1 - (tanh * tanh);
}

std::unique_ptr<ActivationFunction> TanH::clone() const {
    return std::unique_ptr<ActivationFunction>(std::make_unique<TanH>(*this).release());
}

void TanH::print(std::ostream& o) const {
    o << "TanH, alpha = " << this->_alpha;
}

/* Sigmoid */
Matrix Sigmoid::function(const Matrix &inputs) {
    auto f = [](double x) { return 1 / (1 + std::exp(-x)); };
    return inputs.map(f);
}

double Sigmoid::derivative(const double &input) {
    const double sigmoidBase = 1 / (1 + std::exp(-input));
    return sigmoidBase * (1 - sigmoidBase);
}

std::unique_ptr<ActivationFunction> Sigmoid::clone() const {
    return std::unique_ptr<ActivationFunction>(std::make_unique<Sigmoid>(*this).release());
}

void Sigmoid::print(std::ostream& o) const {
    o << "Sigmoid";
}