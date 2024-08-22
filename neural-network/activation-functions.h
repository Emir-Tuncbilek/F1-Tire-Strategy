//
// Created by Emir Tuncbilek on 7/16/24.
//

#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <iostream>
#include <string>
#include <ostream>

#include "./matrix.h"

class ActivationFunction {
public:
    ActivationFunction() = default;
    ~ActivationFunction() = default;
    ActivationFunction(const ActivationFunction& o) = default;
    virtual Matrix function(const Matrix& inputs) = 0;
    virtual double derivative(const double& input) = 0;
    virtual std::unique_ptr<ActivationFunction> clone() const = 0;
    friend std::ostream& operator << (std::ostream& o, const ActivationFunction& f);
    virtual void print(std::ostream& o) const = 0;
};

class NoActivation: public ActivationFunction {
public:
    NoActivation() = default;
    ~NoActivation() = default;
    NoActivation(const NoActivation& o) = default;
    Matrix function(const Matrix& inputs) override;
    double derivative(const double& input) override;
    std::unique_ptr<ActivationFunction> clone() const override;
private:
    void print(std::ostream& o) const override;
};

class ReLU: public ActivationFunction {
public:
    ReLU() = default;
    ~ReLU() = default;
    Matrix function(const Matrix& inputs) override;
    double derivative(const double& input) override;
    std::unique_ptr<ActivationFunction> clone() const override;
private:
    void print(std::ostream& o) const override;};

class LeakyReLU: public ActivationFunction {
public:
    explicit LeakyReLU(const double& alpha): _alpha(alpha) {}
    ~LeakyReLU() = default;
    Matrix function(const Matrix& inputs) override;
    double derivative(const double& input) override;
    std::unique_ptr<ActivationFunction> clone() const override;
private:
    void print(std::ostream& o) const override;
    const double _alpha;
};

class ELU: public ActivationFunction {
public:
    explicit ELU(const double& alpha): _alpha(alpha) {}
    ~ELU() = default;
    Matrix function(const Matrix& inputs) override;
    double derivative(const double& input) override;
    std::unique_ptr<ActivationFunction> clone() const override;
private:
    void print(std::ostream& o) const override;
    const double _alpha;
};

class TanH: public ActivationFunction {
public:
    explicit TanH(const double& alpha): _alpha(alpha) {}
    ~TanH() = default;
    Matrix function(const Matrix& inputs) override;
    double derivative(const double& input) override;
    std::unique_ptr<ActivationFunction> clone() const override;

private:
    void print(std::ostream& o) const override;
    const double _alpha;
};

class Sigmoid: public ActivationFunction {
public:
    Sigmoid() = default;
    ~Sigmoid() = default;
    Matrix function(const Matrix& inputs) override;
    double derivative(const double& input) override;
    std::unique_ptr<ActivationFunction> clone() const override;
private:
    void print(std::ostream& o) const override;
};

#endif // ACTIVATION_FUNCTIONS_H
