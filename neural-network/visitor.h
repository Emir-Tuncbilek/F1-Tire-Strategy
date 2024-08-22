//
// Created by Emir Tuncbilek on 8/15/24.
//


#ifndef VISITOR_H
#define VISITOR_H

#include <iostream>
#include <string>
#include <utility>
#include <fstream>

#include "model.h"

enum class ExportImportTypes { WEIGHTS, BIASES, ACTIVATIONS };

class Visitor {
public:
    Visitor() = default;
    virtual ~Visitor() = default;
    virtual void doSomethingWithWeight(Model * model) = 0;
    virtual void doSomethingWithBias(Model * model) = 0;
    virtual void doSomethingWithActivations(Model * model) = 0;

protected:
    std::unique_ptr<Model> _model;
};

class ExportVisitor : public Visitor {
public:
    explicit ExportVisitor(std::string path) : _path(std::move(path)) {}
    ~ExportVisitor() = default;
    void doSomethingWithWeight(Model * model) override;
    void doSomethingWithBias(Model * model) override;
    void doSomethingWithActivations(Model * model) override;
private:
    void exportModel(const ExportImportTypes& type, Model * model);
    std::string _path;
};

class ImportVisitor : public Visitor {
public:
    explicit ImportVisitor(std::string path) : _path(std::move(path)) {}
    ~ImportVisitor() = default;
    void doSomethingWithWeight(Model * model) override;
    void doSomethingWithBias(Model * model) override;
    void doSomethingWithActivations(Model * model) override;

private:
    std::vector<Matrix> importMatrices(const std::string& filePathPrefix);
    std::vector<std::shared_ptr<ActivationFunction>> importActivationFunctions();
    std::string _path;
};

#endif
