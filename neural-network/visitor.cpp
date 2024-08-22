//
// Created by Emir Tuncbilek on 8/15/24.
//

#include "visitor.h"

std::vector<float> readMatrixLine(const std::string& s) {
    std::vector<float> result = {};
    std::istringstream iss(s);
    std::string number;
    while(iss >> number) {
        result.push_back(std::stof(number));
    }
    return result;
}

std::string exportImportTypesToStr(const ExportImportTypes& type) {
    switch (type) {
        case ExportImportTypes::WEIGHTS:
            return "weights";
        case ExportImportTypes::BIASES:
            return "biases";
        case ExportImportTypes::ACTIVATIONS:
            return "activations";
    }
}

double findAlpha(const std::string& s) {
    size_t pos = s.find('=');
    if (pos == std::string::npos) throw  std::invalid_argument("Format error");
    return std::stof(s.substr(pos + 1));
}

std::shared_ptr<ActivationFunction> readActivationType(const std::string& s) {
    if (s.starts_with("NoActivation"))
        return std::make_shared<NoActivation>();
    else if (s.starts_with("ReLU"))
        return std::make_shared<ReLU>();
    else if (s.starts_with("LeakyReLU")) {
        double alpha = findAlpha(s);
        return std::make_shared<LeakyReLU>(alpha);
    } else if (s.starts_with("ELU")) {
        double alpha = findAlpha(s);
        return std::make_shared<ELU>(alpha);
    }
    else if (s.starts_with("TanH")) {
        double alpha = findAlpha(s);
        return std::make_shared<TanH>(alpha);
    } else if (s.starts_with("Sigmoid")) {
        return std::make_shared<Sigmoid>();
    }
    return nullptr;
}

/* Export */
void ExportVisitor::doSomethingWithWeight(Model * model) {
    this->exportModel(ExportImportTypes::WEIGHTS, model);
}

void ExportVisitor::doSomethingWithBias(Model *model) {
    this->exportModel(ExportImportTypes::BIASES, model);
}

void ExportVisitor::doSomethingWithActivations(Model *model) {
    this->exportModel(ExportImportTypes::ACTIVATIONS, model);
}

void ExportVisitor::exportModel(const ExportImportTypes& type, Model * model) {
    std::ofstream file;
    file.open("../" + exportImportTypesToStr(type) + "_" + this->_path);
    if (file.is_open()) {
        std::shared_ptr<Layer> layers = model->getInputLayer();
        while (layers) {
            file << "Layer #" << layers->getLayerNumber() << ":\n";
            switch (type) {
                case ExportImportTypes::WEIGHTS:
                    file << layers->getWeight();
                    break;
                case ExportImportTypes::BIASES:
                    file << layers->getBiases();
                    break;
                case ExportImportTypes::ACTIVATIONS:
                    file << *layers->getActivation();
                    break;
            }
            file << std::endl;
            layers = layers->getNextLayer();
        }
        file.close();
    } else {
        std::cerr << "Unable to open file!" << std::endl;
    }
}

/* Import */

void ImportVisitor::doSomethingWithWeight(Model * model) {
    auto matrices = this->importMatrices(exportImportTypesToStr(ExportImportTypes::WEIGHTS) + "_");
    std::shared_ptr<Layer> currentLayer = model->getInputLayer();
    std::shared_ptr<Layer> firstLayer = model->getInputLayer();
    int counter = 0;
    while (counter < matrices.size()) {
        currentLayer->setWeights(matrices[counter ++]);
        if (!currentLayer->isNextLayer() && counter < matrices.size()) {
            currentLayer->addLayer(NoActivation(), matrices[counter].getRowSize());
        }
        currentLayer = currentLayer->getNextLayer();
    }
}

void ImportVisitor::doSomethingWithBias(Model *model) {
    auto matrices = this->importMatrices(exportImportTypesToStr(ExportImportTypes::BIASES) + "_");
    std::shared_ptr<Layer> currentLayer = model->getInputLayer();
    int counter = 0;
    while (counter < matrices.size()) {
        currentLayer->setBiases(matrices[counter ++]);
        if (!currentLayer->isNextLayer() && counter < matrices.size()) {
            currentLayer->addLayer(NoActivation(), matrices[counter].getRowSize());
        }
        currentLayer = currentLayer->getNextLayer();
    }
}

void ImportVisitor::doSomethingWithActivations(Model *model) {
    auto activations = this->importActivationFunctions();
    std::shared_ptr<Layer> currentLayer = model->getInputLayer();
    int counter = 0;
    while (counter < activations.size()) {
        currentLayer->setActivationFunction(activations[counter ++]);
        if (!currentLayer->isNextLayer() && counter < activations.size()) {
            currentLayer->addLayer(*activations[counter], 2);
        }
        currentLayer = currentLayer->getNextLayer();
    }
}

std::vector<Matrix> ImportVisitor::importMatrices(const std::string& filePathPrefix) {
    std::string path = "../" + filePathPrefix + this->_path;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::invalid_argument("Couldn't open file \"" + path + "\"\n");
    }
    std::string line;
    std::vector<Matrix> matrices;
    size_t columns = 0, rows = 0;
    std::vector<float> values = {};
    bool isNewMatrix = false, isClosingMatrix = false;
    while (std::getline(file, line)) {
        if (line[0] != '[') continue;
        if (line[1] == '[') isNewMatrix = true;
        if (line.ends_with("]]")) isClosingMatrix = true;

        line.erase(std::remove(line.begin(), line.end(), '['), line.end());
        line.erase(std::remove(line.begin(), line.end(), ']'), line.end());
        line.erase(std::remove(line.begin(), line.end(), ','), line.end());
        auto res = readMatrixLine(line);

        values.insert(values.end(), res.begin(), res.end());
        rows ++;

        if (isNewMatrix) {
            columns = res.size();
            isNewMatrix = false;
        }
        if (isClosingMatrix) {
            matrices.push_back(Matrix::fromVector(values, columns, rows));
            values.clear();
            rows = 0;
            columns = 0;
            isClosingMatrix = false;
        }
    }
    return matrices;
}

std::vector<std::shared_ptr<ActivationFunction>> ImportVisitor::importActivationFunctions() {
    std::string line;
    std::string path = "../" + exportImportTypesToStr(ExportImportTypes::ACTIVATIONS) + "_" + this->_path;
    std::ifstream file(path);
    std::vector<std::shared_ptr<ActivationFunction>> result = {};
    if (!file.is_open()) {
        throw std::invalid_argument("Couldn't open file \"" + path + "\"\n");
    }
    while (std::getline(file, line)) {
        if (line.starts_with("Layer")) continue;
        result.push_back(readActivationType(line));
    }
    return result;
}