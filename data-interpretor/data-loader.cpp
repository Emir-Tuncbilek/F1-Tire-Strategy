//
// Created by Emir Tuncbilek on 7/29/24.
//

#include "data-loader.h"

std::pair<std::vector<float>, size_t> DataLoader::load(const std::string& path) {
    std::vector<float> data;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    size_t num_columns = 0;
    bool firstLine = true;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        size_t current_column = 0;
        while (std::getline(ss, value, ',')) {
            if (!firstLine) {
                data.push_back(std::stof(value));
            } else {
                ++current_column;
            }
        }
        firstLine = false;
        if (num_columns == 0) {
            num_columns = current_column;
        }
    }
    file.close();
    return std::make_pair(data, num_columns);
}

std::vector<Matrix> DataLoader::generateVectors(const std::pair<std::vector<float>, size_t>& data) {
    assert(data.first.size() % data.second == 0 && "Data is of incompatible size");
    std::vector<Matrix> vectors;
    vectors.reserve(data.first.size() / data.second);
    for (size_t i = 0; i < data.first.size(); i += data.second) {
        std::vector<float> vectorCopy;
        std::copy(data.first.begin() + i, data.first.begin() + i + data.second, std::back_inserter(vectorCopy));
        vectors.push_back(Matrix::fromVector(vectorCopy, 1, data.second));
    }
    return vectors;
}
