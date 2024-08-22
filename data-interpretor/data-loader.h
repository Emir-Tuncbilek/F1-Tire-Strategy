//
// Created by Emir Tuncbilek on 7/29/24.
//

#ifndef F1_STRATEGIES_DATA_LOADER_H
#define F1_STRATEGIES_DATA_LOADER_H

#pragma once
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include "../neural-network/matrix.h"


class DataLoader {
public:
    DataLoader() = default;
    static std::pair<std::vector<float>, size_t> load(const std::string& path);
    static std::vector<Matrix> generateVectors(const std::pair<std::vector<float>, size_t>& data);
};

#endif //F1_STRATEGIES_DATA_LOADER_H
