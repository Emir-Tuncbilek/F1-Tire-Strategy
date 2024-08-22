//
// Created by Emir Tuncbilek on 8/19/24.
//

#include "./neural-network/model.h"
#include "./data-interpretor/data-loader.h"

int main(int argc, char* argv[]) {

    auto model = Model::importModel("file.autism");
    std::cout << "Model loaded" << std::endl;
    const std::pair<std::vector<float>, size_t> Xdata = DataLoader::load("../x-preprocessed-data.csv");
    const std::pair<std::vector<float>, size_t> targetData = DataLoader::load("../y-preprocessed-data.csv");
    auto X = DataLoader::generateVectors(Xdata);
    auto targets = DataLoader::generateVectors(targetData);
    double totalDeviation = 0.0;
    int minIndex = -1, maxIndex = -1;
    double maxDeviation = 0.0, minDeviation = 1.0;
    for (int i = 0; i < X.size(); i ++) {
        auto result = model->predict(X[i]);
        auto diff = (targets[i] - result).map([](auto x) { return abs(x); }).sum() / (double)result.getRowSize();
        totalDeviation += diff;
        if (diff > maxDeviation) { maxDeviation = diff; maxIndex = i; }
        if (diff < minDeviation) { minDeviation = diff; minIndex = i; }
        std::cout << "@ [" << i + 1 << "] -> Deviation (%) : " << diff * 100. << std::endl;
    }

    std::cout << "Avg accuracy : " << ( 1. - totalDeviation / (double) X.size()) * 100. << " %" << std::endl;
    std::cout << "Min deviation : " << minDeviation * 100. << " % @ position: " << minIndex + 1 << std::endl;
    std::cout << "Max deviation : " << maxDeviation * 100. << " % @ position: " << maxIndex + 1 << std::endl;

    return 0;
}
