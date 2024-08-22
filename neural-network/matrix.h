//
// Created by Emir Tuncbilek on 7/16/24.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <sstream>
#include <random>
#include <ostream>
#include "./env.h"
#include "./GPUfunctions.h"

class Matrix {
public:
    /* Static methods */
    static Matrix identity(const size_t& size);
    static Matrix randomMatrix(const size_t& rows, const size_t& columns);
    static Matrix nullMatrix(const size_t& rows, const size_t& columns);
    static Matrix nullVector(const size_t& size);
    static Matrix randomVector(const size_t& size);
    static Matrix fromVector(const std::vector<float>& result, const size_t& columns, const size_t& rows);


    /* Constructor */
    Matrix() = default;
    explicit Matrix(std::vector<std::unique_ptr<std::vector<double>>> data);

    /* Copy constructors */
    Matrix(const Matrix& other);
    Matrix& operator = (const Matrix& other);

    /* Default Destroyer */
    ~Matrix() = default;

    /* Operators */
    Matrix operator * (const Matrix& other) const;
    Matrix& operator *= (const Matrix& other);
    Matrix operator * (const double& other) const;
    Matrix& operator *= (const double& other);
    Matrix operator + (const Matrix& other) const;
    Matrix& operator += (const Matrix& other);
    Matrix operator - (const Matrix& other) const;
    Matrix& operator -= (const Matrix& other);
    std::vector<double> operator [] (int index) const;

    /* Class Methods */
    [[nodiscard]] Matrix transpose() const;

    Matrix map(const std::function<double(double)>& callback) const;
    Matrix map(const Matrix& other, const std::function<double(double, double)>& callback) const;
    double sum();
    [[nodiscard]] Matrix clone() const;
    Matrix getColumn(size_t columnIndex) const;
    [[nodiscard]] size_t getRowSize() const;
    [[nodiscard]] size_t getColumnSize() const;
    std::vector<float> toVector() const;
    void setGPUMatrixMult(const std::shared_ptr<GPUMatrixMultiplier>& f) { this->gpuMatrixMultiplier = f; }

    friend std::ostream& operator << (std::ostream& o, const Matrix& matrix);

private:
    size_t rows;
    size_t columns;
    // this may seem dumb, but it forces no copies of the data, leading to better performances.
    // data is of the form data[][], or size(data) is the size of the rows and size(data[n]) is
    // the size of the columns
    std::vector<std::unique_ptr<std::vector<double>>> data;
    std::shared_ptr<GPUMatrixMultiplier> gpuMatrixMultiplier;


    /* CPU implementations */
    [[nodiscard]] Matrix multCPU(const Matrix& other) const;
    [[nodiscard]] Matrix multCPU(const double& scalar) const;
    [[nodiscard]] Matrix addCPU(const Matrix& other) const;
    [[nodiscard]] Matrix subCPU(const Matrix& other) const;
    [[nodiscard]] Matrix mapCPU(const std::function<double(double)>& callback) const;
    [[nodiscard]] Matrix mapCPU(const Matrix& other, const std::function<double(double, double)>& callback) const;
    [[nodiscard]] double sumCPU() const;
    [[nodiscard]] Matrix transposeCPU() const;
    static Matrix identityCPU(const size_t& size);
    static Matrix randomMatrixCPU(const size_t& rows, const size_t& columns);


    /* GPU implementations */
    [[nodiscard]] Matrix multGPU(const Matrix& other) const;
    [[nodiscard]] Matrix multGPU(const double& scalar) const;
    [[nodiscard]] Matrix addGPU(const Matrix& other) const;
    [[nodiscard]] Matrix subGPU(const Matrix& other) const;
    [[nodiscard]] Matrix mapGPU(const std::function<double(double)>& callback) const;
    [[nodiscard]] Matrix mapGPU(const Matrix& other, const std::function<double(double, double)>& callback) const;
    [[nodiscard]] double sumGPU() const;
    [[nodiscard]] Matrix transposeGPU() const;
    static Matrix identityGPU(const size_t& size);
    static Matrix randomMatrixGPU(const size_t& rows, const size_t& columns);

};

#endif // MATRIX_H
