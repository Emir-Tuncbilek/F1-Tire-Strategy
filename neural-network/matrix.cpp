//
// Created by Emir Tuncbilek on 7/16/24.
//

#include "matrix.h"


/* utility function */
float generateRandomNeg1_1() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    return dis(gen);
}

/* Static Methods */
Matrix Matrix::identity(const size_t& size) {
#if USE_GPU
    return Matrix::identityGPU(size);
#else
    return Matrix::identityCPU(size);
#endif
}

Matrix Matrix::randomMatrix(const size_t& rows, const size_t& columns) {
#if USE_GPU
    return Matrix::randomMatrixGPU(rows, columns);
#else
    return Matrix::randomMatrixCPU(rows, columns);
#endif
}

Matrix Matrix::nullMatrix(const size_t &rows, const size_t &columns) {
    std::vector<std::unique_ptr<std::vector<double>>> data(rows);
    for (int i = 0; i < rows; i++) {
        data[i] = std::make_unique<std::vector<double>>(columns, 0.);
    }
    return Matrix(std::move(data));
}

Matrix Matrix::nullVector(const size_t& size) {
    std::vector<std::unique_ptr<std::vector<double>>> data(size);
    for (int i = 0; i < size; i ++) {
        data[i] = std::make_unique<std::vector<double>>(1, 0);
    }
    return Matrix(std::move(data));
}

Matrix Matrix::randomVector(const size_t& size) {
    std::vector<std::unique_ptr<std::vector<double>>> data(size);
    for (int i = 0; i < size; i ++) {
        data[i] = std::make_unique<std::vector<double>>(1, generateRandomNeg1_1());
    }
    return Matrix(std::move(data));
}

Matrix Matrix::fromVector(const std::vector<float> &result, const size_t &columns, const size_t &rows) {
    Matrix resultMatrix;
    resultMatrix.rows = rows;
    resultMatrix.columns = columns;
    resultMatrix.data.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
        resultMatrix.data[i] = std::make_unique<std::vector<double>>(result.begin() + i * columns, result.begin() + (i + 1) * columns);
    }
    return resultMatrix;
}

/* Constructor */
Matrix::Matrix(std::vector<std::unique_ptr<std::vector<double>>> data) {
    if (data.empty()) throw std::invalid_argument("Data shouldn't be empty!");
    if (data[0]->empty()) throw std::invalid_argument("Columns shouldn't be empty!");
    this->columns = data[0]->size();
    for (int i = 1; i < (int)data.size(); i ++) {
        if (data[i]->size() != this->columns)
            throw std::invalid_argument("All columns must be of the same size!");
    }
    this->rows = data.size();
    this->data = std::move(data);
}

/* Copy constructors */
Matrix::Matrix(const Matrix &other) {
    *this = other;
}

Matrix& Matrix::operator = (const Matrix &other) {
    //this->data.reserve(other.rows);
    this->data = std::vector<std::unique_ptr<std::vector<double>>>(other.rows);
    for (int i = 0; i < other.rows; i ++) {
        this->data[i] = std::make_unique<std::vector<double>>(other[i]);
    }
    this->columns = other.columns;
    this->rows = other.rows;
    return *this;
}

/* Operators */

Matrix Matrix::operator * (const Matrix &other) const {
#if USE_GPU
    return this->multGPU(other);
#else
    return this->multCPU(other);
#endif
}

Matrix& Matrix::operator *= (const Matrix &other) {
    *this = *this * other;
    return *this;
}

Matrix& Matrix::operator*=(const double& scalar) {
    *this = *this * scalar;
    return *this;
}

Matrix Matrix::operator * (const double& scalar) const {
#if USE_GPU
    return this->multGPU(scalar);
#else
    return this->multCPU(scalar);
#endif
}

Matrix Matrix::operator + (const Matrix &other) const {
#if USE_GPU
    return this->addGPU(other);
#else
    return this->addCPU(other);
#endif
}

Matrix& Matrix::operator += (const Matrix &other) {
    *this = *this + other;
    return *this;
}

Matrix Matrix::operator - (const Matrix &other) const {
#if USE_GPU
    return this->subGPU(other);
#else
    return this->subCPU(other);
#endif
}

Matrix& Matrix::operator -= (const Matrix &other) {
    *this = *this - other;
    return *this;
}

std::vector<double> Matrix::operator [] (int index) const {
    return *this->data[index];
}

/* Class methods */

Matrix Matrix::transpose() const {
#if USE_GPU
    return this->transposeGPU();
#else
    return this->transposeCPU();
#endif
}

Matrix Matrix::map(const std::function<double(double)> &callback) const {
#if USE_GPU
    return this->mapGPU(callback);
#else
    return this->mapCPU(callback);
#endif
}

Matrix Matrix::map(const Matrix& other, const std::function<double(double, double)>& callback) const {
#if USE_GPU
    return this->mapGPU(other, callback);
#else
    return this->mapCPU(other, callback);
#endif
}

Matrix Matrix::clone() const {
    auto identity = [](double x) { return x; };
    Matrix result = this->map(identity);
    result.setGPUMatrixMult(this->gpuMatrixMultiplier);
    return result;
}

double Matrix::sum() {
#if USE_GPU
    return this->sumGPU();
#else
    return this->sumCPU();
#endif
}

Matrix Matrix::getColumn(size_t columnIndex) const {
    if (columnIndex >= columns) {
        throw std::out_of_range("Column index out of range");
    }
    std::vector<std::unique_ptr<std::vector<double>>> columnData(this->rows);
    for (int i = 0; i < this->rows; i ++) {
        columnData[i] = std::make_unique<std::vector<double>>(1, (*this->data[i])[columnIndex]);
    }
    return Matrix(std::move(columnData));

}

size_t Matrix::getColumnSize() const { return this->columns; }

size_t Matrix::getRowSize() const { return this->rows; }

std::vector<float> Matrix::toVector() const {
    std::vector<float> result;
    result.reserve(this->columns * this->rows);
    for (auto&& row : this->data) {
        result.insert(result.end(), row->begin(), row->end());
    }
    return result;
}


/* CPU instructions */

Matrix Matrix::multCPU(const Matrix &other) const {
    if (this->columns != other.rows) {
        std::ostringstream oss;
        oss << "Can't perform the multiplication of a " << this->rows << "x" << this->columns
            << " matrix with a " << other.rows << "x" << other.columns << " matrix.";
        throw std::invalid_argument(oss.str());
    }
    std::vector<std::unique_ptr<std::vector<double>>> newData(this->rows);
    for (int i = 0; i < (int)this->rows; i ++) {
        newData[i] = std::make_unique<std::vector<double>>(other.columns, 0);
        for (int j = 0; j < (int) other.columns; j++) {
            for (int k = 0; k < (int) this->columns; k++) {
                (*newData[i])[j] += (*this->data[i])[k] * (*other.data[k])[j];
            }
        }
    }
    return Matrix(std::move(newData));
}

Matrix Matrix::multCPU(const double &scalar) const {
    std::vector<std::unique_ptr<std::vector<double>>> result(this->rows);
    for (int i = 0; i < this->rows; i ++) {
        result[i] = std::make_unique<std::vector<double>>(this->columns);
        for (int j = 0; j < this->columns; j ++) {
            (*result[i])[j] = (*this)[i][j] * scalar;
        }
    }
    return Matrix(std::move(result));
}

Matrix Matrix::addCPU(const Matrix &other) const {

    if (this->columns != other.columns || this->rows != other.rows ) {
        std::ostringstream oss;
        oss << "Can't perform the addition of a " << this->rows << "x" << this->columns
            << " matrix with a " << other.rows << "x" << other.columns << " matrix.";
        throw std::invalid_argument(oss.str());
    }
    for (int i = 0; i < (int)this->rows; i ++) {
        for (int j = 0; j < (int)this->columns; j ++) {
            (*this->data[i])[j] += (*other.data[i])[j];
        }
    }
    return *this;
}

Matrix Matrix::subCPU(const Matrix &other) const {
    if (this->columns != other.columns || this->rows != other.rows ) {
        std::ostringstream oss;
        oss << "Can't perform the subtraction of a " << this->rows << "x" << this->columns
            << " matrix with a " << other.rows << "x" << other.columns << " matrix.";
        throw std::invalid_argument(oss.str());
    }
    for (int i = 0; i < (int)this->rows; i ++) {
        for (int j = 0; j < (int)this->columns; j ++) {
            (*this->data[i])[j] -= (*other.data[i])[j];
        }
    }
    return *this;
}

Matrix Matrix::mapCPU(const std::function<double(double)> &callback) const {
    std::vector<std::unique_ptr<std::vector<double>>> newData(this->rows);
    for (int i = 0; i < this->rows; i ++) {
        newData[i] = std::make_unique<std::vector<double>>(this->columns);
        for (int j = 0; j < this->columns; j ++) {
            (*newData[i])[j] = callback((*this->data[i])[j]);
        }
    }
    return Matrix(std::move(newData));
}

Matrix Matrix::mapCPU(const Matrix &other, const std::function<double(double, double)> &callback) const {
    if (this->rows != other.rows || this->columns != other.columns) {
        throw std::invalid_argument("Matrices must have the same dimensions for map operation.");
    }
    std::vector<std::unique_ptr<std::vector<double>>> newData(this->rows);
    for (int i = 0; i < this->rows; i++) {
        newData[i] = std::make_unique<std::vector<double>>(this->columns);
        for (int j = 0; j < this->columns; j++) {
            (*newData[i])[j] = callback((*this->data[i])[j], (*other.data[i])[j]);
        }
    }
    return Matrix(std::move(newData));
}

double Matrix::sumCPU() const {
    if (this->columns != 1)
        throw std::invalid_argument("Can only sum a vector or a N x 1 Matrix!");
    double sum = 0;
    for (int i = 0; i < this->rows; i ++) {
        sum += (*this->data[i])[0];
    }
    return sum;
}

Matrix Matrix::transposeCPU() const {
    std::vector<std::unique_ptr<std::vector<double>>> newData(this->columns);
    for (int i = 0; i < this->columns; ++i) {
        newData[i] = std::make_unique<std::vector<double>>(this->rows);
    }
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->columns; ++j) {
            (*newData[j])[i] = (*this->data[i])[j];
        }
    }
    return Matrix(std::move(newData));
}

    /* Static methods */
Matrix Matrix::identityCPU(const size_t &size) {
    std::vector<std::unique_ptr<std::vector<double>>> data(size);
    for (int i = 0; i < size; i ++) {
        data[i] = std::make_unique<std::vector<double>>(size, 0);
        for (int j = 0; j < size; j ++) {
            if (i == j) (*data[i])[j] = 1;
        }
    }
    return Matrix(std::move(data));
}

Matrix Matrix::randomMatrixCPU(const size_t &rows, const size_t &columns) {
    std::vector<std::unique_ptr<std::vector<double>>> data(rows);
    for (int i = 0; i < rows; i ++) {
        data[i] = std::make_unique<std::vector<double>>(columns);
        for (int j = 0; j < columns; j ++) {
            (*data[i])[j] = generateRandomNeg1_1();
        }
    }
    return Matrix(std::move(data));
}


/* GPU instructions */
Matrix Matrix::multGPU(const Matrix &other) const {
    if (getColumnSize() != other.getRowSize()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    std::vector<float> result(getRowSize() * other.getColumnSize(), 0.0);

    // Execute matrix multiplication on the GPU
    if (!this->gpuMatrixMultiplier->execute(
            this->toVector(),
            other.toVector(),
            result,
            this->getRowSize(),
            other.getColumnSize(),
            this->getColumnSize())) {
        throw std::runtime_error("Failed to execute GPU matrix multiplication");
    }

    return Matrix::fromVector(result, other.columns, this->rows);
}

Matrix Matrix::multGPU(const double &scalar) const {
    return *this;
}

Matrix Matrix::addGPU(const Matrix &other) const {
    return *this;
}

Matrix Matrix::subGPU(const Matrix &other) const {
    return *this;
}

Matrix Matrix::mapGPU(const std::function<double(double)> &callback) const {
    return *this;
}

Matrix Matrix::mapGPU(const Matrix &other, const std::function<double(double, double)> &callback) const {
    return *this;
}

double Matrix::sumGPU() const {
    return 0.0;
}

Matrix Matrix::transposeGPU() const {
    return Matrix::randomMatrix(this->columns, this->rows);
}

    /* Static methods */
Matrix Matrix::identityGPU(const size_t &size) {
    return Matrix::identityCPU(size);   // change me
}

Matrix Matrix::randomMatrixGPU(const size_t &rows, const size_t &columns) {
    return Matrix::randomMatrixCPU(rows, columns); // change me
}

std::ostream& operator << (std::ostream& o, const Matrix& matrix) {
    o << "[";
    for (int i = 0; i < matrix.rows; i ++) {
        o << "[";
        for (int j = 0; j < matrix.columns; j ++) {
            o << (*matrix.data[i])[j] << (j + 1 != matrix.columns ? ", " : "");
        }
        o << "]" << (i + 1 != matrix.rows ? ",\n" : "");
    }
    o << "]" << std:: endl;
    return o;
}
