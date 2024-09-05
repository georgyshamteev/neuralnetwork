#include "dataloader.h"

#include <random>

namespace utils {

DataLoader::DataLoader(Tensor2D data, Tensor2D labels, size_t batch_size,
                       DataLoader::shuffle enable_shuffle)
    : data_(std::move(data)),
      labels_(std::move(labels)),
      batch_size_(batch_size),
      enable_shuffle_(enable_shuffle),
      permutations_matrix_(data_.rows()) {
    permutations_matrix_.setIdentity();
    if (enable_shuffle == shuffle::True) {
        Shuffle();
    }
}

void DataLoader::Shuffle() {
    static std::random_device r;
    static std::mt19937 eng1(r());
    std::shuffle(permutations_matrix_.indices().data(),
                 permutations_matrix_.indices().data() + permutations_matrix_.indices().size(),
                 eng1);

    data_ = permutations_matrix_ * data_;
    labels_ = permutations_matrix_ * labels_;
}

void DataLoader::Restore() {
    const auto& inverse = permutations_matrix_.inverse().eval();
    data_ = inverse * data_;
    labels_ = inverse * labels_;
    permutations_matrix_.setIdentity();
}

//// DataLoaderIterator begin

DataLoader::DataLoaderIterator::DataLoaderIterator(const Tensor2D* data_ptr,
                                                   const Tensor2D* labels_ptr, size_t idx,
                                                   size_t batch_size)
    : data_ptr_(data_ptr), labels_ptr_(labels_ptr), index_(idx), batch_size_(batch_size) {
}

DataLoader::DataLoaderIterator::Batch DataLoader::DataLoaderIterator::operator*() const {
    if (index_ + batch_size_ > data_ptr_->rows()) {
        size_t size = data_ptr_->rows() % batch_size_;
        return {data_ptr_->middleRows(index_, size), labels_ptr_->middleRows(index_, size)};
    }
    return {data_ptr_->middleRows(index_, batch_size_),
            labels_ptr_->middleRows(index_, batch_size_)};
}

DataLoader::DataLoaderIterator& DataLoader::DataLoaderIterator::operator++() {
    index_ += batch_size_;
    return *this;
}

DataLoader::DataLoaderIterator DataLoader::DataLoaderIterator::operator++(int) {
    auto tmp = *this;
    index_ += batch_size_;
    return tmp;
}

bool operator!=(const DataLoader::DataLoaderIterator& lhs,
                const DataLoader::DataLoaderIterator& rhs) {
    return lhs.index_ != rhs.index_;
}

//// DataLoaderIterator end

DataLoader::DataLoaderIterator DataLoader::begin() {
    return DataLoader::DataLoaderIterator(&data_, &labels_, 0, batch_size_);
}

DataLoader::DataLoaderIterator DataLoader::end() {
    size_t end_idx = data_.rows() + (batch_size_ - data_.rows() % batch_size_) % batch_size_;
    return DataLoader::DataLoaderIterator(&data_, &labels_, end_idx, batch_size_);
}

size_t DataLoader::Size() const {
    return labels_.rows();
}

Tensor2D ReadCSV(std::string path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        if (rows == 0) {
            ++rows;
            continue;
        }
        std::stringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    --rows;
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        values.data(), rows, values.size() / rows);
}

Tensor2D Argmax(const Tensor2D& mat, size_t axis) {
    if (axis == 1) {
        Tensor2D ret(mat.rows(), 1);
        for (Index i = 0; i < mat.rows(); ++i) {
            size_t idx = 0;
            double mx = mat(i, 0);
            for (Index j = 0; j < mat.cols(); ++j) {
                if (mat(i, j) > mx) {
                    mx = mat(i, j);
                    idx = j;
                }
            }
            ret(i, 0) = idx;
        }
        return ret;
    } else {
        Tensor2D ret(1, mat.cols());
        for (Index i = 0; i < mat.cols(); ++i) {
            size_t idx = 0;
            double mx = mat(0, i);
            for (Index j = 0; j < mat.rows(); ++j) {
                if (mat(i, j) > mx) {
                    mx = mat(i, j);
                    idx = j;
                }
            }
            ret(0, i) = idx;
        }
        return ret;
    }
}

}  // namespace utils
