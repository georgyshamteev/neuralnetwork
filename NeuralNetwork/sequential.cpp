#include "sequential.h"

nn::Tensor2D nn::Sequential::operator()(const nn::Tensor2D& x) {
    Tensor2D data = x;
    for (size_t i = 0; i < layers_.size(); ++i) {
        data = (layers_[i])(data);
    }
    return data;
}

void nn::Sequential::Backward(const nn::Tensor2D& x) {
    Tensor2D grad = x;
    for (size_t i = layers_.size(); i != 0; --i) {
        grad = layers_[i - 1]->Update(grad);
    }
}
