#pragma once

#include "neuraldefines.h"
#include "iostream"
#include "AnyLayer.h"

namespace nn {

class Sequential {
public:
    template <class... Args>
    Sequential(Args&&... args) {
        (layers_.emplace_back(std::move(args)), ..., 0);
    }

    Tensor2D operator()(const Tensor2D& x);
    void Backward(const Tensor2D& x);
    std::vector<ParameterPack> TrainingParams();

private:
    std::vector<AnyLayer> layers_;
};

}  // namespace nn
