#pragma once

#include "AnyLayer.h"
#include "iostream"
#include "neuraldefines.h"

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

    friend std::fstream& operator<<(std::fstream& in, Sequential& layer);
    friend std::fstream& operator>>(std::fstream& in, Sequential& layer);

private:
    std::vector<AnyLayer> layers_;
};

}  // namespace nn
