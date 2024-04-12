#pragma once

#include "neuraldefines.h"
#include <tuple>
#include "AnyType.h"

namespace nn {

class Sequential {
public:
    template <class... Args>
    Sequential(Args&&... args) {
        ((layers_.emplace_back(std::move(args)), 0), ..., 0);
    }

    Tensor2D operator()(const Tensor2D& x);

    void Backward(const Tensor2D& x);

private:
    std::vector<AnyObject> layers_;
};

}  // namespace nn
