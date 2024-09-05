#pragma once

#include "neuraldefines.h"

namespace nn {

class Loss : protected NeuralDefines {

public:
    static Loss MSE();
    double operator()(const Tensor2D& x, const Tensor2D& y);
    Tensor2D Gradient() const;

private:
    Loss(std::function<double(const Tensor2D&, const Tensor2D&)>,
         std::function<Tensor2D(const Tensor2D&, const Tensor2D&)>);
    std::function<double(const Tensor2D&, const Tensor2D&)> loss_;
    std::function<Tensor2D(const Tensor2D&, const Tensor2D&)> grad_;
    Tensor2D input_;
    Tensor2D label_;
};

}  // namespace nn
