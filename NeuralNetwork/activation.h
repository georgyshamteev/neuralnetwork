#pragma once

#include "neuraldefines.h"

namespace nn {

class ActivationFunction : protected NeuralDefines {

public:
    static ActivationFunction ReLU();
    static ActivationFunction Sigmoid();
    static ActivationFunction Tanh();
    static ActivationFunction Softmax();

    Tensor2D operator()(const Tensor2D& x);
    Tensor2D Update(const Tensor2D& u);
    std::vector<ParameterPack> TrainingParams();

    friend std::fstream& operator<<(std::fstream& in, const ActivationFunction& layer);
    friend std::fstream& operator>>(std::fstream& in, ActivationFunction& layer);

private:
    ActivationFunction(std::function<Tensor2D(const Tensor2D&)>,
                       std::function<Tensor2D(const Tensor1D&)>);
    Tensor2D input_;
    std::function<Tensor2D(const Tensor2D&)> sigma_;
    std::function<Tensor2D(const Tensor1D&)> dsigma_;
};

}  // namespace nn
