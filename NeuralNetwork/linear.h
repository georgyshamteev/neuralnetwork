#pragma once
#include "neuraldefines.h"
#include "memory"

namespace nn {

enum class Bias {
    enable,
    disable,
};

class Linear : protected NeuralDefines {

public:
    Linear(Index in_features, Index out_features, Bias enable_bias = Bias::enable);
    Tensor2D operator()(const Tensor2D& x);
    Tensor2D Update(Tensor2D& u);
    void Train();
    void Eval();

private:
    struct LinearData {
        Tensor2D input_seq_;
        Tensor2D weight_grad_;
        Tensor1D bias_grad_;
    };
    enum class ModelState {
        train,
        eval,
    };

    Tensor2D InitializeWeights(int64_t in_features, int64_t out_features);
    Tensor1D InitializeBias(Bias enable_bias, int64_t in_features, int64_t out_features);
    ModelState model_state_;
    Bias bias_state_;
    Tensor2D weight_;
    Tensor1D bias_;
    std::unique_ptr<LinearData> data_ptr_;
};

}  // namespace nn
