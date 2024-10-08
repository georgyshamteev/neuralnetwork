#pragma once

#include <fstream>

#include "memory"
#include "neuraldefines.h"

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
    std::vector<ParameterPack> TrainingParams();

    void Train();
    void Eval();

    friend std::fstream& operator<<(std::fstream& in, const Linear& layer);
    friend std::fstream& operator>>(std::fstream& in, Linear& layer);

private:
    struct LinearData {
        Tensor2D input_seq_;
        Tensor2D weight_grad_;
        Tensor2D bias_grad_;
    };
    enum class ModelState {
        train,
        eval,
    };

    Tensor2D InitializeWeights(int64_t in_features, int64_t out_features);
    Tensor2D InitializeBias(Bias enable_bias, int64_t in_features, int64_t out_features);
    ModelState model_state_;
    Bias bias_state_;
    Tensor2D weight_;
    Tensor2D bias_;
    std::unique_ptr<LinearData> data_ptr_;
};

}  // namespace nn
