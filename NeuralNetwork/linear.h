#include "neuraldefines.h"
#include "memory"

namespace nn {

class Linear : protected NeuralDefines {

public:
    Linear(int64_t in_features, int64_t out_features, bool enable_bias = true);
    Tensor2D operator()(const Tensor2D& x);
    Tensor2D Update(Tensor2D& u, double lambda);

private:
    Tensor2D InitializeWeights(int64_t in_features, int64_t out_features);
    Tensor1D InitializeBias(bool enable_bias, int64_t in_features, int64_t out_features);
    Tensor2D input_seq_;
    Tensor2D weight_grad_;
    Tensor1D bias_grad_;
    Tensor2D weight_;
    Tensor1D bias_;
    nn_random::Random rng_;
};

}  // namespace nn
