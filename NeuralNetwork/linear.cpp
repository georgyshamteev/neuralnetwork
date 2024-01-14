#include "linear.h"

namespace nn {
Linear::Linear(Index in_features, Index out_features, bool enable_bias)
    : weight_(InitializeWeights(in_features, out_features)),
      bias_(InitializeBias(enable_bias, in_features, out_features)) {
    //// TODO: insert asserts
    //// TODO: initialize other class fields
}

Linear::Tensor2D Linear::operator()(const Linear::Tensor2D& x) const {
    return x * weight_ + bias_;
}

Linear::Tensor2D Linear::Update(Linear::Tensor2D& u, double lambda) {
    /*
     * We expect that vector u is coming from non-linear layer.
     * Thus vector u coming for update is actually vector u` = d sigma * u,
     * where u is the gradient vector that was fed to non-linear layer
     */
    //// TODO: insert asserts
    Tensor2D rethrow_gradient_vector = u.transpose() * weight_;

    weight_grad_ = u * input_seq_;
    bias_grad_ = u.rowwise().sum();
    weight_ -= lambda * weight_grad_;
    bias_ -= lambda * bias_grad_;
    weight_grad_.setZero();
    bias_grad_.setZero();

    return rethrow_gradient_vector;
}

Linear::Tensor2D Linear::InitializeWeights(Linear::Index in_features, Linear::Index out_features) {
    double k = 1 / in_features;
    double sqrt_k = std::sqrt(k);
    return rng_.GetUniformMatrix(-sqrt_k, sqrt_k, in_features, out_features);
}

Linear::Tensor1D Linear::InitializeBias(bool enable_bias, Linear::Index in_features,
                                        Linear::Index out_features) {
    if (enable_bias) {
        double k = 1 / in_features;
        double sqrt_k = std::sqrt(k);
        return rng_.GetUniformVector(-sqrt_k, sqrt_k, out_features);
    }
    return Tensor1D(out_features).setZero();
}
}  // namespace nn
