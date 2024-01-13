#include "linear.h"

Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : weight_(in_features, out_features), bias_(out_features) {
    double k = 1 / in_features;
    double sqrt_k = std::sqrt(k);
    Eigen::Rand::UniformRealGen gen(-sqrt_k, sqrt_k);
    if (bias) {
        bias_ = gen.generate<Tensor1D>(1, out_features, urng);
    }
    weight_ = gen.generate<Tensor2D>(in_features, out_features, urng);

    //// TODO initialize other class fields
}

Linear::Tensor2D Linear::operator()(Linear::Tensor2D x) {
    return weight_ * x + bias_;
}

void Linear::Update(Linear::Tensor2D u) {
    /*
     * We expect that vector u is coming from non-linear layer.
     * Thus vector u coming for update is actually vector u` = d sigma * u,
     * where u is the gradient vector that was fed to non-linear layer
     */
    weight_grad_ = u * input_seq_;
    bias_grad_ = u;
    weight_ -= weight_grad_;
    bias_ -= bias_grad_;
    weight_grad_.setZero();
    bias_grad_.setZero();
}
Linear::Tensor2D Linear::RethrowGradient(Linear::Tensor2D u) {
    //// Needs to be done before weight updating or not?
    return weight_ * u;
}
