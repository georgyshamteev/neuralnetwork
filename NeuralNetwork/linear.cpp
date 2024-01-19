#include "linear.h"

namespace nn {

Linear::Linear(int64_t in_features, int64_t out_features, bool enable_bias)
    : weight_(InitializeWeights(in_features, out_features)),
      bias_(InitializeBias(enable_bias, in_features, out_features)) {
    //// TODO: insert asserts
    //// TODO: initialize other class fields
}

Linear::Tensor2D Linear::operator()(const Linear::Tensor2D& x) {
    input_seq_ = x;
    return x * weight_ + (Tensor1D(x.rows()).setOnes().transpose()) * bias_;
}

Linear::Tensor2D Linear::Update(Linear::Tensor2D& u, double lambda) {
    /*
     * We expect that vector u is coming from non-linear layer.
     * Thus vector u coming for update is actually vector u` = d sigma * u,
     * where u is the gradient vector that was fed to non-linear layer.
     * The size of u` is expected to be (m, k), while weight size is (n, m).
     */
    //// TODO: insert asserts
    Tensor2D rethrow_gradient_vector = (weight_ * u).transpose() ;

    weight_grad_ = (u * input_seq_).transpose();
    bias_grad_ = u.rowwise().sum();
    weight_ -= lambda * weight_grad_;
    bias_ -= lambda * bias_grad_;
    weight_grad_.setZero();
    bias_grad_.setZero();

    return rethrow_gradient_vector;
}

Linear::Tensor2D Linear::InitializeWeights(int64_t in_features, int64_t out_features) {
    double k = 1 / in_features;
    double sqrt_k = std::sqrt(k);
    return rng_.GetUniformMatrix(-sqrt_k, sqrt_k, in_features, out_features);
}

Linear::Tensor1D Linear::InitializeBias(bool enable_bias, int64_t in_features,
                                        int64_t out_features) {
    if (enable_bias) {
        double k = 1 / in_features;
        double sqrt_k = std::sqrt(k);
        return rng_.GetUniformVector(-sqrt_k, sqrt_k, out_features);
    }
    return Tensor1D(out_features).setZero();
}

}  // namespace nn
