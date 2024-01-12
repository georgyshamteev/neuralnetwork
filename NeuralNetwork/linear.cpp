#include "linear.h"

Linear::Linear(size_t in_features, size_t out_features, bool bias) :
        weight_(in_features, out_features), bias_(out_features){
    double k = 1 / in_features;
    double sqrt_k = std::sqrt(k);
    Eigen::Rand::UniformRealGen gen(-sqrt_k, sqrt_k);
    if (bias) {
        bias_ = gen.generate<Tensor1D>(1, out_features, urng);
    }

}
