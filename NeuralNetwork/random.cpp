#include "random.h"

namespace nn_random {

Random::Tensor1D Random::GetGaussVector(Index cols) {
    Eigen::Rand::NormalGen<double> gen;
    return gen.generate<Tensor1D>(1, cols, Gen()) / 10;
}

Random::Tensor2D Random::GetGaussMatrix(Index rows, Index cols) {
    Eigen::Rand::NormalGen<double> gen;
    return gen.generate<Tensor2D>(rows, cols, Gen()) / 10;
}

}  // namespace nn_random
