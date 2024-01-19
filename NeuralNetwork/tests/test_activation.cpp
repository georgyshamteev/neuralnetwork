#include <catch.hpp>
#include "activation.h"

using Index = Eigen::Index;
using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Tensor3D = std::vector<Tensor2D>;

TEST_CASE("Activation function creates successfully") {
    REQUIRE_NOTHROW(nn::ActivationFunction::ReLU());
    REQUIRE_NOTHROW(nn::ActivationFunction::Sigmoid());
//    REQUIRE_NOTHROW(nn::ActivationFunction::Tanh()); TODO
}

TEST_CASE()
