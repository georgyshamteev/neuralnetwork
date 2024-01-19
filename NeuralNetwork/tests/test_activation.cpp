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

TEST_CASE("ReLU operator() for positive") {
    nn::ActivationFunction relu = nn::ActivationFunction::ReLU();
    Tensor2D input(10, 20);
    const double v = 10.0;
    input.setConstant(v);
    auto out = relu(input);
    REQUIRE(out.rows() == input.rows());
    REQUIRE(out.cols() == input.cols());
    for (int j = 0; j < out.rows(); ++j) {
        for (auto i : out.row(j)) {
            REQUIRE(i == v);
        }
    }
}

TEST_CASE("ReLU operator() for negative") {
    nn::ActivationFunction relu = nn::ActivationFunction::ReLU();
    Tensor2D input(10, 20);
    const double v = -10.0;
    input.setConstant(v);
    auto out = relu(input);
    REQUIRE(out.rows() == input.rows());
    REQUIRE(out.cols() == input.cols());
    for (int j = 0; j < out.rows(); ++j) {
        for (auto i : out.row(j)) {
            REQUIRE(i == 0);
        }
    }
}

TEST_CASE("ReLU Update for positive") {
    nn::ActivationFunction relu = nn::ActivationFunction::ReLU();
    Tensor2D input(100, 10);
    const double v = 10.0;
    input.setConstant(v);
    auto out = relu(input);

    Tensor2D u(100, 10);
    u.setConstant(0.5);
    auto ret = relu.Update(u);
    REQUIRE(ret.isConstant(0.5));
    REQUIRE(ret.rows() == u.cols());
    REQUIRE(ret.cols() == u.rows());
}

TEST_CASE("ReLU Update for negative") {
    nn::ActivationFunction relu = nn::ActivationFunction::ReLU();
    Tensor2D input(100, 10);
    const double v = -10.0;
    input.setConstant(v);
    auto out = relu(input);

    Tensor2D u(100, 10);
    u.setConstant(0.5);
    auto ret = relu.Update(u);
    REQUIRE(ret.isConstant(0));
    REQUIRE(ret.rows() == u.cols());
    REQUIRE(ret.cols() == u.rows());
}