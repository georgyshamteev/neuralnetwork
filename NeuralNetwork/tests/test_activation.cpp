#include <catch.hpp>
#include "activation.h"

TEST_CASE("Activation function creates successfully") {
    REQUIRE_NOTHROW(nn::ActivationFunction::ReLU());
    REQUIRE_NOTHROW(nn::ActivationFunction::Sigmoid());
    //    REQUIRE_NOTHROW(nn::ActivationFunction::Tanh()); TODO
}

TEST_CASE("ReLU operator() for positive") {
    nn::ActivationFunction relu = nn::ActivationFunction::ReLU();
    nn::Tensor2D input(10, 20);
    const double v = 10.0;
    input.setConstant(v);
    auto out = relu(input);
    REQUIRE(out.rows() == input.rows());
    REQUIRE(out.cols() == input.cols());
    REQUIRE(out.isConstant(v));
}

TEST_CASE("ReLU operator() for negative") {
    nn::ActivationFunction relu = nn::ActivationFunction::ReLU();
    nn::Tensor2D input(10, 20);
    const double v = -10.0;
    input.setConstant(v);
    auto out = relu(input);
    REQUIRE(out.rows() == input.rows());
    REQUIRE(out.cols() == input.cols());
    REQUIRE(out.isConstant(0));
}

TEST_CASE("ReLU Update for positive") {
    constexpr double kV = 10.0;

    nn::ActivationFunction relu = nn::ActivationFunction::ReLU();
    nn::Tensor2D input(100, 10);
    input.setConstant(kV);
    auto out = relu(input);

    nn::Tensor2D u(100, 10);
    u.setConstant(0.5);
    auto ret = relu.Update(u);
    REQUIRE(ret.isConstant(0.5));
    REQUIRE(ret.rows() == u.cols());
    REQUIRE(ret.cols() == u.rows());
}

TEST_CASE("ReLU Update for negative") {
    constexpr double kV = -10.0;

    nn::ActivationFunction relu = nn::ActivationFunction::ReLU();
    nn::Tensor2D input(100, 10);
    input.setConstant(kV);
    auto out = relu(input);

    nn::Tensor2D u(100, 10);
    u.setConstant(0.5);
    auto ret = relu.Update(u);
    REQUIRE(ret.isConstant(0));
    REQUIRE(ret.rows() == u.cols());
    REQUIRE(ret.cols() == u.rows());
}
