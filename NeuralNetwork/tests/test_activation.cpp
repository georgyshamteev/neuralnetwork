#include <catch.hpp>
#include "activation.h"
#include "iostream"

TEST_CASE("Activation function creates successfully") {
    REQUIRE_NOTHROW(nn::ActivationFunction::ReLU());
    REQUIRE_NOTHROW(nn::ActivationFunction::Sigmoid());
    REQUIRE_NOTHROW(nn::ActivationFunction::Tanh());
    REQUIRE_NOTHROW(nn::ActivationFunction::Softmax());
}

//// Test cases for ReLU

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

//// Test cases for Sigmoid

TEST_CASE("Sigmoid operator() for positive") {
    nn::ActivationFunction sigmoid = nn::ActivationFunction::Sigmoid();
    nn::Tensor2D input(10, 20);
    const double v = 10.0;
    input.setConstant(v);
    auto out = sigmoid(input);
    REQUIRE(out.rows() == input.rows());
    REQUIRE(out.cols() == input.cols());
    REQUIRE(out.isApproxToConstant(0.9999, 1e-4f));
}

TEST_CASE("Sigmoid operator() for negative") {
    nn::ActivationFunction sigmoid = nn::ActivationFunction::Sigmoid();
    nn::Tensor2D input(10, 20);
    const double v = -10.0;
    input.setConstant(v);
    auto out = sigmoid(input);
    REQUIRE(out.rows() == input.rows());
    REQUIRE(out.cols() == input.cols());
    REQUIRE(out.isApproxToConstant(4.5e-05, 0.01));
}

TEST_CASE("Sigmoid Update for positive") {
    constexpr double kV = 10.0;

    nn::ActivationFunction sigmoid = nn::ActivationFunction::Sigmoid();
    nn::Tensor2D input(100, 10);
    input.setConstant(kV);
    auto out = sigmoid(input);

    nn::Tensor2D u(100, 10);
    u.setOnes();
    auto ret = sigmoid.Update(u);
    REQUIRE(ret.isApproxToConstant(4.5e-05, 0.01));
    REQUIRE(ret.rows() == u.cols());
    REQUIRE(ret.cols() == u.rows());
}

TEST_CASE("Sigmoid Update for negative") {
    constexpr double kV = -10.0;

    nn::ActivationFunction sigmoid = nn::ActivationFunction::Sigmoid();
    nn::Tensor2D input(100, 10);
    input.setConstant(kV);
    auto out = sigmoid(input);

    nn::Tensor2D u(100, 10);
    u.setOnes();
    auto ret = sigmoid.Update(u);
    REQUIRE(ret.isApproxToConstant(4.5e-05, 0.01));
    REQUIRE(ret.rows() == u.cols());
    REQUIRE(ret.cols() == u.rows());
}

//// Test cases for Tanh

TEST_CASE("Tanh operator() for positive") {
    nn::ActivationFunction tanh = nn::ActivationFunction::Tanh();
    nn::Tensor2D input(10, 20);
    const double v = 0.5;
    input.setConstant(v);
    auto out = tanh(input);
    REQUIRE(out.rows() == input.rows());
    REQUIRE(out.cols() == input.cols());
    REQUIRE(out.isApproxToConstant(0.4621, 1e-2));
}

TEST_CASE("Tanh operator() for negative") {
    nn::ActivationFunction tanh = nn::ActivationFunction::Tanh();
    nn::Tensor2D input(10, 20);
    const double v = -0.5;
    input.setConstant(v);
    auto out = tanh(input);
    REQUIRE(out.rows() == input.rows());
    REQUIRE(out.cols() == input.cols());
    REQUIRE(out.isApproxToConstant(-0.4621, 1e-2));
}

TEST_CASE("Tanh Update for positive") {
    constexpr double kV = 0.5;

    nn::ActivationFunction tanh = nn::ActivationFunction::Tanh();
    nn::Tensor2D input(100, 10);
    input.setConstant(kV);
    auto out = tanh(input);

    nn::Tensor2D u(100, 10);
    u.setOnes();
    auto ret = tanh.Update(u);
    REQUIRE(ret.isApproxToConstant(0.7864, 0.001));
    REQUIRE(ret.rows() == u.cols());
    REQUIRE(ret.cols() == u.rows());
}

TEST_CASE("Tanh Update for negative") {
    constexpr double kV = -0.5;

    nn::ActivationFunction tanh = nn::ActivationFunction::Tanh();
    nn::Tensor2D input(100, 10);
    input.setConstant(kV);
    auto out = tanh(input);

    nn::Tensor2D u(100, 10);
    u.setOnes();
    auto ret = tanh.Update(u);
    REQUIRE(ret.isApproxToConstant(0.7864, 0.001));
    REQUIRE(ret.rows() == u.cols());
    REQUIRE(ret.cols() == u.rows());
}

//// Test cases for Softmax

TEST_CASE("softmax for probabilities 1") {
    nn::Tensor1D v(4);
    v << 1, 2, 3, 4;

    nn::ActivationFunction softmax = nn::ActivationFunction::Softmax();
    auto out = softmax(v);

    nn::Tensor1D cmp(4);
    cmp << 0.032, 0.087, 0.236, 0.643;

    REQUIRE(out.isApprox(cmp, 0.01));
}

TEST_CASE("softmax for probabilities 2") {
    nn::Tensor1D v(4);
    v << 1, 1, 1, 1;

    nn::ActivationFunction softmax = nn::ActivationFunction::Softmax();
    auto out = softmax(v);

    nn::Tensor1D cmp(4);
    cmp << 0.25, 0.25, 0.25, 0.25;

    REQUIRE(out.isApprox(cmp, 0.01));
}

TEST_CASE("softmax for probabilities 3") {
    nn::Tensor1D v(4);
    v << 1000, -1000, -1000, -1000;

    nn::ActivationFunction softmax = nn::ActivationFunction::Softmax();
    auto out = softmax(v);

    nn::Tensor1D cmp(4);
    cmp << 1, 0, 0, 0;

    REQUIRE(out.isApprox(cmp, 0.01));
}

TEST_CASE("softmax for probabilities 4") {
    nn::Tensor1D v(4);
    v << 1000, 2000, -1000, -1000;

    nn::ActivationFunction softmax = nn::ActivationFunction::Softmax();
    auto out = softmax(v);

    nn::Tensor1D cmp(4);
    cmp << 0, 1, 0, 0;

    REQUIRE(out.isApprox(cmp, 0.01));
}

TEST_CASE("softmax updates correctly") {
    nn::Tensor1D v(4);
    v << 1, 2, 3, 4;

    nn::ActivationFunction softmax = nn::ActivationFunction::Softmax();
    auto out = softmax(v);

    nn::Tensor2D u(1, 4);
    u << 1, 0, 0, 0;

    auto ret = softmax.Update(u);

    nn::Tensor1D cmp(4);
    cmp << 0.031, -0.0027, -0.0075, -0.02064;

    REQUIRE(ret.isApprox(cmp.transpose(), 0.1));
}

////
