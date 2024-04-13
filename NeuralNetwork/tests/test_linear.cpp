#include <catch.hpp>
#include "linear.h"

TEST_CASE("Linear is successfully created") {
    REQUIRE_NOTHROW(nn::Linear(10, 10, nn::Bias::enable));
    REQUIRE_NOTHROW(nn::Linear(20, 10, nn::Bias::disable));
    REQUIRE_NOTHROW(nn::Linear(32, 16));
}

TEST_CASE("Linear operator() is working with single input") {
    nn::Linear layer(20, 10, nn::Bias::enable);
    nn::Tensor2D input(1, 20);
    auto output = layer(input);
    REQUIRE(output.rows() == 1);
    REQUIRE(output.cols() == 10);
}

TEST_CASE("Linear operator() is working with multiple input") {
    nn::Linear layer(20, 10, nn::Bias::enable);
    nn::Tensor2D input(100, 20);
    auto output = layer(input);
    REQUIRE(output.rows() == 100);
    REQUIRE(output.cols() == 10);
}

TEST_CASE("Linear layer update is working with single input") {
    nn::Linear layer(20, 10, nn::Bias::enable);
    nn::Tensor2D input(1, 20);
    auto output = layer(input);
    REQUIRE(output.rows() == 1);
    REQUIRE(output.cols() == 10);

    nn::Tensor2D u(10, 1);
    u.setRandom();
    double lambda = 2e-5;
    auto grad = layer.Update(u);
    REQUIRE(grad.rows() == 1);
    REQUIRE(grad.cols() == 20);
}

TEST_CASE("Linear layer update is working with multiple input") {
    nn::Linear layer(20, 10, nn::Bias::enable);
    nn::Tensor2D input(100, 20);
    auto output = layer(input);
    REQUIRE(output.rows() == 100);
    REQUIRE(output.cols() == 10);

    nn::Tensor2D u(10, 100);
    u.setRandom();
    double lambda = 2e-5;
    auto grad = layer.Update(u);
    REQUIRE(grad.rows() == 100);
    REQUIRE(grad.cols() == 20);
}
