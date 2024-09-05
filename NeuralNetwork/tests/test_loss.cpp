#include <catch.hpp>

#include "loss.h"

TEST_CASE("Loss creates") {
    auto l = nn::Loss::MSE();
}

TEST_CASE("Loss calculates") {
    nn::Tensor1D input(4);
    input << 1, 1, 1, 1;
    nn::Tensor1D label(4);
    label << 2, 2, 2, 2;
    auto loss = nn::Loss::MSE();
    REQUIRE(loss(input, label) == 4);
}

TEST_CASE("Grad computation works") {
    nn::Tensor1D input(4);
    input << 1, 1, 1, 1;
    nn::Tensor1D label(4);
    label << 2, 2, 2, 2;
    auto loss = nn::Loss::MSE();
    REQUIRE(loss(input, label) == 4);

    nn::Tensor1D grad(4);
    grad << -2, -2, -2, -2;
    REQUIRE(loss.Gradient() == grad);
}
