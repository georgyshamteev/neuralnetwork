#include <catch.hpp>
#include "optimizer.h"

TEST_CASE("Constant optimizer creates and runs correctly") {
    nn::Tensor2D weight(10, 10);
    weight.setConstant(10);
    nn::Tensor2D grad(10, 10);
    grad.setConstant(4);

    nn::ConstantOptimizer optimizer({{weight, grad}}, 0.1);
    optimizer.Step();

    REQUIRE(weight.isConstant(9.6));
}

TEST_CASE("Hyperbolic optimizer creates and runs correctly") {
    nn::Tensor2D weight(10, 10);
    weight.setConstant(10);
    nn::Tensor2D grad(10, 10);
    grad.setConstant(4);

    nn::HyperbolicOptimizer optimizer({{weight, grad}}, 0.3);
    optimizer.Step();
    optimizer.Step();
    optimizer.Step();

    /*
     weight = 10;
     weight -= 4 * 0.3
     weight -= 4 * 0.15
     weight -= 4 * 0.1
     weight = 7.8
    */

    REQUIRE(weight.isConstant(7.8));
}
