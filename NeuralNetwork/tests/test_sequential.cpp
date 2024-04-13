#include <catch.hpp>
#include "sequential.h"
#include "linear.h"
#include "activation.h"

TEST_CASE("test") {
    REQUIRE_NOTHROW(nn::Sequential{});
    REQUIRE_NOTHROW(nn::Sequential{nn::Linear(10, 20)});
    REQUIRE_NOTHROW(nn::Sequential{nn::Linear(10, 20), nn::ActivationFunction::ReLU(),
                                   nn::Linear(20, 40), nn::ActivationFunction::Sigmoid(),
                                   nn::Linear(40, 10), nn::ActivationFunction::Tanh()});
}

TEST_CASE("Sequential operator()") {
    nn::Tensor1D x = nn_random::Random::GetGaussVector(784);
    nn::Sequential model(nn::Linear(784, 196), nn::ActivationFunction::ReLU(), nn::Linear(196, 49),
                         nn::ActivationFunction::ReLU(), nn::Linear(49, 10));
    auto y = model(x);
}
