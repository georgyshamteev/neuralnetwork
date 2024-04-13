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

TEST_CASE("Write weights to file") {
    nn::Tensor1D x = nn_random::Random::GetGaussVector(784);
    nn::Sequential model(nn::Linear(784, 196), nn::ActivationFunction::ReLU(), nn::Linear(196, 49),
                         nn::ActivationFunction::ReLU(), nn::Linear(49, 10));
    auto y = model(x);

    std::fstream stream;
    stream.open("/home/georgyshamteev/CLionProjects/coursework/NeuralNetwork/tests/weights.txt",
                std::ios::out);
    stream << model;
    stream.close();

    nn::Sequential model1(nn::Linear(784, 196), nn::ActivationFunction::ReLU(), nn::Linear(196, 49),
                          nn::ActivationFunction::ReLU(), nn::Linear(49, 10));

    std::fstream stream1;
    stream1.open("/home/georgyshamteev/CLionProjects/coursework/NeuralNetwork/tests/weights.txt",
                 std::ios::in);

    stream1 >> model1;

    auto y1 = model1(x);

    REQUIRE(y.isApprox(y1, 1e-5f));
}
