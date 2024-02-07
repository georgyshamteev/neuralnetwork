#include "random.h"
#include <catch.hpp>
#include "set"

TEST_CASE("Get random vector works correctly") {
    nn_random::Random rng;
    for (size_t cols = 10; cols < 100; ++cols) {
        auto gen = rng.GetGaussVector(cols);
        std::set<double> st;
        for (auto i : gen) {
            st.insert(i);
        }
        REQUIRE(gen.cols() == cols);
        REQUIRE(st.size() > cols / 2);
    }
}

void TestGaussMatrix(size_t rows, size_t cols) {
    auto gen = nn_random::Random::GetGaussMatrix(rows, cols);
    std::set<double> st;
    for (int i = 0; i < gen.rows(); ++i) {
        for (auto j : gen.row(i)) {
            st.insert(j);
        }
    }

    REQUIRE(gen.cols() == cols);
    REQUIRE(gen.rows() == rows);
    REQUIRE(st.size() > rows * cols / 2);
}

TEST_CASE("Get random matrix works correctly") {
    for (size_t rows = 10; rows < 60; ++rows) {
        for (size_t cols = 10; cols < 60; ++cols) {
            TestGaussMatrix(rows, cols);
        }
    }
}
