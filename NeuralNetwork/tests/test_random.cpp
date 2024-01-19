#include "../random.h"
#include <catch.hpp>
#include <iostream>
#include "set"

using Index = Eigen::Index;
using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Tensor3D = std::vector<Tensor2D>;

TEST_CASE("Get random vector works correctly") {
    nn_random::Random rng;
    for (size_t cols = 10; cols < 100; ++cols) {
        double low = -std::sqrt(1 / cols);
        double high = std::sqrt(1 / cols);
        auto gen = rng.GetUniformVector(low, high, cols);
        std::set<double> st;
        for (auto i : gen) {
            st.insert(i);
        }
        REQUIRE(gen.cols() == cols);
        REQUIRE(st.size() > cols / 2);
    }
}

TEST_CASE("Get random matrix works correctly") {
    nn_random::Random rng;
    for (size_t rows = 10; rows < 60; ++rows) {
        for (size_t cols = 10; cols < 60; ++cols) {
            double low = -std::sqrt(1 / cols);
            double high = std::sqrt(1 / cols);
            auto gen = rng.GetUniformMatrix(low, high, rows, cols);
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
    }
}
