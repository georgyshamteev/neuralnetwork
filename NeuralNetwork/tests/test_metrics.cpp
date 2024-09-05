#include <catch.hpp>
#include <iostream>

#include "metrics.h"

TEST_CASE("test metrics") {
    /*
     data:

     class A: 3 * 5 = 15
     class B: 1 * 5 = 5
     class C: 2 * 5 = 10
     class D: 3 * 5 = 15

     */
    metrics::Tensor1D pred(45);
    pred << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 1,
        2, 2, 2, 2, 0, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3;
    metrics::Tensor1D labels(45);
    labels << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3;

    REQUIRE(Approx(metrics::Accuracy(pred, labels, {0, 1, 2, 3})).epsilon(0.01) == 0.8222);
    REQUIRE(Approx(metrics::Precision(pred, labels, {0, 1, 2, 3})).epsilon(0.01) == 0.7388);
    REQUIRE(Approx(metrics::Recall(pred, labels, {0, 1, 2, 3})).epsilon(0.01) == 0.725);
}
