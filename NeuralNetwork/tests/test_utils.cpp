#include <catch.hpp>
#include <iostream>

#include "dataloader.h"

TEST_CASE("DataLoader iterates, no shuffle") {
    using Tensor2D = utils::Tensor2D;

    Tensor2D data(7, 4);
    data.setRandom();
    Tensor2D labels(7, 1);
    labels << 1, 2, 3, 4, 5, 6, 7;
    size_t k = 0;

    std::cout << "Data:" << std::endl;
    std::cout << data << std::endl;
    std::cout << "Labels:" << std::endl;
    std::cout << labels << std::endl;

    //// utils::DataLoader::shuffle::true is too long, how can be fixed?
    for (auto i : utils::DataLoader(data, labels, 2, utils::DataLoader::shuffle::False)) {
        std::cout << "Batch number " << k << ':' << std::endl;
        std::cout << i.data.eval() << std::endl;
        std::cout << "-----------------" << std::endl;
        std::cout << i.labels.eval() << std::endl;
        std::cout << "-----------------" << std::endl;
    }
}

TEST_CASE("DataLoader iterates, shuffle") {
    using Tensor2D = utils::Tensor2D;

    Tensor2D data(7, 4);
    data.setRandom();
    Tensor2D labels(7, 1);
    labels << 1, 2, 3, 4, 5, 6, 7;
    size_t k = 0;

    std::cout << "Data:" << std::endl;
    std::cout << data << std::endl;
    std::cout << "Labels:" << std::endl;
    std::cout << labels << std::endl;

    //// utils::DataLoader::shuffle::true is too long, how can be fixed?
    for (auto i : utils::DataLoader(data, labels, 2, utils::DataLoader::shuffle::True)) {
        std::cout << "Batch number " << k << ':' << std::endl;
        std::cout << i.data.eval() << std::endl;
        std::cout << "-----------------" << std::endl;
        std::cout << i.labels.eval() << std::endl;
        std::cout << "-----------------" << std::endl;
    }
}

TEST_CASE("Test ReadCSV") {
    std::string s = "/home/georgyshamteev/archive/mnist_test.csv";
    std::cout << s;
    REQUIRE_NOTHROW(utils::ReadCSV(s));
}
