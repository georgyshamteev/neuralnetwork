find_package(Eigen3)

include(sources.cmake)

add_catch(test_linear tests/test_linear.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        optimizer.cpp
        )

target_link_libraries(test_linear Eigen3::Eigen)

add_catch(test_random tests/test_random.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        optimizer.cpp
        )

target_link_libraries(test_random Eigen3::Eigen)

add_catch(test_activation tests/test_activation.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        optimizer.cpp
        )

target_link_libraries(test_activation Eigen3::Eigen)

add_catch(test_loss tests/test_loss.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        optimizer.cpp
        )

target_link_libraries(test_loss Eigen3::Eigen)

add_catch(test_sequential tests/test_sequential.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        optimizer.cpp
        dataloader.cpp
        )

target_link_libraries(test_sequential Eigen3::Eigen)

add_catch(test_optimizer tests/test_optimizer.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        optimizer.cpp
)

target_link_libraries(test_optimizer Eigen3::Eigen)

add_catch(test_utils tests/test_utils.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        optimizer.cpp
        dataloader.cpp
)

target_link_libraries(test_utils Eigen3::Eigen)

add_catch(test_metrics tests/test_metrics.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        optimizer.cpp
        dataloader.cpp
        metrics.cpp
)

target_link_libraries(test_metrics Eigen3::Eigen)

add_executable(test_mnist tests/test_mnist.cpp
        activation.cpp
        activation.h
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        optimizer.cpp
        dataloader.cpp
        metrics.cpp
)

target_link_libraries(test_mnist Eigen3::Eigen)
