find_package(Eigen3)

include(sources.cmake)

add_catch(test_linear tests/test_linear.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        )

target_link_libraries(test_linear Eigen3::Eigen)

add_catch(test_random tests/test_random.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        )

target_link_libraries(test_random Eigen3::Eigen)

add_catch(test_activation tests/test_activation.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        )

target_link_libraries(test_activation Eigen3::Eigen)

add_catch(test_loss tests/test_loss.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        )

target_link_libraries(test_loss Eigen3::Eigen)

add_catch(test_sequential tests/test_sequential.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp
        )

target_link_libraries(test_sequential Eigen3::Eigen)
