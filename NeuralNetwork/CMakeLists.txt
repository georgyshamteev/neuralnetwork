find_package(Eigen3)

include(sources.cmake)

add_catch(test_linear tests/test_linear.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp)

target_link_libraries(test_linear Eigen3::Eigen)

add_catch(test_random tests/test_random.cpp
        activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp)

target_link_libraries(test_random Eigen3::Eigen)



