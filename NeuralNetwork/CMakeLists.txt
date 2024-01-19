find_package(Eigen3)

add_catch(test_network tests/test_linear.cpp activation.cpp
        linear.cpp
        loss.cpp
        random.cpp
        sequential.cpp)

target_link_libraries(test_network Eigen3::Eigen)



