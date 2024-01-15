find_package(Eigen3)
add_executable(main main.cpp linear.cpp activation.cpp random.cpp
        loss.cpp)

target_link_libraries(main Eigen3::Eigen)
