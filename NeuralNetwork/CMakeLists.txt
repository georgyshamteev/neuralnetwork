find_package(Eigen3 3.3.4 REQUIRED NO_MODULE)
add_executable(main main.cpp linear.cpp)

target_link_libraries(main Eigen3::Eigen)
