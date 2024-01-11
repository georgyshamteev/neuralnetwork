find_package(Eigen3)
add_executable(main main.cpp)

target_link_libraries(main Eigen3::Eigen)