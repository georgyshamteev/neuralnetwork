cmake_minimum_required(VERSION 3.27)
project(coursework)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

include_directories(./EigenRand)

add_subdirectory(eigen)
add_subdirectory(EigenRand)
add_subdirectory(NeuralNetwork)
