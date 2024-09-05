cmake_minimum_required(VERSION 3.27)
project(coursework)

set(CMAKE_BUILD_TYPE Relwithdebinfo)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/TestSolution.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "-fstandalone-debug")

include_directories(./EigenRand)
include_directories(catch)

find_package(Catch REQUIRED)

add_subdirectory(eigen)
add_subdirectory(EigenRand)
add_subdirectory(NeuralNetwork)
