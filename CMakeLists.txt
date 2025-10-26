cmake_minimum_required(VERSION 3.12.0)
project(project2 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(project2 src_hyper/main.cpp src_hyper/Hyperbolic.cpp)
deal_ii_setup_target(project2)