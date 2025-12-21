cmake_minimum_required(VERSION 3.12.0)
project(OtherImplementation LANGUAGES CXX C)

include(./cmake-common.cmake)

add_executable(OtherImplementation
  src/Main.cpp
  src/Wave.cpp
)
deal_ii_setup_target(OtherImplementation)

add_executable(TimeConvergence
  src/TimeConvergence.cpp
  src/Wave.cpp
)
deal_ii_setup_target(TimeConvergence)

add_executable(SpaceConvergence
  src/SpaceConvergence.cpp
  src/Wave.cpp
)
deal_ii_setup_target(SpaceConvergence)
