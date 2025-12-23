cmake_minimum_required(VERSION 3.12.0)
project(WaveEquationSolver LANGUAGES CXX C)

# Include common CMake configuration (MPI, Boost, deal.II)
include(./cmake-common.cmake)

# Include directory for headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Main wave simulation executable
add_executable(WaveSolver
  src/Main.cpp
  src/Wave.cpp
)
deal_ii_setup_target(WaveSolver RELEASE)

# Time convergence study
add_executable(TimeConvergence
  src/TimeConvergence.cpp
  src/Wave.cpp
)
deal_ii_setup_target(TimeConvergence RELEASE)

# Space convergence study
add_executable(SpaceConvergence
  src/SpaceConvergence.cpp
  src/Wave.cpp
)
deal_ii_setup_target(SpaceConvergence RELEASE)

# Dissipation study (theta-scheme energy decay analysis)
add_executable(DissipationStudy
  src/DissipationStudy.cpp
  src/Wave.cpp
)
deal_ii_setup_target(DissipationStudy RELEASE)

# Dispersion study (phase shift analysis)
add_executable(DispersionOmegaDtStudy
  src/DispersionOmegaDtStudy.cpp
  src/Wave.cpp
)
deal_ii_setup_target(DispersionOmegaDtStudy RELEASE)

# Dispersion spatial property study
add_executable(DispersionSpatialStudy
  src/DispersionSpatialStudy.cpp
  src/Wave.cpp    
)
deal_ii_setup_target(DispersionSpatialStudy RELEASE)
