cmake_minimum_required(VERSION 3.12.0)

# Force the compiler to match deal.II before project()
set(CMAKE_CXX_COMPILER "/u/sw/toolchains/gcc-glibc/11.2.0/prefix/bin/g++")

project(WaveEquationTest LANGUAGES CXX)

# Locate deal.II and initialize its variables.
set(CMAKE_PREFIX_PATH "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1/lib/cmake/deal.II" ${CMAKE_PREFIX_PATH})
find_package(deal.II 9.5.1 REQUIRED
        HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

# Add executable
add_executable(test_wave_1d
    src/apps/main_cg.cpp
    src/cg/CGWaveSolver.cpp
    src/core/Problems/GaussianPulse.cpp
    src/core/Utilities/VTKOutput.cpp
    src/core/Utilities/ErrorCalculator.cpp
)

# Add include directories
target_include_directories(test_wave_1d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/wave_solver
)

# Setup deal.II for the target
deal_ii_setup_target(test_wave_1d)

# Set compiler flags
target_compile_options(test_wave_1d PRIVATE 
    $<$<CXX_COMPILER_ID:GNU>:-O2>
)