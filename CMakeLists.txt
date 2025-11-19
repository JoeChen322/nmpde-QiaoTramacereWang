cmake_minimum_required(VERSION 3.12.0)

# Force the compiler to match deal.II before project()
set(CMAKE_CXX_COMPILER "/u/sw/toolchains/gcc-glibc/11.2.0/prefix/bin/g++")

# Set build type to DebugRelease to match deal.II installation
set(CMAKE_BUILD_TYPE "DebugRelease" CACHE STRING "Choose the type of build")

project(WaveEquationTest LANGUAGES CXX)

# Locate deal.II and initialize its variables.
set(CMAKE_PREFIX_PATH "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1/lib/cmake/deal.II" ${CMAKE_PREFIX_PATH})
find_package(deal.II 9.5.1 REQUIRED
        HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

# Common utility sources used by multiple targets
set(COMMON_UTILITY_SOURCES
    src/core/Utilities/VTKOutput.cpp
    src/core/Utilities/EnergyCalculator.cpp
)

# Helper function to add a wave equation executable with common configuration
function(add_wave_executable TARGET_NAME)
    cmake_parse_arguments(PARSED "" "" "SOURCES" ${ARGN})
    
    add_executable(${TARGET_NAME} ${PARSED_SOURCES})
    
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/wave_solver
    )
    
    deal_ii_setup_target(${TARGET_NAME} RELEASE)
endfunction()

# 1D wave solver executables
add_wave_executable(wave_1d
    SOURCES
        src/apps/main_1d.cpp
        src/cg/CGWaveSolver.cpp
        src/core/Problems/GaussianPulse.cpp
        ${COMMON_UTILITY_SOURCES}
)

add_wave_executable(test_mms
    SOURCES
        src/apps/test_mms.cpp
        src/cg/CGWaveSolver.cpp
        src/core/Problems/ManufacturedSolution.cpp
        ${COMMON_UTILITY_SOURCES}
)

add_wave_executable(test_convergence_space
    SOURCES
        src/apps/test_convergence_space.cpp
        src/cg/CGWaveSolver.cpp
        src/core/Problems/ManufacturedSolution.cpp
        ${COMMON_UTILITY_SOURCES}
)

add_wave_executable(test_convergence_time
    SOURCES
        src/apps/test_convergence_time.cpp
        src/cg/CGWaveSolver.cpp
        src/core/Problems/ManufacturedSolution.cpp
        ${COMMON_UTILITY_SOURCES}
)

add_wave_executable(test_convergence_spacetime
    SOURCES
        src/apps/test_convergence_spacetime.cpp
        src/cg/CGWaveSolver.cpp
        src/core/Problems/ManufacturedSolution.cpp
        ${COMMON_UTILITY_SOURCES}
)

# 2D wave solver executables
add_wave_executable(wave_2d
    SOURCES
        src/apps/main_2d.cpp
        src/cg/CGWaveSolver2D.cpp
        src/core/Problems/Gaussian2DPulse.cpp
        ${COMMON_UTILITY_SOURCES}
)

add_wave_executable(test_mms_2d
    SOURCES
        src/apps/test_mms_2d.cpp
        src/cg/CGWaveSolver2D.cpp
        src/core/Problems/ManufacturedSolution2D.cpp
        ${COMMON_UTILITY_SOURCES}
)

add_wave_executable(test_convergence_space_2d
    SOURCES
        src/apps/test_convergence_space_2d.cpp
        src/cg/CGWaveSolver2D.cpp
        src/core/Problems/ManufacturedSolution2D.cpp
        ${COMMON_UTILITY_SOURCES}
)

add_wave_executable(test_convergence_time_2d
    SOURCES
        src/apps/test_convergence_time_2d.cpp
        src/cg/CGWaveSolver2D.cpp
        src/core/Problems/ManufacturedSolution2D.cpp
        ${COMMON_UTILITY_SOURCES}
)

add_wave_executable(test_convergence_spacetime_2d
    SOURCES
        src/apps/test_convergence_spacetime_2d.cpp
        src/cg/CGWaveSolver2D.cpp
        src/core/Problems/ManufacturedSolution2D.cpp
        ${COMMON_UTILITY_SOURCES}
)

# 2D parallel wave solver (minimal dependencies)
add_wave_executable(wave_2d_parallel
    SOURCES
        src/apps/wave_2d_parallel.cpp
        src/cg/CGWaveSolver2DParallel.cpp
)
