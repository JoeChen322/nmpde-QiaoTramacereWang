#include "CGWaveSolver2D.hpp"
#include "../core/Problems/ManufacturedSolution2D.cpp"
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/mapping_fe.h>
#include <iostream>
#include <iomanip>

using namespace WaveEquation;

/**
 * 2D Method of Manufactured Solutions Test
 * 
 * Verifies the correctness of the 2D wave solver by comparing against
 * a known analytical solution.
 */

int main(int argc, char *argv[])
{
    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        
        std::cout << "=== 2D Method of Manufactured Solutions Test ===" << std::endl;
        std::cout << "Testing the 2D wave solver with known analytical solution" << std::endl;
        std::cout << std::endl;
        
        // Problem parameters
        const double c = 1.0;  // Wave speed
        ManufacturedSolution2D<2> mms_problem(c);
        
        std::cout << "Manufactured solution: u(x,y,t) = cos(π*x)*cos(π*y)*cos(ω*t)" << std::endl;
        std::cout << "Wave speed c = " << c << std::endl;
        std::cout << "Angular frequency ω = c*π*√2 = " << mms_problem.get_omega() << std::endl;
        std::cout << "Period T = 2π/ω = " << mms_problem.get_period() << " seconds" << std::endl;
        std::cout << "Domain: [-1, 1] × [-1, 1]" << std::endl;
        std::cout << "Boundary conditions: ∂u/∂n = 0 (Neumann/reflecting)" << std::endl;
        std::cout << "Source term: f = 0 (exact solution of homogeneous equation)" << std::endl;
        std::cout << std::endl;
        
        // Test parameters
        std::cout << "Testing convergence on different mesh refinements:" << std::endl;
        std::cout << std::endl;
        
        // Run tests on multiple refinement levels
        std::vector<unsigned int> levels = {0, 1, 2, 3};
        std::vector<double> l2_errors;
        std::vector<unsigned int> n_dofs;
        
        for (unsigned int level : levels)
        {
            // Create solver
            CGWaveSolver2D solver;
            
            // Set mesh
            std::string mesh_file = "../src/core/Meshes/square_level_" + std::to_string(level) + ".msh";
            solver.set_mesh_file(mesh_file);
            
            // Set problem (need to create new instance each iteration)
            solver.set_problem(std::make_unique<ManufacturedSolution2D<2>>(c));
            solver.set_wave_speed(c);
            solver.set_pulse_injection_interval(0);  // Disable pulse injection for MMS test
            
            // Time parameters: run for exactly one period (T = 2π/(c*π*√2) = √2/c)
            const double T = std::sqrt(2.0) / c;  // Period for mode (1,1)
            const double dt = 0.01;  // Fixed timestep
            
            solver.set_time_step(dt);
            solver.set_final_time(T);
            
            // Disable output for faster testing
            solver.set_output_interval(0);
            
            std::cout << "Level " << level << ": ";
            std::cout.flush();
            
            // Setup and run
            solver.setup();
            solver.run();
            
            // Get solution vector
            const auto &u_h = solver.get_solution_u();  // Get displacement solution
            
            // Compute L2 error at final time (should return to initial condition)
            dealii::MappingFE<2> mapping(solver.get_fe());
            dealii::Vector<double> difference_per_cell(solver.get_triangulation().n_active_cells());
            
            // Get problem and create a local copy for time setting
            ManufacturedSolution2D<2> mms_final(c);
            mms_final.set_time(T);
            dealii::VectorTools::integrate_difference(
                mapping,
                solver.get_dof_handler(),
                u_h,
                mms_final,
                difference_per_cell,
                dealii::QGaussSimplex<2>(3),
                dealii::VectorTools::L2_norm);
            
            const double l2_error = dealii::VectorTools::compute_global_error(
                solver.get_triangulation(),
                difference_per_cell,
                dealii::VectorTools::L2_norm);
            
            l2_errors.push_back(l2_error);
            n_dofs.push_back(solver.get_n_dofs());
            
            std::cout << "DoFs = " << std::setw(6) << solver.get_n_dofs()
                      << ", L2 error = " << std::scientific << std::setprecision(4) 
                      << l2_error << std::endl;
        }
        
        // Compute convergence rates
        std::cout << std::endl;
        std::cout << "Convergence rates:" << std::endl;
        for (size_t i = 1; i < l2_errors.size(); ++i)
        {
            const double rate = std::log(l2_errors[i-1] / l2_errors[i]) / 
                               std::log(std::sqrt(static_cast<double>(n_dofs[i]) / n_dofs[i-1]));
            std::cout << "Level " << levels[i-1] << " -> " << levels[i] << ": "
                      << "rate = " << std::fixed << std::setprecision(3) << rate << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "Expected convergence rate: ~2.0 for P1 elements" << std::endl;
        std::cout << std::endl;
        
        // Success criterion
        const double final_error = l2_errors.back();
        const double tolerance = 1e-2;  // Reasonable tolerance for level 3 mesh
        
        if (final_error < tolerance)
        {
            std::cout << "✓ TEST PASSED: Final L2 error (" << std::scientific 
                      << final_error << ") < tolerance (" << tolerance << ")" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "✗ TEST FAILED: Final L2 error (" << std::scientific 
                      << final_error << ") >= tolerance (" << tolerance << ")" << std::endl;
            return 1;
        }
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
}
