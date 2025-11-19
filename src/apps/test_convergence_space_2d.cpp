#include "CGWaveSolver2D.hpp"
#include "ErrorComputer.hpp"
#include "../core/Problems/ManufacturedSolution2D.cpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace WaveEquation;

/**
 * SPATIAL Convergence Test for 2D Wave Solver
 * 
 * This test fixes the time step and varies only the mesh size to isolate
 * spatial discretization error and measure the spatial convergence rate.
 * 
 * For FE with degree p, we expect O(h^{p+1}) convergence in L2 norm.
 * With linear elements (p=1), we should see second-order convergence: O(h^2)
 */

int main(int argc, char *argv[])
{
    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        
        std::cout << "=============================================================" << std::endl;
        std::cout << "2D SPATIAL Convergence Test (Fixed Time Step, Varying Mesh)" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::endl;
        
        // Test parameters
        const double wave_speed = 1.0;
        const double final_time = std::sqrt(2.0);  // One period for mode (1,1)
        
        // FIXED time step (small enough to make temporal error negligible)
        const double dt_fixed = 0.001;  // Very small to isolate spatial error
        
        // Store convergence data
        std::vector<unsigned int> levels;
        std::vector<unsigned int> n_dofs;
        std::vector<unsigned int> n_cells;
        std::vector<double> mesh_sizes;
        std::vector<double> l2_errors;
        
        // Run convergence study
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Wave equation: u_tt = c^2 * ∇²u" << std::endl;
        std::cout << "  Exact solution: u(x,y,t) = cos(π*x) * cos(π*y) * cos(ω*t)" << std::endl;
        std::cout << "  Angular frequency: ω = c*π*√2" << std::endl;
        std::cout << "  Domain: [-1, 1] × [-1, 1]" << std::endl;
        std::cout << "  Boundary conditions: ∂u/∂n = 0 (Neumann/reflecting)" << std::endl;
        std::cout << "  Elements: Linear triangular (P1 simplex)" << std::endl;
        std::cout << "  Time integrator: Crank-Nicolson (theta=0.5)" << std::endl;
        std::cout << "  FIXED time step: dt = " << std::scientific << dt_fixed << std::endl;
        std::cout << "  Expected convergence rate: O(h^2) for linear FE" << std::endl;
        std::cout << std::endl;
        
        // Test on mesh levels 0-3
        for (unsigned int level = 0; level <= 3; ++level)
        {
            std::cout << "Mesh level: " << level << std::endl;
            std::cout << "------------------------------------" << std::endl;
            
            // Create solver
            CGWaveSolver2D solver;
            
            // Set mesh
            std::string mesh_file = "../src/core/Meshes/square_level_" + std::to_string(level) + ".msh";
            solver.set_mesh_file(mesh_file);
            
            // Create manufactured solution problem
            solver.set_problem(std::make_unique<ManufacturedSolution2D<2>>(wave_speed));
            solver.set_wave_speed(wave_speed);
            
            // FIXED time step (key for spatial convergence test)
            solver.set_time_step(dt_fixed);
            solver.set_final_time(final_time);
            
            // Disable pulse injection and output for faster testing
            solver.set_pulse_injection_interval(0);
            solver.set_output_interval(0);
            
            // Setup and run
            solver.setup();
            
            const unsigned int n_dofs_current = solver.get_n_dofs();
            const unsigned int n_cells_current = solver.get_triangulation().n_active_cells();
            
            // Estimate mesh size (square root of average cell area)
            const double domain_area = 4.0;  // [-1,1] x [-1,1]
            const double avg_cell_area = domain_area / n_cells_current;
            const double h = std::sqrt(avg_cell_area);
            
            std::cout << "  Cells: " << n_cells_current << std::endl;
            std::cout << "  DoFs: " << n_dofs_current << std::endl;
            std::cout << "  Mesh size h: " << std::scientific << std::setprecision(4) << h << std::endl;
            
            solver.run();
            
            // Compute error at final time
            ManufacturedSolution2D<2> exact_solution(wave_speed);
            const double l2_error = Utilities::ErrorComputer<2>::compute_l2_error(
                solver.get_dof_handler(),
                solver.get_fe(),
                solver.get_solution_u(),
                exact_solution,
                final_time);
            
            std::cout << "  L2 error: " << std::scientific << std::setprecision(4) << l2_error << std::endl;
            std::cout << std::endl;
            
            // Store results
            levels.push_back(level);
            n_dofs.push_back(n_dofs_current);
            n_cells.push_back(n_cells_current);
            mesh_sizes.push_back(h);
            l2_errors.push_back(l2_error);
        }
        
        // Print summary table
        std::cout << "=============================================================" << std::endl;
        std::cout << "SPATIAL CONVERGENCE SUMMARY" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::endl;
        
        std::cout << std::setw(8) << "Level"
                  << std::setw(12) << "Cells"
                  << std::setw(12) << "DoFs"
                  << std::setw(14) << "h"
                  << std::setw(14) << "L2 Error"
                  << std::setw(14) << "Rate"
                  << std::endl;
        std::cout << std::string(74, '-') << std::endl;
        
        for (size_t i = 0; i < levels.size(); ++i)
        {
            std::cout << std::setw(8) << levels[i]
                      << std::setw(12) << n_cells[i]
                      << std::setw(12) << n_dofs[i]
                      << std::setw(14) << std::scientific << std::setprecision(4) << mesh_sizes[i]
                      << std::setw(14) << std::scientific << std::setprecision(4) << l2_errors[i];
            
            if (i > 0)
            {
                const double rate = std::log(l2_errors[i-1] / l2_errors[i]) / 
                                   std::log(mesh_sizes[i-1] / mesh_sizes[i]);
                std::cout << std::setw(14) << std::fixed << std::setprecision(3) << rate;
            }
            else
            {
                std::cout << std::setw(14) << "-";
            }
            
            std::cout << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "Expected rate: 2.0 (second-order for P1 elements)" << std::endl;
        std::cout << std::endl;
        
        // Check if convergence rate is reasonable
        if (levels.size() >= 2)
        {
            const size_t last = levels.size() - 1;
            const double final_rate = std::log(l2_errors[last-1] / l2_errors[last]) / 
                                     std::log(mesh_sizes[last-1] / mesh_sizes[last]);
            
            if (final_rate > 1.8 && final_rate < 2.5)
            {
                std::cout << "✓ SPATIAL CONVERGENCE TEST PASSED" << std::endl;
                std::cout << "  Observed rate (" << std::fixed << std::setprecision(2) << final_rate 
                          << ") is close to expected (2.0)" << std::endl;
                return 0;
            }
            else
            {
                std::cout << "✗ SPATIAL CONVERGENCE TEST FAILED" << std::endl;
                std::cout << "  Observed rate (" << std::fixed << std::setprecision(2) << final_rate 
                          << ") differs from expected (2.0)" << std::endl;
                return 1;
            }
        }
        
        return 0;
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
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
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
}
