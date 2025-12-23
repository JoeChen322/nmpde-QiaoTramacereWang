#include "CGWaveSolver.hpp"
#include "ErrorComputer.hpp"
#include "ProblemBase.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// Include the manufactured solution problem
#include "../../src/core/Problems/ManufacturedSolution.cpp"

/**
 * SPATIAL Convergence Test using Method of Manufactured Solutions
 * 
 * This test fixes the time step and varies only the mesh size to isolate
 * spatial discretization error and measure the spatial convergence rate.
 * 
 * For FE with degree p, we expect O(h^{p+1}) convergence in L2 norm.
 * With linear elements (p=1), we should see second-order convergence: O(h^2)
 */

int main()
{
    try
    {
        std::cout << "=============================================================" << std::endl;
        std::cout << "SPATIAL Convergence Test (Fixed Time Step, Varying Mesh)" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::endl;
        
        // Test parameters
        const double wave_speed = 1.0;
        const double final_time = 1.0;  // Run for one period
        
        // FIXED time step (small enough to make temporal error negligible)
        const double dt_fixed = 0.001;  // Very small to isolate spatial error
        
        // Store convergence data
        std::vector<unsigned int> refinements;
        std::vector<unsigned int> n_dofs;
        std::vector<double> mesh_sizes;
        std::vector<double> time_steps;
        std::vector<double> l2_errors;
        
        // Run convergence study
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Wave equation: u_tt = c^2 * u_xx" << std::endl;
        std::cout << "  Exact solution: u(x,t) = cos(π*x) * cos(ω*t), ω = c*π" << std::endl;
        std::cout << "  Domain: [-1, 1]" << std::endl;
        std::cout << "  Time integrator: Crank-Nicolson (theta=0.5)" << std::endl;
        std::cout << "  FIXED time step: dt = " << std::scientific << dt_fixed << std::endl;
        std::cout << "  Expected convergence rate: O(h^2) for linear FE" << std::endl;
        std::cout << std::endl;
        
        for (unsigned int refinement = 3; refinement <= 7; ++refinement)
        {
            std::cout << "Refinement level: " << refinement << std::endl;
            std::cout << "------------------------------------" << std::endl;
            
            // Create solver
            WaveEquation::CGWaveSolver<1> solver;
            
            // Create manufactured solution problem
            auto problem = std::make_unique<WaveEquation::ManufacturedSolution<1>>(wave_speed);
            const WaveEquation::ManufacturedSolution<1> *problem_ptr = problem.get();
            solver.set_problem(std::move(problem));
            
            // Configure solver
            solver.set_mesh_refinements(refinement);
            solver.set_final_time(final_time);
            solver.set_wave_speed(wave_speed);
            
            // FIXED time step (key for spatial convergence test)
            solver.set_time_step(dt_fixed);
            
            // Disable pulse injection for MMS test
            solver.set_pulse_injection_interval(1000000);
            
            // Reduce output frequency for finer meshes
            solver.set_output_interval(1000);
            
            // Calculate mesh size
            const unsigned int n_cells = std::pow(2, refinement);
            const double h = 2.0 / n_cells;  // mesh size for [-1, 1]
            
            // Setup and run
            solver.setup();
            solver.run();
            
            // Compute error
            const double l2_error = WaveEquation::Utilities::ErrorComputer<1>::compute_l2_error(
                solver.get_dof_handler(),
                solver.get_solution_u(),
                *problem_ptr,
                final_time);
            
            // Store results
            refinements.push_back(refinement);
            n_dofs.push_back(solver.get_dof_handler().n_dofs());
            mesh_sizes.push_back(h);
            time_steps.push_back(dt_fixed);
            l2_errors.push_back(l2_error);
            
            std::cout << "  Number of DOFs: " << n_dofs.back() << std::endl;
            std::cout << "  Mesh size (h):  " << std::scientific << std::setprecision(4) << h << std::endl;
            std::cout << "  Time step (dt): " << std::scientific << std::setprecision(4) << dt_fixed << " (FIXED)" << std::endl;
            std::cout << "  L2 Error:       " << std::scientific << std::setprecision(6) << l2_error << std::endl;
            std::cout << std::endl;
        }
        
        // Print convergence table
        std::cout << "=============================================================" << std::endl;
        std::cout << "SPATIAL Convergence Results" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::setw(10) << "Refine"
                  << std::setw(12) << "DOFs"
                  << std::setw(15) << "h"
                  << std::setw(15) << "L2 Error"
                  << std::setw(15) << "Rate"
                  << std::endl;
        std::cout << "-------------------------------------------------------------" << std::endl;
        
        for (size_t i = 0; i < refinements.size(); ++i)
        {
            std::cout << std::setw(10) << refinements[i]
                      << std::setw(12) << n_dofs[i]
                      << std::setw(15) << std::scientific << std::setprecision(4) << mesh_sizes[i]
                      << std::setw(15) << std::scientific << std::setprecision(6) << l2_errors[i];
            
            if (i > 0)
            {
                const double rate = std::log(l2_errors[i-1] / l2_errors[i]) / 
                                   std::log(mesh_sizes[i-1] / mesh_sizes[i]);
                std::cout << std::setw(15) << std::fixed << std::setprecision(2) << rate;
            }
            else
            {
                std::cout << std::setw(15) << "-";
            }
            
            std::cout << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "Expected spatial convergence rate: ~2.00 (O(h^2) for linear FE)" << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }
    
    return 0;
}
