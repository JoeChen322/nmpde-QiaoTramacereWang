#include "CGWaveSolver2D.hpp"
#include "../core/Problems/ManufacturedSolution2D.cpp"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace WaveEquation;

/**
 * TEMPORAL Convergence Test for 2D Wave Solver
 * 
 * This test fixes the mesh size and varies only the time step to isolate
 * temporal discretization error and measure the temporal convergence rate.
 * 
 * For Crank-Nicolson (theta=0.5), we expect O(dt^2) convergence (second-order).
 */

// Helper class to compute errors for 2D simplex elements
class ErrorComputer2D
{
public:
    static double compute_l2_error(
        const dealii::DoFHandler<2> &dof_handler,
        const dealii::FiniteElement<2> &fe,
        const dealii::Vector<double> &numerical_solution,
        const ManufacturedSolution2D<2> &exact_problem,
        double time)
    {
        dealii::QGaussSimplex<2> quadrature(3);
        dealii::MappingFE<2> mapping(fe);
        dealii::FEValues<2> fe_values(mapping,
                                     fe,
                                     quadrature,
                                     dealii::update_values |
                                     dealii::update_quadrature_points |
                                     dealii::update_JxW_values);
        
        const unsigned int n_q_points = quadrature.size();
        std::vector<double> numerical_values(n_q_points);
        
        double l2_error_squared = 0.0;
        
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            fe_values.get_function_values(numerical_solution, numerical_values);
            
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const dealii::Point<2> &q_point = fe_values.quadrature_point(q);
                const double exact_value = exact_problem.exact_solution(q_point, time);
                const double error = numerical_values[q] - exact_value;
                
                l2_error_squared += error * error * fe_values.JxW(q);
            }
        }
        
        return std::sqrt(l2_error_squared);
    }
};

int main(int argc, char *argv[])
{
    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        
        std::cout << "=============================================================" << std::endl;
        std::cout << "2D TEMPORAL Convergence Test (Fixed Mesh, Varying Time Step)" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::endl;
        
        // Test parameters
        const double wave_speed = 1.0;
        const double final_time = std::sqrt(2.0);  // One period for mode (1,1)
        
        // FIXED mesh (fine enough to make spatial error negligible)
        const unsigned int level_fixed = 5;
        
        // Store convergence data
        std::vector<double> time_steps;
        std::vector<double> l2_errors;
        std::vector<unsigned int> n_timesteps;
        
        // Run convergence study
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Wave equation: u_tt = c^2 * ∇²u" << std::endl;
        std::cout << "  Exact solution: u(x,y,t) = cos(π*x) * cos(π*y) * cos(ω*t)" << std::endl;
        std::cout << "  Angular frequency: ω = c*π*√2" << std::endl;
        std::cout << "  Domain: [-1, 1] × [-1, 1]" << std::endl;
        std::cout << "  Boundary conditions: ∂u/∂n = 0 (Neumann/reflecting)" << std::endl;
        std::cout << "  Elements: Linear triangular (P1 simplex)" << std::endl;
        std::cout << "  Time integrator: Crank-Nicolson (theta=0.5)" << std::endl;
        std::cout << "  FIXED mesh level: " << level_fixed << std::endl;
        std::cout << "  Expected convergence rate: O(dt^2) for Crank-Nicolson" << std::endl;
        std::cout << std::endl;
        
        // Vary time step: adjusted for CFL < 1 with h ~ 0.0156
        std::vector<double> dt_values = {0.01, 0.005, 0.0025, 0.00125, 0.000625};
        
        for (const double dt : dt_values)
        {
            std::cout << "Time step: dt = " << std::scientific << std::setprecision(4) << dt << std::endl;
            std::cout << "------------------------------------" << std::endl;
            
            // Create solver
            CGWaveSolver2D solver;
            
            // Set fixed mesh
            std::string mesh_file = "../src/core/Meshes/square_level_" + std::to_string(level_fixed) + ".msh";
            solver.set_mesh_file(mesh_file);
            
            // Create manufactured solution problem
            solver.set_problem(std::make_unique<ManufacturedSolution2D<2>>(wave_speed));
            solver.set_wave_speed(wave_speed);
            
            // Set VARYING time step (key for temporal convergence test)
            solver.set_time_step(dt);
            solver.set_final_time(final_time);
            
            // Disable pulse injection and output for faster testing
            solver.set_pulse_injection_interval(0);
            solver.set_output_interval(0);
            
            // Setup and run
            solver.setup();
            
            const unsigned int n_dofs = solver.get_n_dofs();
            const unsigned int n_steps = static_cast<unsigned int>(std::round(final_time / dt));
            
            std::cout << "  DoFs: " << n_dofs << std::endl;
            std::cout << "  Time steps: " << n_steps << std::endl;
            
            solver.run();
            
            // Compute error at final time
            ManufacturedSolution2D<2> exact_solution(wave_speed);
            const double l2_error = ErrorComputer2D::compute_l2_error(
                solver.get_dof_handler(),
                solver.get_fe(),
                solver.get_solution_u(),
                exact_solution,
                final_time);
            
            std::cout << "  L2 error: " << std::scientific << std::setprecision(4) << l2_error << std::endl;
            std::cout << std::endl;
            
            // Store results
            time_steps.push_back(dt);
            n_timesteps.push_back(n_steps);
            l2_errors.push_back(l2_error);
        }
        
        // Print summary table
        std::cout << "=============================================================" << std::endl;
        std::cout << "TEMPORAL CONVERGENCE SUMMARY" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::endl;
        
        std::cout << std::setw(14) << "dt"
                  << std::setw(12) << "Steps"
                  << std::setw(14) << "L2 Error"
                  << std::setw(14) << "Rate"
                  << std::endl;
        std::cout << std::string(54, '-') << std::endl;
        
        for (size_t i = 0; i < time_steps.size(); ++i)
        {
            std::cout << std::setw(14) << std::scientific << std::setprecision(4) << time_steps[i]
                      << std::setw(12) << n_timesteps[i]
                      << std::setw(14) << std::scientific << std::setprecision(4) << l2_errors[i];
            
            if (i > 0)
            {
                const double rate = std::log(l2_errors[i-1] / l2_errors[i]) / 
                                   std::log(time_steps[i-1] / time_steps[i]);
                std::cout << std::setw(14) << std::fixed << std::setprecision(3) << rate;
            }
            else
            {
                std::cout << std::setw(14) << "-";
            }
            
            std::cout << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "Expected rate: 2.0 (second-order for Crank-Nicolson)" << std::endl;
        std::cout << std::endl;
        
        // Check if convergence rate is reasonable
        if (time_steps.size() >= 2)
        {
            const size_t last = time_steps.size() - 1;
            const double final_rate = std::log(l2_errors[last-1] / l2_errors[last]) / 
                                     std::log(time_steps[last-1] / time_steps[last]);
            
            if (final_rate > 1.8 && final_rate < 2.5)
            {
                std::cout << "✓ TEMPORAL CONVERGENCE TEST PASSED" << std::endl;
                std::cout << "  Observed rate (" << std::fixed << std::setprecision(2) << final_rate 
                          << ") is close to expected (2.0)" << std::endl;
                return 0;
            }
            else
            {
                std::cout << "✗ TEMPORAL CONVERGENCE TEST FAILED" << std::endl;
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
