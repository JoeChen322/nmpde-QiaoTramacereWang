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
 * SPACE-TIME Convergence Test for 2D Wave Solver
 * 
 * This test varies both mesh size and time step proportionally to measure
 * the combined convergence rate.
 * 
 * For Crank-Nicolson with linear elements, we expect O(h^2 + dt^2) overall.
 * When h and dt are reduced proportionally, we should see second-order convergence.
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
        std::cout << "2D SPACE-TIME Convergence Test (Varying Both h and dt)" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::endl;
        
        // Test parameters
        const double wave_speed = 1.0;
        const double final_time = std::sqrt(2.0);  // One period for mode (1,1)
        
        // Store convergence data
        std::vector<unsigned int> levels;
        std::vector<unsigned int> n_dofs;
        std::vector<unsigned int> n_cells;
        std::vector<double> mesh_sizes;
        std::vector<double> time_steps;
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
        std::cout << "  Strategy: Refine BOTH mesh and time step together" << std::endl;
        std::cout << "  Expected convergence rate: O(h^2) = O(dt^2) ≈ 2.0" << std::endl;
        std::cout << std::endl;
        
        // Define time steps for each mesh level (roughly proportional to h)
        std::vector<std::pair<unsigned int, double>> level_dt_pairs = {
            {0, 0.1},      // Coarse mesh, coarse time step
            {1, 0.05},     // Finer mesh, finer time step
            {2, 0.025},    // Even finer
            {3, 0.0125}    // Finest
        };
        
        for (const auto &[level, dt] : level_dt_pairs)
        {
            std::cout << "Mesh level: " << level << ", Time step: " << std::scientific 
                      << std::setprecision(4) << dt << std::endl;
            std::cout << "------------------------------------" << std::endl;
            
            // Create solver
            CGWaveSolver2D solver;
            
            // Set mesh
            std::string mesh_file = "../src/core/Meshes/square_level_" + std::to_string(level) + ".msh";
            solver.set_mesh_file(mesh_file);
            
            // Create manufactured solution problem
            solver.set_problem(std::make_unique<ManufacturedSolution2D<2>>(wave_speed));
            solver.set_wave_speed(wave_speed);
            
            // Set time step (proportional to mesh size)
            solver.set_time_step(dt);
            solver.set_final_time(final_time);
            
            // Disable pulse injection and output for faster testing
            solver.set_pulse_injection_interval(0);
            solver.set_output_interval(0);
            
            // Setup and run
            solver.setup();
            
            const unsigned int n_dofs_current = solver.get_n_dofs();
            const unsigned int n_cells_current = solver.get_triangulation().n_active_cells();
            const unsigned int n_steps = static_cast<unsigned int>(std::round(final_time / dt));
            
            // Estimate mesh size
            const double domain_area = 4.0;  // [-1,1] x [-1,1]
            const double avg_cell_area = domain_area / n_cells_current;
            const double h = std::sqrt(avg_cell_area);
            
            std::cout << "  Cells: " << n_cells_current << std::endl;
            std::cout << "  DoFs: " << n_dofs_current << std::endl;
            std::cout << "  Mesh size h: " << std::scientific << std::setprecision(4) << h << std::endl;
            std::cout << "  Time steps: " << n_steps << std::endl;
            std::cout << "  Ratio dt/h: " << std::fixed << std::setprecision(3) << (dt / h) << std::endl;
            
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
            levels.push_back(level);
            n_dofs.push_back(n_dofs_current);
            n_cells.push_back(n_cells_current);
            mesh_sizes.push_back(h);
            time_steps.push_back(dt);
            l2_errors.push_back(l2_error);
        }
        
        // Print summary table
        std::cout << "=============================================================" << std::endl;
        std::cout << "SPACE-TIME CONVERGENCE SUMMARY" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::endl;
        
        std::cout << std::setw(8) << "Level"
                  << std::setw(12) << "Cells"
                  << std::setw(12) << "DoFs"
                  << std::setw(14) << "h"
                  << std::setw(14) << "dt"
                  << std::setw(14) << "L2 Error"
                  << std::setw(14) << "Rate"
                  << std::endl;
        std::cout << std::string(88, '-') << std::endl;
        
        for (size_t i = 0; i < levels.size(); ++i)
        {
            std::cout << std::setw(8) << levels[i]
                      << std::setw(12) << n_cells[i]
                      << std::setw(12) << n_dofs[i]
                      << std::setw(14) << std::scientific << std::setprecision(4) << mesh_sizes[i]
                      << std::setw(14) << std::scientific << std::setprecision(4) << time_steps[i]
                      << std::setw(14) << std::scientific << std::setprecision(4) << l2_errors[i];
            
            if (i > 0)
            {
                // Compute rate based on mesh size (since dt ∝ h)
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
        std::cout << "Expected rate: 2.0 (second-order for h and dt refined together)" << std::endl;
        std::cout << std::endl;
        
        // Check if convergence rate is reasonable
        if (levels.size() >= 2)
        {
            const size_t last = levels.size() - 1;
            const double final_rate = std::log(l2_errors[last-1] / l2_errors[last]) / 
                                     std::log(mesh_sizes[last-1] / mesh_sizes[last]);
            
            if (final_rate > 1.8 && final_rate < 2.5)
            {
                std::cout << "✓ SPACE-TIME CONVERGENCE TEST PASSED" << std::endl;
                std::cout << "  Observed rate (" << std::fixed << std::setprecision(2) << final_rate 
                          << ") is close to expected (2.0)" << std::endl;
                return 0;
            }
            else
            {
                std::cout << "✗ SPACE-TIME CONVERGENCE TEST FAILED" << std::endl;
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
