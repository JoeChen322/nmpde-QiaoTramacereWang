#include "CGWaveSolver.hpp"
#include "ProblemBase.hpp"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// Include the manufactured solution problem
#include "../../src/core/Problems/ManufacturedSolution.cpp"

/**
 * TEMPORAL Convergence Test using Method of Manufactured Solutions
 * 
 * This test fixes the mesh size and varies only the time step to isolate
 * temporal discretization error and measure the temporal convergence rate.
 * 
 * For Crank-Nicolson (theta=0.5), we expect O(dt^2) convergence (second-order).
 * For implicit Euler (theta=1.0), we expect O(dt) convergence (first-order).
 */

// Helper class to compute errors
template <int dim>
class ErrorComputer
{
public:
    static double compute_l2_error(
        const dealii::DoFHandler<dim> &dof_handler,
        const dealii::Vector<double> &numerical_solution,
        const WaveEquation::ManufacturedSolution<dim> &exact_problem,
        double time)
    {
        dealii::QGauss<dim> quadrature(3);  // Use 3-point Gauss quadrature
        dealii::FEValues<dim> fe_values(dof_handler.get_fe(),
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
                const dealii::Point<dim> &q_point = fe_values.quadrature_point(q);
                const double exact_value = exact_problem.exact_solution(q_point, time);
                const double error = numerical_values[q] - exact_value;
                
                l2_error_squared += error * error * fe_values.JxW(q);
            }
        }
        
        return std::sqrt(l2_error_squared);
    }
};

int main()
{
    try
    {
        std::cout << "=============================================================" << std::endl;
        std::cout << "TEMPORAL Convergence Test (Fixed Mesh, Varying Time Step)" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::endl;
        
        // Test parameters
        const double wave_speed = 1.0;
        const double final_time = 1.0;  // Run for one period
        
        // Fixed spatial discretization
        const unsigned int refinement_fixed = 13;
        
        // Store convergence data
        std::vector<double> time_steps;
        std::vector<double> l2_errors;
        std::vector<unsigned int> n_timesteps;
        
        // Run convergence study
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Wave equation: u_tt = c^2 * u_xx" << std::endl;
        std::cout << "  Exact solution: u(x,t) = cos(π*x) * cos(ω*t), ω = c*π" << std::endl;
        std::cout << "  Domain: [-1, 1]" << std::endl;
        std::cout << "  Time integrator: Crank-Nicolson (theta=0.5)" << std::endl;
        std::cout << "  FIXED mesh refinement: " << refinement_fixed << std::endl;
        
        // Calculate mesh size
        const unsigned int n_cells = std::pow(2, refinement_fixed);
        const double h = 2.0 / n_cells;
        const unsigned int n_dofs_fixed = n_cells + 1;
        
        std::cout << "  FIXED mesh size: h = " << std::scientific << h << std::endl;
        std::cout << "  FIXED DOFs: " << n_dofs_fixed << std::endl;
        std::cout << "  Expected convergence rate: O(dt^2) for Crank-Nicolson" << std::endl;
        std::cout << std::endl;
        
        // Vary time step: start with dt = 0.1, then halve it each time
        std::vector<double> dt_values = {0.1, 0.05, 0.025, 0.0125, 0.00625};
        
        for (const double dt : dt_values)
        {
            std::cout << "Time step: dt = " << std::scientific << std::setprecision(4) << dt << std::endl;
            std::cout << "------------------------------------" << std::endl;
            
            // Create solver
            WaveEquation::CGWaveSolver<1> solver;
            
            // Create manufactured solution problem
            auto problem = std::make_unique<WaveEquation::ManufacturedSolution<1>>(wave_speed);
            const WaveEquation::ManufacturedSolution<1> *problem_ptr = problem.get();
            solver.set_problem(std::move(problem));
            
            // Configure solver
            solver.set_mesh_refinements(refinement_fixed);  // FIXED mesh
            solver.set_final_time(final_time);
            solver.set_wave_speed(wave_speed);
            
            // VARYING time step (key for temporal convergence test)
            solver.set_time_step(dt);
            
            // Disable pulse injection for MMS test
            solver.set_pulse_injection_interval(1000000);
            
            // Reduce output frequency
            solver.set_output_interval(1000);
            
            // Setup and run
            solver.setup();
            solver.run();
            
            // Compute error at the ACTUAL final time reached by the solver
            // Note: The solver stops when time >= final_time, so we need to compute
            // the actual time from the number of steps taken
            const unsigned int actual_steps = static_cast<unsigned int>(std::ceil(final_time / dt));
            const double actual_final_time = actual_steps * dt;
            
            const double l2_error = ErrorComputer<1>::compute_l2_error(
                solver.get_dof_handler(),
                solver.get_solution_u(),
                *problem_ptr,
                actual_final_time);
            
            // Store results
            time_steps.push_back(dt);
            l2_errors.push_back(l2_error);
            n_timesteps.push_back(actual_steps);
            
            std::cout << "  Number of time steps: " << actual_steps << std::endl;
            std::cout << "  final_time / dt:      " << std::fixed << std::setprecision(16) << (final_time / dt) << std::endl;
            std::cout << "  Actual final time:    " << std::fixed << std::setprecision(16) << actual_final_time << std::endl;
            std::cout << "  Time step (dt):       " << std::scientific << std::setprecision(16) << dt << std::endl;
            std::cout << "  L2 Error:             " << std::scientific << std::setprecision(6) << l2_error << std::endl;
            std::cout << std::endl;
        }
        
        // Print convergence table
        std::cout << "=============================================================" << std::endl;
        std::cout << "TEMPORAL Convergence Results" << std::endl;
        std::cout << "=============================================================" << std::endl;
        std::cout << std::setw(15) << "dt"
                  << std::setw(15) << "# Steps"
                  << std::setw(15) << "L2 Error"
                  << std::setw(15) << "Rate"
                  << std::endl;
        std::cout << "-------------------------------------------------------------" << std::endl;
        
        for (size_t i = 0; i < time_steps.size(); ++i)
        {
            std::cout << std::setw(15) << std::scientific << std::setprecision(4) << time_steps[i]
                      << std::setw(15) << n_timesteps[i]
                      << std::setw(15) << std::scientific << std::setprecision(6) << l2_errors[i];
            
            if (i > 0)
            {
                const double rate = std::log(l2_errors[i-1] / l2_errors[i]) / 
                                   std::log(time_steps[i-1] / time_steps[i]);
                std::cout << std::setw(15) << std::fixed << std::setprecision(2) << rate;
            }
            else
            {
                std::cout << std::setw(15) << "-";
            }
            
            std::cout << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "Expected temporal convergence rate: ~2.00 (O(dt^2) for Crank-Nicolson)" << std::endl;
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
