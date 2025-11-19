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
 * Test with manufactured solution that computes and prints errors at each step
 * This verifies that the numerical solution remains accurate over time
 */

// Helper function to compute L2 error
template <int dim>
double compute_l2_error(
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::Vector<double> &numerical_solution,
    const WaveEquation::ManufacturedSolution<dim> &exact_problem,
    double time)
{
    dealii::QGauss<dim> quadrature(3);
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

// Custom solver that prints errors
template <int dim>
class MMSTestSolver : public WaveEquation::CGWaveSolver<dim>
{
public:
    void run_with_error_tracking(const WaveEquation::ManufacturedSolution<dim> *exact_problem)
    {
        std::cout << "Running Wave Equation Solver with Error Tracking" << std::endl;
        std::cout << "=================================================" << std::endl;
        
        // Store error data
        std::vector<double> times;
        std::vector<double> errors;
        
        // Call parent setup
        this->setup();
        
        // Custom run loop with error tracking
        const double final_time = this->final_time_;
        const double time_step = this->time_step_;
        const unsigned int output_interval = this->output_interval_;
        
        this->apply_initial_conditions();
        
        unsigned int step = 0;
        double time = 0.0;
        
        // Compute initial error
        double l2_error = compute_l2_error(this->dof_handler_, this->solution_u_, *exact_problem, time);
        times.push_back(time);
        errors.push_back(l2_error);
        
        std::cout << std::endl;
        std::cout << "Time Evolution of L2 Error:" << std::endl;
        std::cout << std::setw(10) << "Step"
                  << std::setw(15) << "Time"
                  << std::setw(15) << "L2 Error"
                  << std::setw(15) << "Energy"
                  << std::endl;
        std::cout << std::string(55, '-') << std::endl;
        
        std::cout << std::setw(10) << step
                  << std::setw(15) << std::scientific << std::setprecision(6) << time
                  << std::setw(15) << std::scientific << std::setprecision(6) << l2_error
                  << std::setw(15) << std::scientific << std::setprecision(6) << this->compute_energy()
                  << std::endl;
        
        this->output_results(step, time);
        
        // Time stepping loop
        while (time < final_time)
        {
            time += time_step;
            ++step;
            
            this->solve_time_step(time);
            
            // Compute error at this step
            l2_error = compute_l2_error(this->dof_handler_, this->solution_u_, *exact_problem, time);
            times.push_back(time);
            errors.push_back(l2_error);
            
            // Print error periodically
            if (step % output_interval == 0 || step == 1)
            {
                std::cout << std::setw(10) << step
                          << std::setw(15) << std::scientific << std::setprecision(6) << time
                          << std::setw(15) << std::scientific << std::setprecision(6) << l2_error
                          << std::setw(15) << std::scientific << std::setprecision(6) << this->compute_energy()
                          << std::endl;
            }
            
            if (step % output_interval == 0)
            {
                this->output_results(step, time);
            }
            
            // Update old solutions
            this->old_solution_u_ = this->solution_u_;
            this->old_solution_v_ = this->solution_v_;
        }
        
        // Final output
        this->output_results(step, time);
        
        std::cout << std::string(55, '-') << std::endl;
        std::cout << std::endl;
        std::cout << "Simulation Summary:" << std::endl;
        std::cout << "  Total steps:     " << step << std::endl;
        std::cout << "  Final time:      " << time << std::endl;
        std::cout << "  Initial error:   " << std::scientific << std::setprecision(6) << errors.front() << std::endl;
        std::cout << "  Final error:     " << std::scientific << std::setprecision(6) << errors.back() << std::endl;
        std::cout << "  Max error:       " << std::scientific << std::setprecision(6) 
                  << *std::max_element(errors.begin(), errors.end()) << std::endl;
        std::cout << "  Final energy:    " << std::scientific << std::setprecision(6) << this->compute_energy() << std::endl;
        
        // Check if error remains bounded
        const double max_error = *std::max_element(errors.begin(), errors.end());
        
        if (max_error < 0.1 * std::sqrt(2.0))  // Error should be small
        {
            std::cout << "  ✓ Error remains bounded and small!" << std::endl;
        }
        else
        {
            std::cout << "  ✗ Warning: Error is larger than expected!" << std::endl;
        }
    }
    
private:
    using WaveEquation::CGWaveSolver<dim>::final_time_;
    using WaveEquation::CGWaveSolver<dim>::time_step_;
    using WaveEquation::CGWaveSolver<dim>::output_interval_;
    using WaveEquation::CGWaveSolver<dim>::dof_handler_;
    using WaveEquation::CGWaveSolver<dim>::solution_u_;
    using WaveEquation::CGWaveSolver<dim>::solution_v_;
    using WaveEquation::CGWaveSolver<dim>::old_solution_u_;
    using WaveEquation::CGWaveSolver<dim>::old_solution_v_;
};

int main()
{
    try
    {
        std::cout << "==================================================" << std::endl;
        std::cout << "MMS Test: Wave Equation with Error Tracking" << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << std::endl;
        std::cout << "Exact solution: u(x,t) = cos(π*x) * cos(π*t)" << std::endl;
        std::cout << "Domain: [-1, 1] with Neumann BCs" << std::endl;
        std::cout << std::endl;
        
        // Create solver
        MMSTestSolver<1> solver;
        
        // Create and set manufactured solution problem
        auto problem = std::make_unique<WaveEquation::ManufacturedSolution<1>>(1.0);
        const WaveEquation::ManufacturedSolution<1> *problem_ptr = problem.get();
        solver.set_problem(std::move(problem));
        
        // Configure solver
        solver.set_mesh_refinements(6);  // 64 cells
        solver.set_final_time(2.0);      // Run for 2 seconds
        solver.set_time_step(0.005);     // Small time step for accuracy
        solver.set_output_interval(20);  // Print every 20 steps
        solver.set_wave_speed(1.0);
        solver.set_pulse_injection_interval(1000000);  // Disable pulse injection
        
        // Run with error tracking
        solver.run_with_error_tracking(problem_ptr);
        
        std::cout << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
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
    
    return 0;
}
