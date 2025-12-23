#pragma once

#include "WaveSolverBase.hpp"
#include "ProblemBase.hpp"

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

namespace WaveEquation
{
    template <int dim>
    class CGWaveSolver : public WaveSolverBase<dim>
    {
    public:
        CGWaveSolver();
        
        void set_problem(std::unique_ptr<ProblemBase<dim>> problem);
        
        // Configuration setters for MMS and testing
        void set_mesh_refinements(unsigned int refinements) { mesh_refinements_ = refinements; }
        void set_final_time(double final_time) { final_time_ = final_time; }
        void set_time_step(double time_step) { time_step_ = time_step; }
        void set_output_interval(unsigned int interval) { output_interval_ = interval; }
        void set_wave_speed(double wave_speed) { wave_speed_ = wave_speed; }
        void set_pulse_injection_interval(unsigned int interval) { input_pulse_interval_ = interval; }
        
        // Getters for MMS and testing
        const dealii::DoFHandler<dim>& get_dof_handler() const { return dof_handler_; }
        const dealii::Vector<double>& get_solution_u() const { return solution_u_; }
        const dealii::Vector<double>& get_solution_v() const { return solution_v_; }
        const ProblemBase<dim>* get_problem() const { return problem_.get(); }
        
        // Main interface implementation
        void setup() override;
        void run() override;
        
        // Analysis
        double compute_energy() const override;
        
        // Output
        void output_results(unsigned int step, double time) const override;

    protected:
        // Protected methods for derived classes (e.g., for testing)
        void apply_initial_conditions();
        void solve_time_step(double time);

    private:
        // Setup methods
        void create_mesh();
        void setup_dofs();
        void setup_matrices();
        void setup_vectors();
        
        // Assembly methods
        void assemble_mass_matrix();
        void assemble_stiffness_matrix();
        
        // Solution methods
        void inject_pulse();  // Inject a new pulse at current time
        
    protected:
        // Protected member variables (accessible by derived classes for testing)
        std::unique_ptr<ProblemBase<dim>> problem_;
        
        // Finite element infrastructure
        dealii::Triangulation<dim> triangulation_;
        dealii::FE_Q<dim> fe_;
        dealii::DoFHandler<dim> dof_handler_;
        
        // Linear algebra objects
        dealii::SparsityPattern sparsity_pattern_;
        dealii::SparseMatrix<double> mass_matrix_;
        dealii::SparseMatrix<double> stiffness_matrix_;
        dealii::SparseMatrix<double> system_matrix_;
        
        dealii::Vector<double> solution_u_;
        dealii::Vector<double> solution_v_;
        dealii::Vector<double> old_solution_u_;
        dealii::Vector<double> old_solution_v_;
        dealii::Vector<double> system_rhs_;
        
        // Protected parameters (accessible by derived classes for testing)
        unsigned int mesh_refinements_ = 6;
        double final_time_ = 6.0;  // Longer simulation to see multiple reflections
        double time_step_ = 0.01;
        unsigned int output_interval_ = 10;
        double wave_speed_ = 1.0;  // Wave propagation speed (c in u_tt = c² ∇²u)
        unsigned int input_pulse_interval_ = 100; // Inject new pulse every 100 time steps (0 = disabled)
        
    private:
        // Time stepping state
        bool first_time_step_ = true;
    };

} // namespace WaveEquation