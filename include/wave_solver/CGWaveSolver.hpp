#pragma once

#include "WaveSolverBase.hpp"
#include "ProblemBase.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/cgal/triangulation.h>
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
        
        // Main interface implementation
        void setup() override;
        void run() override;
        
        // Analysis
        double compute_energy() const override;
        
        // Output
        void output_results(unsigned int step, double time) const override;

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
        void apply_initial_conditions();
        void inject_pulse();  // Inject a new pulse at current time
        void solve_time_step(double time);
        
        // Member variables
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
        
        // Parameters
        unsigned int mesh_refinements_ = 6;
        double final_time_ = 6.0;  // Longer simulation to see multiple reflections
        double time_step_ = 0.01;
        unsigned int output_interval_ = 10;
        double wave_speed_ = 1.0;  // Wave propagation speed (c in u_tt = c² ∇²u)
        unsigned int input_pulse_interval_ = 100; // Inject new pulse every 50 time steps
        
        // Time stepping state
        bool first_time_step_ = true;
    };

} // namespace WaveEquation