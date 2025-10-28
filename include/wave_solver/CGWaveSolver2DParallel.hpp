#pragma once

#include "ProblemBase.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>
#include <string>

namespace WaveEquation
{
    /**
     * @brief Parallel 2D Wave Solver using MPI
     * 
     * This solver implements parallel time-domain wave equation solving
     * using MPI parallelization with Trilinos linear algebra.
     * 
     * Wave equation: u_tt = c^2 * ∇²u
     * Time integration: Crank-Nicolson (theta = 0.5)
     * Spatial discretization: Linear simplex elements (P1)
     * Parallelization: Domain decomposition with MPI
     */
    class CGWaveSolver2DParallel
    {
    public:
        static constexpr unsigned int dim = 2;

        /**
         * @brief Constructor
         * @param use_mpi If true, use MPI parallelization; if false, run in serial mode
         */
        CGWaveSolver2DParallel(bool use_mpi = true);
        
        /**
         * @brief Set the problem configuration
         */
        void set_problem(std::unique_ptr<ProblemBase<dim>> problem);
        
        /**
         * @brief Configuration setters
         */
        void set_mesh_file(const std::string &filename) { mesh_filename_ = filename; }
        void set_final_time(double final_time) { final_time_ = final_time; }
        void set_time_step(double time_step) { time_step_ = time_step; }
        void set_output_interval(unsigned int interval) { output_interval_ = interval; }
        void set_wave_speed(double wave_speed) { wave_speed_ = wave_speed; }
        void set_theta(double theta) { theta_ = theta; }
        
        /**
         * @brief Getters
         */
        const dealii::DoFHandler<dim>& get_dof_handler() const { return dof_handler_; }
        const dealii::parallel::fullydistributed::Triangulation<dim>& get_triangulation() const { return mesh_; }
        const dealii::FiniteElement<dim>& get_fe() const { return *fe_; }
        unsigned int get_n_dofs() const;
        
        /**
         * @brief Main interface
         */
        void setup();
        void run();
        
        /**
         * @brief Analysis
         */
        double compute_energy() const;
        
        /**
         * @brief Output results
         */
        void output_results(unsigned int step, double time) const;

    private:
        /**
         * @brief Setup methods
         */
        void load_mesh();
        void setup_dofs();
        void setup_matrices();
        void setup_vectors();
        
        /**
         * @brief Assembly methods
         */
        void assemble_mass_matrix();
        void assemble_stiffness_matrix();
        
        /**
         * @brief Apply initial conditions
         */
        void apply_initial_conditions();
        
        /**
         * @brief Solve one time step
         */
        void solve_time_step(double time);

        /**
         * @brief MPI configuration
         */
        bool use_mpi_;
        unsigned int mpi_size_;
        unsigned int mpi_rank_;
        
        /**
         * @brief Problem definition
         */
        std::unique_ptr<ProblemBase<dim>> problem_;
        
        /**
         * @brief Finite element infrastructure
         */
        dealii::parallel::fullydistributed::Triangulation<dim> mesh_;
        std::unique_ptr<dealii::FiniteElement<dim>> fe_;
        dealii::DoFHandler<dim> dof_handler_;
        
        /**
         * @brief Linear algebra objects (Trilinos for parallel)
         */
        dealii::TrilinosWrappers::SparseMatrix mass_matrix_;
        dealii::TrilinosWrappers::SparseMatrix stiffness_matrix_;
        dealii::TrilinosWrappers::SparseMatrix system_matrix_;
        
        dealii::TrilinosWrappers::MPI::Vector solution_u_;
        dealii::TrilinosWrappers::MPI::Vector solution_v_;
        dealii::TrilinosWrappers::MPI::Vector old_solution_u_;
        dealii::TrilinosWrappers::MPI::Vector old_solution_v_;
        dealii::TrilinosWrappers::MPI::Vector system_rhs_;
        
        /**
         * @brief DoF indexing
         */
        dealii::IndexSet locally_owned_dofs_;
        dealii::IndexSet locally_relevant_dofs_;
        
        /**
         * @brief Parallel output stream (only rank 0 prints)
         */
        dealii::ConditionalOStream pcout_;
        
        /**
         * @brief Parameters
         */
        std::string mesh_filename_;
        double final_time_;
        double time_step_;
        unsigned int output_interval_;
        double wave_speed_;
        double theta_;  // Time integration parameter (0.5 = Crank-Nicolson)
        
        /**
         * @brief Time stepping state
         */
        bool first_time_step_;
    };

} // namespace WaveEquation
