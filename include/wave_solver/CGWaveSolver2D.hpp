#pragma once

#include "WaveSolverBase.hpp"
#include "ProblemBase.hpp"

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <string>

namespace WaveEquation
{
    /**
     * @brief 2D Wave Solver using Continuous Galerkin with Simplex Elements
     * 
     * This solver is specialized for 2D triangular meshes and uses FE_SimplexP
     * elements instead of the FE_Q quad elements used in CGWaveSolver<1>.
     */
    class CGWaveSolver2D : public WaveSolverBase<2>
    {
    public:
        CGWaveSolver2D();
        
        void set_problem(std::unique_ptr<ProblemBase<2>> problem);
        
        // Configuration setters
        void set_mesh_file(const std::string &filename) { mesh_filename_ = filename; }
        void set_final_time(double final_time) { final_time_ = final_time; }
        void set_time_step(double time_step) { time_step_ = time_step; }
        void set_output_interval(unsigned int interval) { output_interval_ = interval; }
        void set_wave_speed(double wave_speed) { wave_speed_ = wave_speed; }
        void set_pulse_injection_interval(unsigned int interval) { input_pulse_interval_ = interval; }
        
        // Getters
        const dealii::DoFHandler<2>& get_dof_handler() const { return dof_handler_; }
        const dealii::Triangulation<2>& get_triangulation() const { return triangulation_; }
        const dealii::FiniteElement<2>& get_fe() const { return fe_; }
        const dealii::Vector<double>& get_solution_u() const { return solution_u_; }
        const dealii::Vector<double>& get_solution_v() const { return solution_v_; }
        const ProblemBase<2>* get_problem() const { return problem_.get(); }
        unsigned int get_n_dofs() const { return dof_handler_.n_dofs(); }
        
        // Main interface implementation
        void setup() override;
        void run() override;
        
        // Analysis
        double compute_energy() const override;
        
        // Output
        void output_results(unsigned int step, double time) const override;

    protected:
        // Protected methods
        void apply_initial_conditions();
        void solve_time_step(double time);

    private:
        // Setup methods
        void load_mesh();
        void setup_dofs();
        void setup_matrices();
        void setup_vectors();
        
        // Assembly methods
        void assemble_mass_matrix();
        void assemble_stiffness_matrix();
        
        // Solution methods
        void inject_pulse();  // Inject a new pulse at current time
        
    protected:
        // Protected member variables
        std::unique_ptr<ProblemBase<2>> problem_;
        
        // Finite element infrastructure (Simplex elements for triangles!)
        dealii::Triangulation<2> triangulation_;
        dealii::FE_SimplexP<2> fe_;
        dealii::DoFHandler<2> dof_handler_;
        
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
        
        // Protected parameters
        std::string mesh_filename_ = "";
        double final_time_ = 2.0;
        double time_step_ = 0.01;
        unsigned int output_interval_ = 10;
        double wave_speed_ = 1.0;
        unsigned int input_pulse_interval_ = 100; // Inject new pulse every 100 time steps (0 = disabled)
        
    private:
        // Time stepping state
        bool first_time_step_ = true;
    };

} // namespace WaveEquation
