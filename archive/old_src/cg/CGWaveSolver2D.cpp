#include "CGWaveSolver2D.hpp"
#include "ProblemBase.hpp"
#include "VTKOutput.hpp"
#include "EnergyCalculator.hpp"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <fstream>
#include <limits>

namespace WaveEquation
{
    CGWaveSolver2D::CGWaveSolver2D()
        : fe_(1)  // Linear simplex elements
        , dof_handler_(triangulation_)
    {
    }

    void CGWaveSolver2D::set_problem(std::unique_ptr<ProblemBase<2>> problem)
    {
        problem_ = std::move(problem);
    }

    void CGWaveSolver2D::setup()
    {
        this->pcout << "Setting up 2D CG Wave Equation Solver (Triangular Elements)" << std::endl;
        this->pcout << "=============================================================" << std::endl;

        load_mesh();
        setup_dofs();
        setup_matrices();
        setup_vectors();
        
        this->pcout << "Setup completed successfully!" << std::endl;
    }

    void CGWaveSolver2D::load_mesh()
    {
        this->pcout << "Loading 2D triangular mesh..." << std::endl;
        
        if (mesh_filename_.empty())
        {
            throw std::runtime_error("No mesh file specified! Use set_mesh_file() first.");
        }
        
        // Read mesh from gmsh file
        dealii::GridIn<2> grid_in;
        grid_in.attach_triangulation(triangulation_);
        
        std::ifstream input_file(mesh_filename_);
        if (!input_file)
        {
            throw std::runtime_error("Could not open mesh file: " + mesh_filename_);
        }
        
        grid_in.read_msh(input_file);
        
        this->pcout << "  Mesh file: " << mesh_filename_ << std::endl;
        this->pcout << "  Number of cells: " << triangulation_.n_active_cells() << std::endl;
        this->pcout << "  Number of vertices: " << triangulation_.n_vertices() << std::endl;
    }

    void CGWaveSolver2D::setup_dofs()
    {
        this->pcout << "Setting up degrees of freedom..." << std::endl;
        
        dof_handler_.distribute_dofs(fe_);
        
        this->pcout << "  Number of dofs: " << dof_handler_.n_dofs() << std::endl;
    }

    void CGWaveSolver2D::setup_matrices()
    {
        this->pcout << "Setting up matrices..." << std::endl;
        
        // Create sparsity pattern
        dealii::DynamicSparsityPattern dsp(dof_handler_.n_dofs());
        dealii::DoFTools::make_sparsity_pattern(dof_handler_, dsp);
        
        // Convert to static sparsity pattern
        sparsity_pattern_.copy_from(dsp);
        
        // Initialize matrices
        mass_matrix_.reinit(sparsity_pattern_);
        stiffness_matrix_.reinit(sparsity_pattern_);
        system_matrix_.reinit(sparsity_pattern_);
        
        // Assemble matrices
        assemble_mass_matrix();
        assemble_stiffness_matrix();
        
        this->pcout << "  Matrices assembled successfully!" << std::endl;
    }

    void CGWaveSolver2D::assemble_mass_matrix()
    {
        this->pcout << "  Assembling mass matrix..." << std::endl;
        
        mass_matrix_ = 0.0;
        
        // Use simplex quadrature for triangular elements
        const dealii::QGaussSimplex<2> quadrature(fe_.degree + 1);
        dealii::FEValues<2> fe_values(fe_, quadrature,
                                       dealii::update_values | 
                                       dealii::update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature.size();
        
        dealii::FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        for (const auto &cell : dof_handler_.active_cell_iterators())
        {
            fe_values.reinit(cell);
            cell_mass_matrix = 0.0;
            
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                                 fe_values.shape_value(j, q) *
                                                 fe_values.JxW(q);
                    }
                }
            }
            
            cell->get_dof_indices(local_dof_indices);
            mass_matrix_.add(local_dof_indices, cell_mass_matrix);
        }
    }

    void CGWaveSolver2D::assemble_stiffness_matrix()
    {
        this->pcout << "  Assembling stiffness matrix..." << std::endl;
        
        stiffness_matrix_ = 0.0;
        
        // Use simplex quadrature for triangular elements
        const dealii::QGaussSimplex<2> quadrature(fe_.degree + 1);
        dealii::FEValues<2> fe_values(fe_, quadrature,
                                       dealii::update_gradients | 
                                       dealii::update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature.size();
        
        dealii::FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        for (const auto &cell : dof_handler_.active_cell_iterators())
        {
            fe_values.reinit(cell);
            cell_stiffness_matrix = 0.0;
            
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        // Include c^2 factor for wave equation: u_tt = c^2 ∇^2 u
                        cell_stiffness_matrix(i, j) += wave_speed_ * wave_speed_ *
                                                      fe_values.shape_grad(i, q) *
                                                      fe_values.shape_grad(j, q) *
                                                      fe_values.JxW(q);
                    }
                }
            }
            
            cell->get_dof_indices(local_dof_indices);
            stiffness_matrix_.add(local_dof_indices, cell_stiffness_matrix);
        }
    }

    void CGWaveSolver2D::setup_vectors()
    {
        this->pcout << "Setting up vectors..." << std::endl;
        
        solution_u_.reinit(dof_handler_.n_dofs());
        solution_v_.reinit(dof_handler_.n_dofs());
        old_solution_u_.reinit(dof_handler_.n_dofs());
        old_solution_v_.reinit(dof_handler_.n_dofs());
        system_rhs_.reinit(dof_handler_.n_dofs());
    }

    void CGWaveSolver2D::inject_pulse()
    {
        // Add a new Gaussian pulse at the center
        // This is done by adding the initial displacement and velocity to the current solution
        
        if (!problem_)
        {
            return;
        }
        
        // Create wrapper function for pulse displacement
        class PulseDisplacementFunction : public dealii::Function<2>
        {
        public:
            PulseDisplacementFunction(const ProblemBase<2> *prob) : dealii::Function<2>(), problem(prob) {}
            
            virtual double value(const dealii::Point<2> &p, const unsigned int /*component*/) const override
            {
                return problem->initial_displacement(p);
            }
            
        private:
            const ProblemBase<2> *problem;
        };
        
        // Create temporary vector for the new pulse
        dealii::Vector<double> pulse_displacement(dof_handler_.n_dofs());
        PulseDisplacementFunction pulse_func(problem_.get());
        
        // Use MappingFE for simplex elements
        dealii::MappingFE<2> mapping(fe_);
        dealii::VectorTools::interpolate(mapping,
                                        dof_handler_, 
                                        pulse_func,
                                        pulse_displacement);
        
        // Add the new pulse to the existing solution (superposition)
        solution_u_.add(1.0, pulse_displacement);
        old_solution_u_.add(1.0, pulse_displacement);
        
        // Note: We keep velocity at zero for the new pulse (pure displacement injection)
        // This will cause the new pulse to propagate radially outward
    }

    void CGWaveSolver2D::run()
    {
        this->pcout << "Running 2D Wave Equation Solver" << std::endl;
        this->pcout << "================================" << std::endl;
        
        // Calculate mesh spacing
        double h_min = std::numeric_limits<double>::max();
        for (const auto &cell : triangulation_.active_cell_iterators())
        {
            h_min = std::min(h_min, cell->diameter());
        }
        
        // Calculate CFL number
        const double cfl_number = wave_speed_ * time_step_ / h_min;
        
        this->pcout << "Mesh spacing (h): " << h_min << std::endl;
        this->pcout << "Time step (dt):   " << time_step_ << std::endl;
        this->pcout << "Wave speed (c):   " << wave_speed_ << std::endl;
        this->pcout << "CFL number:       " << cfl_number << std::endl;
        this->pcout << "  Note: For Crank-Nicolson (theta=0.5), unconditionally stable." << std::endl;
        this->pcout << "        Recommended CFL < 1 for accuracy." << std::endl;
        
        if (cfl_number > 1.0)
        {
            this->pcout << "  WARNING: CFL > 1.0 may reduce accuracy!" << std::endl;
        }
        else
        {
            this->pcout << "  CFL condition satisfied for accuracy." << std::endl;
        }
        this->pcout << std::endl;
        
        this->pcout << "Final time: " << final_time_ << std::endl;
        this->pcout << "Time step: " << time_step_ << std::endl;
        
        apply_initial_conditions();
        
        // Compute the number of time steps
        const unsigned int n_time_steps = static_cast<unsigned int>(std::round(final_time_ / time_step_));
        
        unsigned int step = 0;
        double time = 0.0;
        
        if (output_interval_ > 0)
        {
            output_results(step, time);
        }
        
        // Time stepping loop
        for (step = 1; step <= n_time_steps; ++step)
        {
            time = step * time_step_;
            
            if (step % 10 == 0)
            {
                this->pcout << "Step " << step << ", Time = " << time << std::endl;
            }
            
            solve_time_step(time);
            
            // Inject new pulse every input_pulse_interval_ steps
            if (input_pulse_interval_ > 0 && step % input_pulse_interval_ == 0 && step > 0)
            {
                this->pcout << "*** Injecting new pulse at step " << step << " (t=" << time << ") ***" << std::endl;
                inject_pulse();
            }
            
            if (output_interval_ > 0 && step % output_interval_ == 0)
            {
                output_results(step, time);
            }
            
            // Update old solutions
            old_solution_u_ = solution_u_;
            old_solution_v_ = solution_v_;
        }
        
        // Final output
        if (output_interval_ > 0)
        {
            output_results(step, time);
        }
        
        this->pcout << "Simulation completed in " << step << " steps" << std::endl;
        this->pcout << "Final energy: " << compute_energy() << std::endl;
    }

    void CGWaveSolver2D::apply_initial_conditions()
    {
        this->pcout << "Applying initial conditions..." << std::endl;
        
        if (!problem_)
        {
            throw std::runtime_error("No problem defined for 2D CG solver");
        }
        
        // Create wrapper function for initial displacement
        class InitialDisplacementFunction : public dealii::Function<2>
        {
        public:
            InitialDisplacementFunction(const ProblemBase<2> *prob) : dealii::Function<2>(), problem(prob) {}
            
            virtual double value(const dealii::Point<2> &p, const unsigned int /*component*/) const override
            {
                return problem->initial_displacement(p);
            }
            
        private:
            const ProblemBase<2> *problem;
        };
        
        // Create wrapper function for initial velocity
        class InitialVelocityFunction : public dealii::Function<2>
        {
        public:
            InitialVelocityFunction(const ProblemBase<2> *prob) : dealii::Function<2>(), problem(prob) {}
            
            virtual double value(const dealii::Point<2> &p, const unsigned int /*component*/) const override
            {
                return problem->initial_velocity(p);
            }
            
        private:
            const ProblemBase<2> *problem;
        };
        
        // Project initial displacement (need MappingFE for simplex elements)
        dealii::MappingFE<2> mapping(fe_);
        InitialDisplacementFunction disp_func(problem_.get());
        dealii::VectorTools::interpolate(mapping,
                                        dof_handler_, 
                                        disp_func,
                                        solution_u_);
        old_solution_u_ = solution_u_;
        
        // Project initial velocity
        InitialVelocityFunction vel_func(problem_.get());
        dealii::VectorTools::interpolate(mapping,
                                        dof_handler_,
                                        vel_func,
                                        solution_v_);
        old_solution_v_ = solution_v_;
    }

    void CGWaveSolver2D::solve_time_step(double /*time*/)
    {
        // Same theta scheme as 1D solver
        const double dt = time_step_;
        const double theta = 0.5;  // Crank-Nicolson
        
        dealii::Vector<double> tmp(dof_handler_.n_dofs());
        dealii::Vector<double> system_rhs(dof_handler_.n_dofs());
        
        // Step 1: Solve for U^n
        // System: (M + dt²*θ²*A)*U^n = M*U^{n-1} + dt*M*V^{n-1} - dt²*θ(1-θ)*A*U^{n-1}
        
        // RHS = M*U^{n-1}
        mass_matrix_.vmult(system_rhs, old_solution_u_);
        
        // RHS += dt*M*V^{n-1}
        mass_matrix_.vmult(tmp, old_solution_v_);
        system_rhs.add(dt, tmp);
        
        // RHS -= dt²*θ(1-θ)*A*U^{n-1}
        stiffness_matrix_.vmult(tmp, old_solution_u_);
        system_rhs.add(-theta * (1.0 - theta) * dt * dt, tmp);
        
        // LHS matrix: M + dt²*θ²*A
        dealii::SparseMatrix<double> matrix_u;
        matrix_u.reinit(sparsity_pattern_);
        matrix_u.copy_from(mass_matrix_);
        matrix_u.add(theta * theta * dt * dt, stiffness_matrix_);
        
        // No boundary conditions applied - natural Neumann BC (du/dn = 0) for reflecting boundaries
        // This allows perfect wave reflection at the domain boundaries with no energy loss
        
        // Solve for U^n
        dealii::SolverControl solver_control_u(10000, 1e-8 * system_rhs.l2_norm());
        dealii::SolverCG<dealii::Vector<double>> cg_u(solver_control_u);
        cg_u.solve(matrix_u, solution_u_, system_rhs, dealii::PreconditionIdentity());
        
        // Step 2: Solve for V^n
        // System: M*V^n = M*V^{n-1} - dt*[θ*A*U^n + (1-θ)*A*U^{n-1}]
        
        // RHS = M*V^{n-1}
        mass_matrix_.vmult(system_rhs, old_solution_v_);
        
        // RHS -= dt*θ*A*U^n
        stiffness_matrix_.vmult(tmp, solution_u_);
        system_rhs.add(-dt * theta, tmp);
        
        // RHS -= dt*(1-θ)*A*U^{n-1}
        stiffness_matrix_.vmult(tmp, old_solution_u_);
        system_rhs.add(-dt * (1.0 - theta), tmp);
        
        // LHS matrix: M
        dealii::SparseMatrix<double> matrix_v;
        matrix_v.reinit(sparsity_pattern_);
        matrix_v.copy_from(mass_matrix_);
        
        // No boundary conditions applied - natural Neumann BC (dv/dn = 0) for reflecting boundaries
        // This allows perfect wave reflection at the domain boundaries with no energy loss
        
        // Solve for V^n
        dealii::SolverControl solver_control_v(10000, 1e-8 * system_rhs.l2_norm());
        dealii::SolverCG<dealii::Vector<double>> cg_v(solver_control_v);
        cg_v.solve(matrix_v, solution_v_, system_rhs, dealii::PreconditionIdentity());
    }

    void CGWaveSolver2D::output_results(unsigned int step, double time) const
    {
        // Use VTK output utility for ParaView with organized directories
        Utilities::VTKOutput<2>::write_vtk(
            dof_handler_,
            solution_u_,
            solution_v_,
            step,
            "solution_2d",
            "2d/vtk");
        
        // Also write text files for Python visualization
        Utilities::VTKOutput<2>::write_text_2d(
            dof_handler_,
            solution_u_,
            solution_v_,
            step,
            time,
            "output_2d",
            "2d/txt");
        
        if (output_interval_ > 0 && (step % output_interval_ == 0 || step == 0))
        {
            this->pcout << "  Output written for step " << step << " (t=" << time << ")" << std::endl;
        }
    }

    double CGWaveSolver2D::compute_energy() const
    {
        return Utilities::EnergyCalculator::compute_total_energy(
            solution_v_,
            solution_u_,
            mass_matrix_,
            stiffness_matrix_);
    }

} // namespace WaveEquation
