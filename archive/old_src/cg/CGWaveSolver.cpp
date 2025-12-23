#include "CGWaveSolver.hpp"
#include "ProblemBase.hpp"
#include "VTKOutput.hpp"
#include "EnergyCalculator.hpp"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <limits>

namespace WaveEquation
{
    template <int dim>
    CGWaveSolver<dim>::CGWaveSolver()
        : fe_(1)  // Linear elements
        , dof_handler_(triangulation_)
    {
    }

    template <int dim>
    void CGWaveSolver<dim>::set_problem(std::unique_ptr<ProblemBase<dim>> problem)
    {
        problem_ = std::move(problem);
    }

    template <int dim>
    void CGWaveSolver<dim>::setup()
    {
        this->pcout << "Setting up 1D CG Wave Equation Solver" << std::endl;
        this->pcout << "=====================================" << std::endl;

        create_mesh();
        setup_dofs();
        setup_matrices();
        setup_vectors();
        
        this->pcout << "Setup completed successfully!" << std::endl;
    }

    template <int dim>
    void CGWaveSolver<dim>::create_mesh()
    {
        this->pcout << "Creating 1D mesh..." << std::endl;
        
        // Create 1D interval from -1 to 1
        dealii::GridGenerator::hyper_cube(triangulation_, -1.0, 1.0);
        triangulation_.refine_global(mesh_refinements_);
        
        this->pcout << "  Number of cells: " << triangulation_.n_active_cells() << std::endl;
    }

    template <int dim>
    void CGWaveSolver<dim>::setup_dofs()
    {
        this->pcout << "Setting up degrees of freedom..." << std::endl;
        
        dof_handler_.distribute_dofs(fe_);
        
        this->pcout << "  Number of dofs: " << dof_handler_.n_dofs() << std::endl;
    }

    template <int dim>
    void CGWaveSolver<dim>::setup_matrices()
    {
        this->pcout << "Setting up matrices..." << std::endl;
        
        // Create sparsity pattern
        dealii::DynamicSparsityPattern dsp(dof_handler_.n_dofs());
        dealii::DoFTools::make_sparsity_pattern(dof_handler_, dsp);
        
        // Convert to static sparsity pattern and store as member
        sparsity_pattern_.copy_from(dsp);
        
        // Initialize matrices with the member sparsity pattern
        mass_matrix_.reinit(sparsity_pattern_);
        stiffness_matrix_.reinit(sparsity_pattern_);
        system_matrix_.reinit(sparsity_pattern_);
        
        // Assemble matrices
        assemble_mass_matrix();
        assemble_stiffness_matrix();
        
        this->pcout << "  Matrices assembled successfully!" << std::endl;
    }

    template <int dim>
    void CGWaveSolver<dim>::assemble_mass_matrix()
    {
        this->pcout << "  Assembling mass matrix..." << std::endl;
        
        mass_matrix_ = 0.0;
        
        const dealii::QGauss<dim> quadrature(fe_.degree + 1);
        dealii::FEValues<dim> fe_values(fe_, quadrature,
                                       dealii::update_values | 
                                       dealii::update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_.dofs_per_cell;
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

    template <int dim>
    void CGWaveSolver<dim>::assemble_stiffness_matrix()
    {
        this->pcout << "  Assembling stiffness matrix..." << std::endl;
        
        stiffness_matrix_ = 0.0;
        
        const dealii::QGauss<dim> quadrature(fe_.degree + 1);
        dealii::FEValues<dim> fe_values(fe_, quadrature,
                                       dealii::update_gradients | 
                                       dealii::update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_.dofs_per_cell;
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

    template <int dim>
    void CGWaveSolver<dim>::setup_vectors()
    {
        this->pcout << "Setting up vectors..." << std::endl;
        
        solution_u_.reinit(dof_handler_.n_dofs());
        solution_v_.reinit(dof_handler_.n_dofs());
        old_solution_u_.reinit(dof_handler_.n_dofs());
        old_solution_v_.reinit(dof_handler_.n_dofs());
        system_rhs_.reinit(dof_handler_.n_dofs());
    }

    template <int dim>
    void CGWaveSolver<dim>::inject_pulse()
    {
        // Add a new Gaussian pulse at the center
        // This is done by adding the initial displacement and velocity to the current solution
        
        if (!problem_)
        {
            return;
        }
        
        // Create wrapper function for pulse displacement
        class PulseDisplacementFunction : public dealii::Function<dim>
        {
        public:
            PulseDisplacementFunction(const ProblemBase<dim> *prob) : dealii::Function<dim>(), problem(prob) {}
            
            virtual double value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
            {
                return problem->initial_displacement(p);
            }
            
        private:
            const ProblemBase<dim> *problem;
        };
        
        // Create temporary vector for the new pulse
        dealii::Vector<double> pulse_displacement(dof_handler_.n_dofs());
        PulseDisplacementFunction pulse_func(problem_.get());
        dealii::VectorTools::interpolate(dof_handler_, 
                                        pulse_func,
                                        pulse_displacement);
        
        // Add the new pulse to the existing solution (superposition)
        solution_u_.add(1.0, pulse_displacement);
        old_solution_u_.add(1.0, pulse_displacement);
        
        // Note: We keep velocity at zero for the new pulse (pure displacement injection)
        // This will cause the new pulse to split symmetrically just like the initial condition
    }

    template <int dim>
    void CGWaveSolver<dim>::run()
    {
        this->pcout << "Running 1D Wave Equation Solver" << std::endl;
        this->pcout << "===============================" << std::endl;
        
        // Calculate mesh spacing
        double h_min = std::numeric_limits<double>::max();
        for (const auto &cell : triangulation_.active_cell_iterators())
        {
            h_min = std::min(h_min, cell->diameter());
        }
        
        // Calculate CFL number
        // For wave equation: CFL = c * dt / h
        const double cfl_number = wave_speed_ * time_step_ / h_min;
        
        this->pcout << "Mesh spacing (h): " << h_min << std::endl;
        this->pcout << "Time step (dt):   " << time_step_ << std::endl;
        this->pcout << "Wave speed (c):   " << wave_speed_ << std::endl;
        this->pcout << "CFL number:       " << cfl_number << std::endl;
        this->pcout << "  Note: For explicit schemes, CFL < 1 required for stability." << std::endl;
        this->pcout << "        For Crank-Nicolson (theta=0.5), unconditionally stable." << std::endl;
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
        
        // Debug: Print initial velocity values
        this->pcout << "Step 0 - Initial velocity at key points:" << std::endl;
        for (const auto &cell : dof_handler_.active_cell_iterators())
        {
            for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
            {
                const unsigned int dof_idx = cell->vertex_dof_index(v, 0);
                const double x = cell->vertex(v)[0];
                if (std::abs(x + 0.5) < 0.01 || std::abs(x) < 0.01 || std::abs(x - 0.5) < 0.01)
                {
                    this->pcout << "  x=" << x << ", v=" << solution_v_[dof_idx] << std::endl;
                }
            }
        }
        
        // Compute the number of time steps to take
        const unsigned int n_time_steps = static_cast<unsigned int>(std::round(final_time_ / time_step_));
        
        unsigned int step = 0;
        double time = 0.0;
        
        output_results(step, time);
        
        // Time stepping loop - use step counter to avoid floating-point accumulation errors
        for (step = 1; step <= n_time_steps; ++step)
        {
            time = step * time_step_;
            
            if (step % 10 == 0)
            {
                this->pcout << "Step " << step << ", Time = " << time << std::endl;
            }
            
            solve_time_step(time);
            // Inject new pulse every input_pulse_interval_ steps
            if (step % input_pulse_interval_ == 0 && step > 0)
            {
                this->pcout << "*** Injecting new pulse at step " << step << " (t=" << time << ") ***" << std::endl;
                inject_pulse();
            }
            
            // Output at every step to see wave evolution
            output_results(step, time);
            
            // Debug: Print velocity at step 1
            if (step == 1)
            {
                this->pcout << "Step 1 - Velocity after first time step:" << std::endl;
                for (const auto &cell : dof_handler_.active_cell_iterators())
                {
                    for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
                    {
                        const unsigned int dof_idx = cell->vertex_dof_index(v, 0);
                        const double x = cell->vertex(v)[0];
                        if (std::abs(x + 0.5) < 0.01 || std::abs(x) < 0.01 || std::abs(x - 0.5) < 0.01)
                        {
                            this->pcout << "  x=" << x << ", v=" << solution_v_[dof_idx] << std::endl;
                        }
                    }
                }
            }
            
            // Update old solutions
            old_solution_u_ = solution_u_;
            old_solution_v_ = solution_v_;
        }
        
        // Final output
        output_results(step, time);
        
        this->pcout << "Simulation completed in " << step << " steps" << std::endl;
        this->pcout << "Final energy: " << compute_energy() << std::endl;
    }

    template <int dim>
    void CGWaveSolver<dim>::apply_initial_conditions()
    {
        this->pcout << "Applying initial conditions..." << std::endl;
        
        if (!problem_)
        {
            throw std::runtime_error("No problem defined for CG solver");
        }
        
        // Create wrapper function for initial displacement
        class InitialDisplacementFunction : public dealii::Function<dim>
        {
        public:
            InitialDisplacementFunction(const ProblemBase<dim> *prob) : dealii::Function<dim>(), problem(prob) {}
            
            virtual double value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
            {
                return problem->initial_displacement(p);
            }
            
        private:
            const ProblemBase<dim> *problem;
        };
        
        // Create wrapper function for initial velocity
        class InitialVelocityFunction : public dealii::Function<dim>
        {
        public:
            InitialVelocityFunction(const ProblemBase<dim> *prob) : dealii::Function<dim>(), problem(prob) {}
            
            virtual double value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
            {
                return problem->initial_velocity(p);
            }
            
        private:
            const ProblemBase<dim> *problem;
        };
        
        // Project initial displacement
        InitialDisplacementFunction disp_func(problem_.get());
        dealii::VectorTools::interpolate(dof_handler_, 
                                        disp_func,
                                        solution_u_);
        old_solution_u_ = solution_u_;
        
        // Project initial velocity
        InitialVelocityFunction vel_func(problem_.get());
        dealii::VectorTools::interpolate(dof_handler_,
                                        vel_func,
                                        solution_v_);
        old_solution_v_ = solution_v_;
    }

    template <int dim>
    void CGWaveSolver<dim>::solve_time_step(double /*time*/)
    {
        // Following deal.II step-23 theta scheme
        // Wave equation: u_tt = c² ∇²u
        // Rewritten as: u_t = v, v_t = c² ∇²u
        // 
        // Discretization (theta=0.5 for Crank-Nicolson):
        // M(U^n - U^{n-1})/dt = dt*M*V^{n-1} - dt²*θ(1-θ)*A*U^{n-1}
        // M(V^n - V^{n-1})/dt = -dt*[θ*A*U^n + (1-θ)*A*U^{n-1}]
        
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
        // This allows perfect wave reflection with no energy loss
        
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
        // This allows perfect wave reflection with no energy loss
        
        // Solve for V^n
        dealii::SolverControl solver_control_v(10000, 1e-8 * system_rhs.l2_norm());
        dealii::SolverCG<dealii::Vector<double>> cg_v(solver_control_v);
        cg_v.solve(matrix_v, solution_v_, system_rhs, dealii::PreconditionIdentity());
        
    }

    template <int dim>
    void CGWaveSolver<dim>::output_results(unsigned int step, double time) const
    {
        // Use VTK output utility with organized directories
        Utilities::VTKOutput<dim>::write_vtk(
            dof_handler_,
            solution_u_,
            solution_v_,
            step,
            "solution_1d",
            "1d/vtk");
        
        // Also keep the simple text output for 1D testing
        if (dim == 1)
        {
            Utilities::VTKOutput<dim>::write_text_1d(
                solution_u_,
                step,
                time,
                "output_1d",
                "1d/txt");
        }
        
        if (step % 50 == 0)
        {
            this->pcout << "  Output written for step " << step << " (t=" << time << ")" << std::endl;
        }
    }

    template <int dim>
    double CGWaveSolver<dim>::compute_energy() const
    {
        // Use energy calculator utility
        return Utilities::EnergyCalculator::compute_total_energy(
            solution_v_,
            solution_u_,
            mass_matrix_,
            stiffness_matrix_);
    }

    // Explicit instantiation for 1D and 2D
    template class CGWaveSolver<1>;
    template class CGWaveSolver<2>;

} // namespace WaveEquation