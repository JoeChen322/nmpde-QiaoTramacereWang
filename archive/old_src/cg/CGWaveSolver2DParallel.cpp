#include "CGWaveSolver2DParallel.hpp"

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <fstream>
#include <limits>

namespace WaveEquation
{
    CGWaveSolver2DParallel::CGWaveSolver2DParallel(bool use_mpi)
        : use_mpi_(use_mpi)
        , mpi_size_(use_mpi ? dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) : 1)
        , mpi_rank_(use_mpi ? dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) : 0)
        , mesh_(MPI_COMM_WORLD)
        , dof_handler_(mesh_)
        , pcout_(std::cout, mpi_rank_ == 0)
        , mesh_filename_("")
        , final_time_(2.0)
        , time_step_(0.01)
        , output_interval_(10)
        , wave_speed_(1.0)
        , theta_(0.5)  // Crank-Nicolson
        , first_time_step_(true)
    {
        if (use_mpi_)
        {
            pcout_ << "Running in PARALLEL mode with " << mpi_size_ << " MPI processes" << std::endl;
        }
        else
        {
            pcout_ << "Running in SERIAL mode (MPI disabled)" << std::endl;
        }
    }

    void CGWaveSolver2DParallel::set_problem(std::unique_ptr<ProblemBase<dim>> problem)
    {
        problem_ = std::move(problem);
    }

    unsigned int CGWaveSolver2DParallel::get_n_dofs() const
    {
        return dof_handler_.n_dofs();
    }

    void CGWaveSolver2DParallel::setup()
    {
        pcout_ << "===============================================" << std::endl;
        pcout_ << "Setting up 2D Parallel Wave Equation Solver" << std::endl;
        pcout_ << "===============================================" << std::endl;

        load_mesh();
        setup_dofs();
        setup_matrices();
        setup_vectors();
        
        pcout_ << "Setup completed successfully!" << std::endl;
    }

    void CGWaveSolver2DParallel::load_mesh()
    {
        pcout_ << "Initializing the mesh" << std::endl;
        
        if (mesh_filename_.empty())
        {
            throw std::runtime_error("No mesh file specified! Use set_mesh_file() first.");
        }
        
        // First we read the mesh from file into a serial triangulation
        dealii::Triangulation<dim> mesh_serial;
        
        {
            dealii::GridIn<dim> grid_in;
            grid_in.attach_triangulation(mesh_serial);
            
            std::ifstream grid_in_file(mesh_filename_);
            if (!grid_in_file)
            {
                throw std::runtime_error("Could not open mesh file: " + mesh_filename_);
            }
            grid_in.read_msh(grid_in_file);
        }
        
        // Then, we copy the triangulation into the parallel one
        {
            if (use_mpi_)
            {
                dealii::GridTools::partition_triangulation(mpi_size_, mesh_serial);
            }
            
            const auto construction_data = 
                dealii::TriangulationDescription::Utilities::
                    create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
            mesh_.create_triangulation(construction_data);
        }
        
        pcout_ << "  Mesh file: " << mesh_filename_ << std::endl;
        pcout_ << "  Number of elements = " << mesh_.n_global_active_cells() << std::endl;
        
        pcout_ << "-----------------------------------------------" << std::endl;
    }

    void CGWaveSolver2DParallel::setup_dofs()
    {
        pcout_ << "Initializing the finite element space" << std::endl;
        
        // Initialize the finite element (linear simplex for triangles)
        fe_ = std::make_unique<dealii::FE_SimplexP<dim>>(1);
        
        pcout_ << "  Degree                     = " << fe_->degree << std::endl;
        pcout_ << "  DoFs per cell              = " << fe_->dofs_per_cell << std::endl;
        
        pcout_ << "-----------------------------------------------" << std::endl;
        
        pcout_ << "Initializing the DoF handler" << std::endl;
        
        dof_handler_.reinit(mesh_);
        dof_handler_.distribute_dofs(*fe_);
        
        // We retrieve the set of locally owned DoFs
        locally_owned_dofs_ = dof_handler_.locally_owned_dofs();
        
        // Also extract locally relevant DoFs (owned + ghosts)
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_, locally_relevant_dofs_);
        
        pcout_ << "  Number of DoFs = " << dof_handler_.n_dofs() << std::endl;
        
        pcout_ << "-----------------------------------------------" << std::endl;
    }

    void CGWaveSolver2DParallel::setup_matrices()
    {
        pcout_ << "Initializing the linear system" << std::endl;
        
        pcout_ << "  Initializing the sparsity pattern" << std::endl;
        
        // To initialize the sparsity pattern, we use Trilinos' class
        dealii::TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs_, MPI_COMM_WORLD);
        dealii::DoFTools::make_sparsity_pattern(dof_handler_, sparsity);
        
        // Compress so all processes retrieve info they need
        sparsity.compress();
        
        // Initialize matrices
        pcout_ << "  Initializing the mass matrix" << std::endl;
        mass_matrix_.reinit(sparsity);
        
        pcout_ << "  Initializing the stiffness matrix" << std::endl;
        stiffness_matrix_.reinit(sparsity);
        
        pcout_ << "  Initializing the system matrix" << std::endl;
        system_matrix_.reinit(sparsity);
        
        // Assemble matrices
        assemble_mass_matrix();
        assemble_stiffness_matrix();
        
        pcout_ << "  Matrices assembled successfully!" << std::endl;
        
        pcout_ << "-----------------------------------------------" << std::endl;
    }

    void CGWaveSolver2DParallel::assemble_mass_matrix()
    {
        pcout_ << "  Assembling mass matrix..." << std::endl;
        
        mass_matrix_ = 0.0;
        
        const dealii::QGaussSimplex<dim> quadrature(fe_->degree + 1);
        const unsigned int n_q = quadrature.size();
        const unsigned int dofs_per_cell = fe_->dofs_per_cell;
        
        dealii::FEValues<dim> fe_values(*fe_,
                                        quadrature,
                                        dealii::update_values | 
                                        dealii::update_JxW_values);
        
        dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
        
        for (const auto &cell : dof_handler_.active_cell_iterators())
        {
            // Only assemble on cells owned by this process
            if (!cell->is_locally_owned())
                continue;
            
            fe_values.reinit(cell);
            cell_matrix = 0.0;
            
            for (unsigned int q = 0; q < n_q; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) *
                                            fe_values.JxW(q);
                    }
                }
            }
            
            cell->get_dof_indices(dof_indices);
            mass_matrix_.add(dof_indices, cell_matrix);
        }
        
        // Exchange information between processes
        mass_matrix_.compress(dealii::VectorOperation::add);
    }

    void CGWaveSolver2DParallel::assemble_stiffness_matrix()
    {
        pcout_ << "  Assembling stiffness matrix..." << std::endl;
        
        stiffness_matrix_ = 0.0;
        
        const dealii::QGaussSimplex<dim> quadrature(fe_->degree + 1);
        const unsigned int n_q = quadrature.size();
        const unsigned int dofs_per_cell = fe_->dofs_per_cell;
        
        dealii::FEValues<dim> fe_values(*fe_,
                                        quadrature,
                                        dealii::update_gradients | 
                                        dealii::update_JxW_values);
        
        dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
        
        for (const auto &cell : dof_handler_.active_cell_iterators())
        {
            // Only assemble on cells owned by this process
            if (!cell->is_locally_owned())
                continue;
            
            fe_values.reinit(cell);
            cell_matrix = 0.0;
            
            for (unsigned int q = 0; q < n_q; ++q)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        // Include c^2 factor: u_tt = c^2 * ∇²u
                        cell_matrix(i, j) += wave_speed_ * wave_speed_ *
                                            fe_values.shape_grad(i, q) *
                                            fe_values.shape_grad(j, q) *
                                            fe_values.JxW(q);
                    }
                }
            }
            
            cell->get_dof_indices(dof_indices);
            stiffness_matrix_.add(dof_indices, cell_matrix);
        }
        
        // Exchange information between processes
        stiffness_matrix_.compress(dealii::VectorOperation::add);
    }

    void CGWaveSolver2DParallel::setup_vectors()
    {
        pcout_ << "Initializing solution vectors" << std::endl;
        
        solution_u_.reinit(locally_owned_dofs_, MPI_COMM_WORLD);
        solution_v_.reinit(locally_owned_dofs_, MPI_COMM_WORLD);
        old_solution_u_.reinit(locally_owned_dofs_, MPI_COMM_WORLD);
        old_solution_v_.reinit(locally_owned_dofs_, MPI_COMM_WORLD);
        system_rhs_.reinit(locally_owned_dofs_, MPI_COMM_WORLD);
        
        pcout_ << "-----------------------------------------------" << std::endl;
    }

    void CGWaveSolver2DParallel::apply_initial_conditions()
    {
        pcout_ << "Applying initial conditions..." << std::endl;
        
        if (!problem_)
        {
            throw std::runtime_error("No problem defined for 2D parallel solver");
        }
        
        // Create wrapper functions for initial conditions
        class InitialDisplacementFunction : public dealii::Function<dim>
        {
        public:
            InitialDisplacementFunction(const ProblemBase<dim> *prob) 
                : dealii::Function<dim>(), problem(prob) {}
            
            virtual double value(const dealii::Point<dim> &p, 
                               const unsigned int /*component*/) const override
            {
                return problem->initial_displacement(p);
            }
            
        private:
            const ProblemBase<dim> *problem;
        };
        
        class InitialVelocityFunction : public dealii::Function<dim>
        {
        public:
            InitialVelocityFunction(const ProblemBase<dim> *prob) 
                : dealii::Function<dim>(), problem(prob) {}
            
            virtual double value(const dealii::Point<dim> &p, 
                               const unsigned int /*component*/) const override
            {
                return problem->initial_velocity(p);
            }
            
        private:
            const ProblemBase<dim> *problem;
        };
        
        // Create temporary vectors to hold the interpolated values
        dealii::TrilinosWrappers::MPI::Vector temp_u(locally_owned_dofs_, MPI_COMM_WORLD);
        dealii::TrilinosWrappers::MPI::Vector temp_v(locally_owned_dofs_, MPI_COMM_WORLD);
        
        // Use MappingFE for simplex elements
        dealii::MappingFE<dim> mapping(*fe_);
        
        InitialDisplacementFunction disp_func(problem_.get());
        dealii::VectorTools::interpolate(mapping,
                                        dof_handler_, 
                                        disp_func,
                                        temp_u);
        
        InitialVelocityFunction vel_func(problem_.get());
        dealii::VectorTools::interpolate(mapping,
                                        dof_handler_,
                                        vel_func,
                                        temp_v);
        
        // Copy to solution vectors
        solution_u_ = temp_u;
        solution_v_ = temp_v;
        old_solution_u_ = temp_u;
        old_solution_v_ = temp_v;
        
        pcout_ << "  Initial conditions applied successfully!" << std::endl;
    }

    void CGWaveSolver2DParallel::solve_time_step(double /*time*/)
    {
        // Crank-Nicolson time integration (theta = 0.5)
        // M * u^{n+1} + dt^2 * theta * K * u^{n+1} = 
        //     M * (2*u^n - u^{n-1}) - dt^2 * (1-theta) * K * u^n + dt^2 * theta * K * u^{n-1}
        
        const double dt = time_step_;
        const double dt2 = dt * dt;
        
        // Build system matrix: M + dt^2 * theta * K
        system_matrix_.copy_from(mass_matrix_);
        system_matrix_.add(dt2 * theta_, stiffness_matrix_);
        
        // Build RHS
        system_rhs_ = 0.0;
        
        // Term 1: M * (2*u^n - u^{n-1})
        dealii::TrilinosWrappers::MPI::Vector temp(locally_owned_dofs_, MPI_COMM_WORLD);
        temp = solution_u_;
        temp *= 2.0;
        temp.add(-1.0, old_solution_u_);
        
        mass_matrix_.vmult_add(system_rhs_, temp);
        
        // Term 2: - dt^2 * (1-theta) * K * u^n
        stiffness_matrix_.vmult(temp, solution_u_);
        system_rhs_.add(-dt2 * (1.0 - theta_), temp);
        
        // Term 3: + dt^2 * theta * K * u^{n-1}
        stiffness_matrix_.vmult(temp, old_solution_u_);
        system_rhs_.add(dt2 * theta_, temp);
        
        // Solve the system
        dealii::SolverControl solver_control(1000, 1e-12 * system_rhs_.l2_norm());
        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> solver(solver_control);
        
        dealii::TrilinosWrappers::PreconditionSSOR preconditioner;
        preconditioner.initialize(system_matrix_, 
            dealii::TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));
        
        dealii::TrilinosWrappers::MPI::Vector new_solution_u(locally_owned_dofs_, MPI_COMM_WORLD);
        new_solution_u = solution_u_;  // Initial guess
        
        solver.solve(system_matrix_, new_solution_u, system_rhs_, preconditioner);
        
        // Update velocity: v^{n+1} = (u^{n+1} - u^{n-1}) / (2*dt)
        dealii::TrilinosWrappers::MPI::Vector new_solution_v(locally_owned_dofs_, MPI_COMM_WORLD);
        new_solution_v = new_solution_u;
        new_solution_v.add(-1.0, old_solution_u_);
        new_solution_v *= 1.0 / (2.0 * dt);
        
        // Store new solution
        old_solution_u_ = solution_u_;
        old_solution_v_ = solution_v_;
        solution_u_ = new_solution_u;
        solution_v_ = new_solution_v;
    }

    void CGWaveSolver2DParallel::run()
    {
        pcout_ << "===============================================" << std::endl;
        pcout_ << "Running 2D Parallel Wave Equation Solver" << std::endl;
        pcout_ << "===============================================" << std::endl;
        
        // Calculate mesh spacing
        double h_min = std::numeric_limits<double>::max();
        for (const auto &cell : mesh_.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                h_min = std::min(h_min, cell->diameter());
            }
        }
        
        // Get global minimum across all processes
        if (use_mpi_)
        {
            h_min = dealii::Utilities::MPI::min(h_min, MPI_COMM_WORLD);
        }
        
        // Calculate CFL number
        const double cfl_number = wave_speed_ * time_step_ / h_min;
        
        pcout_ << "Mesh spacing (h): " << h_min << std::endl;
        pcout_ << "Time step (dt):   " << time_step_ << std::endl;
        pcout_ << "Wave speed (c):   " << wave_speed_ << std::endl;
        pcout_ << "CFL number:       " << cfl_number << std::endl;
        pcout_ << "  Note: Crank-Nicolson (theta=0.5) is unconditionally stable." << std::endl;
        pcout_ << "        Recommended CFL < 1 for accuracy." << std::endl;
        
        if (cfl_number > 1.0)
        {
            pcout_ << "  WARNING: CFL > 1.0 may reduce accuracy!" << std::endl;
        }
        else
        {
            pcout_ << "  CFL condition satisfied for accuracy." << std::endl;
        }
        pcout_ << std::endl;
        
        pcout_ << "Final time: " << final_time_ << std::endl;
        
        apply_initial_conditions();
        
        // Compute the number of time steps
        const unsigned int n_time_steps = 
            static_cast<unsigned int>(std::round(final_time_ / time_step_));
        
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
                pcout_ << "Step " << step << ", Time = " << time << std::endl;
            }
            
            solve_time_step(time);
            
            if (output_interval_ > 0 && step % output_interval_ == 0)
            {
                output_results(step, time);
            }
        }
        
        // Final output
        if (output_interval_ > 0)
        {
            output_results(step, time);
        }
        
        pcout_ << "Simulation completed in " << step << " steps" << std::endl;
        pcout_ << "Final energy: " << compute_energy() << std::endl;
        
        pcout_ << "===============================================" << std::endl;
    }

    double CGWaveSolver2DParallel::compute_energy() const
    {
        // Energy = 0.5 * (v^T * M * v + u^T * K * u)
        
        dealii::TrilinosWrappers::MPI::Vector temp(locally_owned_dofs_, MPI_COMM_WORLD);
        
        // Kinetic energy: 0.5 * v^T * M * v
        mass_matrix_.vmult(temp, solution_v_);
        double kinetic_energy = 0.5 * (solution_v_ * temp);
        
        // Potential energy: 0.5 * u^T * K * u
        stiffness_matrix_.vmult(temp, solution_u_);
        double potential_energy = 0.5 * (solution_u_ * temp);
        
        return kinetic_energy + potential_energy;
    }

    void CGWaveSolver2DParallel::output_results(unsigned int step, double time) const
    {
        // Create a vector with ghosts for output
        dealii::TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs_,
                                                             locally_relevant_dofs_,
                                                             MPI_COMM_WORLD);
        solution_ghost = solution_u_;
        
        // Build DataOut
        dealii::DataOut<dim> data_out;
        data_out.add_data_vector(dof_handler_, solution_ghost, "u");
        
        // Add partitioning info
        std::vector<unsigned int> partition_int(mesh_.n_active_cells());
        dealii::GridTools::get_subdomain_association(mesh_, partition_int);
        const dealii::Vector<double> partitioning(partition_int.begin(), 
                                                  partition_int.end());
        data_out.add_data_vector(partitioning, "partitioning");
        
        data_out.build_patches();
        
        // Write VTU files with PVTU master file
        // Use a simple basename without step number - the time index handles it
        data_out.write_vtu_with_pvtu_record("./", "solution", step, MPI_COMM_WORLD);
        
        if (mpi_rank_ == 0)
        {
            std::cout << "Output written for step " << step 
                     << " (t=" << time << ")" << std::endl;
        }
    }

} // namespace WaveEquation
