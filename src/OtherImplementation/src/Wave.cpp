#include "Wave.hpp"

void
Wave::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the derivative vector" << std::endl;
    derivative_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    derivative.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void
Wave::assemble_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_mass_matrix      = 0.0;
      cell_stiffness_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double mu_loc = mu.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) * fe_values.JxW(q);

                  cell_stiffness_matrix(i, j) +=
                    mu_loc * fe_values.shape_grad(i, q) *
                    fe_values.shape_grad(j, q) * fe_values.JxW(q);
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      mass_matrix.add(dof_indices, cell_mass_matrix);
      stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
    }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // We build the matrix on the left-hand side of the algebraic problem (the one
  // that we'll invert at each timestep).
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(theta*theta*deltat*deltat, stiffness_matrix);

  // We build the matrix on the right-hand side (the one that multiplies the old
  // solution un in place to leverage memory using).
  rhs_matrix.copy_from(mass_matrix);
  rhs_matrix.add(-(1.0 - theta)*theta*deltat*deltat, stiffness_matrix);

}

void
Wave::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // We need to compute the forcing term at the current time (tn+1) and
          // at the old time (tn). deal.II Functions can be computed at a
          // specific time by calling their set_time method.

          // Compute f(tn+1)
          forcing_term.set_time(time);
          const double f_new_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          // Compute f(tn)
          forcing_term.set_time(time - deltat);
          const double f_old_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += (theta * theta * deltat * deltat * f_new_loc + 
                             (1.0 - theta) * theta * deltat * deltat * f_old_loc) *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_rhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  // system_rhs += rhs_matrix * solution_owned;
  rhs_matrix.vmult_add(system_rhs, solution_owned);
  // Add the contribution from the mass matrix times the derivative to the RHS.
  {
  TrilinosWrappers::MPI::Vector tmp(derivative_owned);
  tmp *= deltat;                             
  mass_matrix.vmult_add(system_rhs, tmp);
  }

  // Boundary conditions.
  {
    // We construct a map that stores, for each DoF corresponding to a
    // Dirichlet condition, the corresponding value. E.g., if the Dirichlet
    // condition is u_i = b, the map will contain the pair (i, b).
    std::map<types::global_dof_index, double> boundary_values;

    // Then, we build a map that, for each boundary tag, stores the
    // corresponding boundary function.

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    for (unsigned int i = 0; i < 6; ++i)
      boundary_functions[i] = &function_g;

    // interpolate_boundary_values fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    // Finally, we modify the linear system to apply the boundary
    // conditions. This replaces the equations for the boundary DoFs with
    // the corresponding u_i = 0 equations.
    MatrixTools::apply_boundary_values(
      boundary_values, lhs_matrix, solution_owned, system_rhs, true);
  }

  pcout << "  ||u^n|| = " << solution_owned.l2_norm()
      << ", ||v^n|| = " << derivative_owned.l2_norm() << std::endl;

}

void 
Wave::solve_time_step(const TrilinosWrappers::MPI::Vector &u_old,
                           const TrilinosWrappers::MPI::Vector &v_old)
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(lhs_matrix,
    TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  // solve for u^{n+1}
  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations\n";

  solution = solution_owned;

  // now update v^{n+1} using the clean formula:
  //
  // v^{n+1} = ((theta-1)/theta) v^n + (1/(theta*k)) (u^{n+1} - u^n)
  //
  derivative_owned = v_old;               // start from v^n
  derivative_owned *= (theta - 1.0)/theta;

  TrilinosWrappers::MPI::Vector diff(solution_owned);
  diff.add(-1.0, u_old);                  // diff = u^{n+1} - u^n
  diff *= 1.0/(theta * deltat);

  derivative_owned += diff;
  derivative = derivative_owned;
}

void Wave::compute_cell_energy(Vector<double> &cell_energy) const
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_JxW_values);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
  Vector<double> u_loc(dofs_per_cell);
  Vector<double> v_loc(dofs_per_cell);

  cell_energy.reinit(mesh.n_active_cells());

  unsigned int cell_index = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(dof_indices);

      // gather local dof values of u and v
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          u_loc[i] = solution[dof_indices[i]];
          v_loc[i] = derivative[dof_indices[i]];
        }

      double E_cell = 0.0;
      double vol    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          double u_q = 0.0;
          double v_q = 0.0;
          Tensor<1, dim> grad_u_q;

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double phi_i      = fe_values.shape_value(i, q);
              const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

              u_q      += u_loc[i] * phi_i;
              v_q      += v_loc[i] * phi_i;
              grad_u_q += u_loc[i] * grad_phi_i;
            }

          const double density_q =
            0.5 * (v_q * v_q + grad_u_q.norm_square());

          const double JxW = fe_values.JxW(q);
          E_cell += density_q * JxW; // integral over cell
          vol    += JxW;
        }

      // store *average* energy density on this cell
      cell_energy[cell_index++] = E_cell / vol;
    }
}

void Wave::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  // Displacement u
  data_out.add_data_vector(solution, "u");

  // cell-wise average energy density
  Vector<double> cell_energy;
  compute_cell_energy(cell_energy);
  data_out.add_data_vector(cell_energy, "energy");

  // Optional: partitioning
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "output", time_step, MPI_COMM_WORLD, 3);
}


void
Wave::solve()
{
  assemble_matrices();

  pcout << "===============================================" << std::endl;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;
    VectorTools::interpolate(dof_handler, w_0, derivative_owned);
    derivative = derivative_owned;

    // Output the initial solution.
    output(0);// write the initial condition value to file at time step 0
    pcout << "-----------------------------------------------" << std::endl;
  }

  TrilinosWrappers::MPI::Vector u_old(solution_owned);
  TrilinosWrappers::MPI::Vector v_old(derivative_owned);

  unsigned int time_step = 0;
  double       time      = 0;

  while (time < T)
    {
      time += deltat;
      ++time_step;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      // use u_old, v_old when assembling RHS for u^{n+1}
      assemble_rhs(time);

      solve_time_step(u_old, v_old); // pass them in

      output(time_step);

      // shift current to old for next loop
      u_old = solution_owned;
      v_old = derivative_owned;
    }
}