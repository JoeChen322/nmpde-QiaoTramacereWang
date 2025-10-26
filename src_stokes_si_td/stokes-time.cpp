#include "stokes-time.hpp"

void
StokesTime::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    const std::string mesh_file_name =
      "../mesh/mesh-cube-5.msh";

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

    const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
    const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
    fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity,
                                         dim,
                                         fe_scalar_pressure,
                                         1);

    pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
          << std::endl;
    pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
          << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We want to reorder DoFs so that all velocity DoFs come first, and then
    // all pressure DoFs.
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Besides the locally owned and locally relevant indices for the whole
    // system (velocity and pressure), we will also need those for the
    // individual velocity and pressure blocks.
    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);
    block_owned_dofs[0]    = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1]    = locally_owned_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    pcout << "  Number of DoFs: " << std::endl;
    pcout << "    velocity = " << n_u << std::endl;
    pcout << "    pressure = " << n_p << std::endl;
    pcout << "    total    = " << n_u + n_p << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // Velocity DoFs interact with other velocity DoFs (the weak formulation has
    // terms involving u times v), and pressure DoFs interact with velocity DoFs
    // (there are terms involving p times v or u times q). However, pressure
    // DoFs do not interact with other pressure DoFs (there are no terms
    // involving p times q). We build a table to store this information, so that
    // the sparsity pattern can be built accordingly.
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::none;
            else // other combinations
              coupling[c][d] = DoFTools::always;
          }
      }

    TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                    MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
    sparsity.compress();

    // We also build a sparsity pattern for the pressure mass matrix.
    for (unsigned int c = 0; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::always;
            else // other combinations
              coupling[c][d] = DoFTools::none;
          }
      }
    TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
      block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    coupling,
                                    sparsity_pressure_mass);
    sparsity_pressure_mass.compress();

    pcout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    system_matrix.reinit(sparsity);
    pressure_mass.reinit(sparsity_pressure_mass);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  }
}

void
StokesTime::assemble_matrices(const double &time )
{
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the matrices" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_matrix = 0.0;
    mass_matrix = 0.0;
    pressure_mass = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_matrix = 0.0;
        cell_mass_matrix = 0.0;
        cell_pressure_mass_matrix = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Assembling mass matrix correctly without the time step factor
                    cell_mass_matrix(i, j) += fe_values[velocity].value(i, q) * fe_values[velocity].value(j, q) / deltat * fe_values.JxW(q);

                    // Implicit treatment of the viscous term
                    cell_matrix(i, j) += nu * scalar_product(fe_values[velocity].gradient(i, q), fe_values[velocity].gradient(j, q)) * fe_values.JxW(q);

                    // Implicit treatment of the pressure terms
                    cell_matrix(i, j) -= fe_values[velocity].divergence(i, q) * fe_values[pressure].value(j, q) * fe_values.JxW(q);
                    cell_matrix(i, j) -= fe_values[velocity].divergence(j, q) * fe_values[pressure].value(i, q) * fe_values.JxW(q);

                    // Pressure mass matrix - remains unchanged as it's still the Laplacian of pressure
                    cell_pressure_mass_matrix(i, j) += fe_values[pressure].value(i, q) * fe_values[pressure].value(j, q) / nu * fe_values.JxW(q);
                }
            }
        }

        cell->get_dof_indices(dof_indices);

        mass_matrix.add(dof_indices, cell_mass_matrix);
        system_matrix.add(dof_indices, cell_matrix);
        pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
    }

    system_matrix.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);
    mass_matrix.compress(VectorOperation::add);

    lhs_matrix.copy_from(mass_matrix);
    lhs_matrix.add(theta, system_matrix);

    rhs_matrix.copy_from(mass_matrix);
    rhs_matrix.add(-(1.0 - theta), system_matrix);

    // Dirichlet boundary conditions.
    /*{
      std::map<types::global_dof_index, double>           boundary_values;
      std::map<types::boundary_id, const Function<dim> *> boundary_functions;

      // We interpolate first the inlet velocity condition alone, then the wall
      // condition alone, so that the latter "win" over the former where the two
      // boundaries touch.
      boundary_functions[18] = &inlet_velocity;
      VectorTools::interpolate_boundary_values(dof_handler,
                                              boundary_functions,
                                              boundary_values,
                                              ComponentMask(
                                                {true, true, true, false}));

      boundary_functions.clear();
      Functions::ZeroFunction<dim> zero_function(dim + 1);
      boundary_functions[20] = &zero_function;
      VectorTools::interpolate_boundary_values(dof_handler,
                                              boundary_functions,
                                              boundary_values,
                                              ComponentMask(
                                                {true, true, true, false}));

      MatrixTools::apply_boundary_values(
        boundary_values, lhs_matrix, solution, system_rhs, false);
    }*/
    // TODO 
}

void
StokesTime::assemble_rhs(const double &time){
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the rhs" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q           = quadrature->size();
    const unsigned int n_q_face      = quadrature_face->size();

    FEValues<dim>     fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(*fe,
                                    *quadrature_face,
                                    update_values | update_normal_vectors |
                                      update_JxW_values);
                                      
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_rhs    = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        fe_values.reinit(cell);

        cell_rhs                  = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
          {
            Vector<double> forcing_term_new_loc(dim);
            Vector<double> forcing_term_old_loc(dim);

            forcing_term.set_time(time);
            forcing_term.vector_value(fe_values.quadrature_point(q),
                                      forcing_term_new_loc);


            forcing_term.set_time(time - deltat);
            forcing_term.vector_value(fe_values.quadrature_point(q),
                                      forcing_term_old_loc);


            Tensor<1, dim> forcing_term_tensor_new;
            Tensor<1, dim> forcing_term_tensor_old;

            for (unsigned int d = 0; d < dim; ++d){
              forcing_term_tensor_new[d] = forcing_term_new_loc[d];
              forcing_term_tensor_old[d] = forcing_term_old_loc[d];
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                // Forcing term.
                cell_rhs(i) += scalar_product(theta  * forcing_term_tensor_new +
                                       (1.0 - theta) * forcing_term_tensor_old,
                                              fe_values[velocity].value(i, q)) *
                              fe_values.JxW(q);
              }
          }

        // Boundary integral for Neumann BCs.
      
        if (cell->at_boundary())
          {
            for (unsigned int f = 0; f < cell->n_faces(); ++f)
              {
                std::cout<<"id "<<cell->face(f)->boundary_id()<<std::endl;
                if (cell->face(f)->at_boundary() &&
                    cell->face(f)->boundary_id() == 19)
                  {
                    fe_face_values.reinit(cell, f);

                    for (unsigned int q = 0; q < n_q_face; ++q)
                      {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            cell_rhs(i) +=
                              -p_out *
                              scalar_product(fe_face_values.normal_vector(q),
                                            fe_face_values[velocity].value(i,
                                                                            q)) *
                              fe_face_values.JxW(q);
                          }
                      }
                  }
              }
          }

        cell->get_dof_indices(dof_indices);

        system_rhs.add(dof_indices, cell_rhs);
      }

    system_rhs.compress(VectorOperation::add);

    rhs_matrix.vmult_add(system_rhs, solution_owned);

    // Dirichlet boundary conditions. TODO
    {
      std::map<types::global_dof_index, double>           boundary_values;
      std::map<types::boundary_id, const Function<dim> *> boundary_functions;

      // We interpolate first the inlet velocity condition alone, then the wall
      // condition alone, so that the latter "win" over the former where the two
      // boundaries touch.
      boundary_functions[18] = &inlet_velocity;
      VectorTools::interpolate_boundary_values(dof_handler,
                                              boundary_functions,
                                              boundary_values,
                                              ComponentMask(
                                                {true, true, true, false}));

      boundary_functions.clear();
      Functions::ZeroFunction<dim> zero_function(dim + 1);
      boundary_functions[20] = &zero_function;
      VectorTools::interpolate_boundary_values(dof_handler,
                                              boundary_functions,
                                              boundary_values,
                                              ComponentMask(
                                                {true, true, true, false}));

      MatrixTools::apply_boundary_values(
        boundary_values, lhs_matrix, solution, system_rhs, false);
    }
}

void
StokesTime::solve_time_step()
{
  pcout << "===============================================" << std::endl;

  SolverControl solver_control(2000, 1e-6 * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  //PreconditionBlockDiagonal preconditioner;
  //preconditioner.initialize(system_matrix.block(0, 0),
  //                          pressure_mass.block(1, 1));

  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(lhs_matrix.block(0, 0),
                            pressure_mass.block(1, 1),
                            lhs_matrix.block(1, 0));

  pcout << "Solving the linear system" << std::endl;
  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " GMRES iterations"
        << std::endl;

  solution = solution_owned;
}

void StokesTime::output(const unsigned int &time_step, const double &time) const
{
  pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  std::vector<std::string> names;
  for(unsigned int d=0; d<dim; ++d){
    names.push_back("velocity");
  }
  names.push_back("pressure");

  data_out.add_data_vector(dof_handler,
                           solution,
                           names,
                           data_component_interpretation);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  // Use a consistent naming pattern for Paraview to recognize the sequence
  std::string filename = "output-stokes-" + std::to_string(time_step);
  filename = std::string(4 - std::to_string(time_step).size(), '0') + filename;
  filename += "-t" + std::to_string(time);

  // Write the data in a way that groups the files for Paraview
  data_out.write_vtu_with_pvtu_record(
    "./", "stokes", time_step, MPI_COMM_WORLD, 3);

  pcout << "Output written to " << filename << ".vtu" << std::endl;
  pcout << "===============================================" << std::endl;
}



void
StokesTime::solve()
{
  unsigned int time_step = 0;
  double       time      = 0;

  assemble_matrices(time);
  pcout << "===============================================" << std::endl;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    // TODO check this interpolation
    Functions::ZeroFunction<dim> u0(dim);
    VectorTools::interpolate(dof_handler, u0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0, 0.0);
    pcout << "-----------------------------------------------" << std::endl;
  }



  while (time < T)
    {
      time += deltat;
      ++time_step;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      pcout << std::endl;

      assemble_matrices(time);
      assemble_rhs(time);
      solve_time_step();
      output(time_step, time);
    }
}