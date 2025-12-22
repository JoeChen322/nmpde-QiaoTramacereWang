#include "Wave.hpp"

#include <cerrno>
#include <iomanip>
#include <stdexcept>

// POSIX mkdir
#include <sys/stat.h>
#include <sys/types.h>

// MPI barrier
#include <mpi.h>

// Ensure that the given directory exists (create it if necessary).

static void ensure_directory_exists(const std::string &dir,
                                    const unsigned int mpi_rank)
{
  if (dir.empty() || dir == ".")
    return;

  if (mpi_rank == 0)
  {
    const int rc = ::mkdir(dir.c_str(), 0755);
    if (rc != 0 && errno != EEXIST)
      throw std::runtime_error("mkdir failed for output dir: " + dir);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Constructor: initialize parameters and MPI-related variables.

Wave::Wave(const std::string &mesh_file_name_,
           const unsigned int &degree_,
           const double &T_,
           const double &deltat_,
           const double &theta_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)), mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)), pcout(std::cout, mpi_rank == 0), initial_u(1, 1) // default mode (1,1)
      ,
      T(T_), mesh_file_name(mesh_file_name_), degree(degree_), deltat(deltat_), theta(theta_), mesh(MPI_COMM_WORLD)
{
  AssertThrow(deltat > 0.0, ExcMessage("deltat must be > 0"));
  AssertThrow(theta > 0.0, ExcMessage("theta must be > 0"));
}

// Setup mesh, FE space, DoF handler, and linear algebra objects.

void Wave::setup()
{
  ensure_directory_exists(output_dir, mpi_rank);

  // Mesh
  if (verbose)
    pcout << "Initializing the mesh" << std::endl;

  {
    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);

    const auto construction_data =
        TriangulationDescription::Utilities::create_description_from_triangulation(
            mesh_serial, MPI_COMM_WORLD);

    mesh.create_triangulation(construction_data);

    if (verbose)
      pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
  }

  if (verbose)
    pcout << "-----------------------------------------------" << std::endl;

  // FE space
  if (verbose)
    pcout << "Initializing the finite element space" << std::endl;

  {
    fe = std::make_unique<FE_SimplexP<dim>>(degree);
    quadrature = std::make_unique<QGaussSimplex<dim>>(degree + 1);

    if (verbose)
    {
      pcout << "  Degree                     = " << fe->degree << std::endl;
      pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;
      pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
    }
  }

  if (verbose)
    pcout << "-----------------------------------------------" << std::endl;

  // DoF handler
  if (verbose)
    pcout << "Initializing the DoF handler" << std::endl;

  {
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    if (verbose)
      pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  if (verbose)
    pcout << "-----------------------------------------------" << std::endl;

  // Linear algebra objects
  if (verbose)
    pcout << "Initializing the linear system" << std::endl;

  {
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               locally_owned_dofs,
                                               locally_relevant_dofs,
                                               MPI_COMM_WORLD);

    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);

    matrix_u_base.reinit(sparsity);
    rhs_operator_u.reinit(sparsity);

    matrix_u.reinit(sparsity);
    matrix_v.reinit(sparsity);

    rhs_u.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    rhs_v.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    forcing_terms.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    u_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    u.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    v_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    v.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

// Assemble mass and stiffness matrices, and prebuild system matrices used in time-stepping.

void Wave::assemble_matrices()
{
  if (verbose)
  {
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the system matrices (M, A)" << std::endl;
  }

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiff(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_mass = 0.0;
    cell_stiff = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      const double mu_q = mu.value(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          cell_mass(i, j) += fe_values.shape_value(i, q) *
                             fe_values.shape_value(j, q) *
                             fe_values.JxW(q);

          cell_stiff(i, j) += mu_q *
                              (fe_values.shape_grad(i, q) *
                               fe_values.shape_grad(j, q)) *
                              fe_values.JxW(q);
        }
    }

    cell->get_dof_indices(dof_indices);
    mass_matrix.add(dof_indices, cell_mass);
    stiffness_matrix.add(dof_indices, cell_stiff);
  }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // Prebuild the (unconstrained) u-system matrix used at every time step:
  //   matrix_u_base = M + theta^2 dt^2 A
  matrix_u_base.copy_from(mass_matrix);
  matrix_u_base.add(theta * theta * deltat * deltat, stiffness_matrix);

  // Prebuild the operator multiplying u^n in the u-RHS:
  //   rhs_operator_u = M - theta(1-theta) dt^2 A
  rhs_operator_u.copy_from(mass_matrix);
  rhs_operator_u.add(-theta * (1.0 - theta) * deltat * deltat, stiffness_matrix);
}

// Compute Dirichlet boundary values for u and v at given time.

void Wave::compute_boundary_values(
    const double time,
    std::map<types::global_dof_index, double> &bv_u,
    std::map<types::global_dof_index, double> &bv_v) const
{
  bv_u.clear();
  bv_v.clear();

  std::map<types::boundary_id, const Function<dim> *> boundary_functions_u;
  std::map<types::boundary_id, const Function<dim> *> boundary_functions_v;

  BoundaryValuesU boundary_u_local = boundary_u;
  BoundaryValuesV boundary_v_local = boundary_v;
  boundary_u_local.set_time(time);
  boundary_v_local.set_time(time);

  for (const auto id : mesh.get_boundary_ids())
  {
    boundary_functions_u[id] = &boundary_u_local;
    boundary_functions_v[id] = &boundary_v_local;
  }

  VectorTools::interpolate_boundary_values(dof_handler,
                                           boundary_functions_u,
                                           bv_u);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           boundary_functions_v,
                                           bv_v);
}

// forcing_terms stores dt * F_theta where F_theta = (1-theta)F^n + theta F^{n+1}.
// u-RHS uses + theta*dt*forcing_terms (=> theta*dt^2*F_theta).
void Wave::assemble_rhs_u(const double time,
                          const TrilinosWrappers::MPI::Vector &old_u,
                          const TrilinosWrappers::MPI::Vector &old_v)
{
  rhs_u = 0.0;
  forcing_terms = 0.0;

  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, MPI_COMM_WORLD);

  // rhs_u = (M - theta(1-theta) dt^2 A) * u^n
  rhs_operator_u.vmult(rhs_u, old_u);

  // rhs_u += dt * M * v^n
  mass_matrix.vmult(tmp, old_v);
  rhs_u.add(deltat, tmp);

  // Assemble forcing_terms = dt*(theta F^{n+1} + (1-theta) F^n) in load-vector form
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                              update_JxW_values);

  Vector<double> cell_force(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);
    cell_force = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      forcing_term.set_time(time);
      const double f_new = forcing_term.value(fe_values.quadrature_point(q));

      forcing_term.set_time(time - deltat);
      const double f_old = forcing_term.value(fe_values.quadrature_point(q));

      const double f_theta = theta * f_new + (1.0 - theta) * f_old;

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        cell_force(i) += (deltat * f_theta) *
                         fe_values.shape_value(i, q) *
                         fe_values.JxW(q);
    }

    cell->get_dof_indices(dof_indices);
    forcing_terms.add(dof_indices, cell_force);
  }

  forcing_terms.compress(VectorOperation::add);

  // u-RHS forcing contribution: + theta*dt * forcing_terms = theta*dt^2*F_theta
  rhs_u.add(theta * deltat, forcing_terms);

  // Constrained u-matrix for this step: matrix_u = matrix_u_base with Dirichlet BC applied.
  matrix_u.copy_from(matrix_u_base);

  std::map<types::global_dof_index, double> bv_u, bv_v_unused;
  compute_boundary_values(time, bv_u, bv_v_unused);

  MatrixTools::apply_boundary_values(bv_u,
                                     matrix_u,
                                     u_owned,
                                     rhs_u,
                                     true);
}

// Velocity update: solve M v^{n+1} = rhs_v (with BC applied).
void Wave::assemble_rhs_v(const double time,
                          const TrilinosWrappers::MPI::Vector &old_u,
                          const TrilinosWrappers::MPI::Vector &old_v)
{
  rhs_v = 0.0;

  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, MPI_COMM_WORLD);

  // rhs_v = -theta dt A u^{n+1}
  stiffness_matrix.vmult(rhs_v, u_owned);
  rhs_v *= (-theta * deltat);

  // rhs_v += M v^n
  mass_matrix.vmult(tmp, old_v);
  rhs_v += tmp;

  // rhs_v += -(1-theta) dt A u^n
  stiffness_matrix.vmult(tmp, old_u);
  rhs_v.add(-(1.0 - theta) * deltat, tmp);

  // rhs_v += forcing_terms  (forcing_terms = dt * F_theta)
  rhs_v += forcing_terms;

  // Constrained v-matrix for this step: matrix_v = M with Dirichlet BC applied for v.
  matrix_v.copy_from(mass_matrix);

  std::map<types::global_dof_index, double> bv_u_unused, bv_v;
  compute_boundary_values(time, bv_u_unused, bv_v);

  MatrixTools::apply_boundary_values(bv_v,
                                     matrix_v,
                                     v_owned,
                                     rhs_v,
                                     true);
}

void Wave::initialize_preconditioner_u()
{
  if (preconditioner_u_initialized)
    return;

  preconditioner_u.initialize(matrix_u,
                              TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));
  preconditioner_u_initialized = true;
}

void Wave::initialize_preconditioner_v()
{
  if (preconditioner_v_initialized)
    return;

  preconditioner_v.initialize(matrix_v,
                              TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));
  preconditioner_v_initialized = true;
}

void Wave::solve_u()
{
  ReductionControl solver_control(/*max_steps*/ 5000,
                                  /*absolute*/ 1e-14,
                                  /*relative*/ 1e-12);
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  initialize_preconditioner_u();

  solver.solve(matrix_u, u_owned, rhs_u, preconditioner_u);

  if (verbose)
    pcout << "  u: " << solver_control.last_step() << " CG iterations\n";

  u = u_owned;
  u.update_ghost_values();
}

void Wave::solve_v()
{
  ReductionControl solver_control(/*max_steps*/ 5000,
                                  /*absolute*/ 1e-14,
                                  /*relative*/ 1e-12);
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  initialize_preconditioner_v();

  solver.solve(matrix_v, v_owned, rhs_v, preconditioner_v);

  if (verbose)
    pcout << "  v: " << solver_control.last_step() << " CG iterations\n";

  v = v_owned;
  v.update_ghost_values();
}

/******************************Begin energy utilities******************************/

// Total energy: E = 0.5 * (v^T M v + u^T A u)

double Wave::energy() const
{
  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, MPI_COMM_WORLD);

  mass_matrix.vmult(tmp, v_owned);
  const double kinetic = v_owned * tmp;

  stiffness_matrix.vmult(tmp, u_owned);
  const double potential = u_owned * tmp;

  return 0.5 * (kinetic + potential);
}

// Minimum cell diameter over locally owned cells, then global min over all MPI ranks.

double Wave::compute_min_cell_diameter() const
{
  double h_min = std::numeric_limits<double>::max();

  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned())
      h_min = std::min(h_min, cell->diameter());

  h_min = Utilities::MPI::min(h_min, MPI_COMM_WORLD);
  return h_min;
}

// Compute cell-wise energy density and store in cell_energy_density vector for potential visualization.

void Wave::compute_cell_energy_density(Vector<double> &cell_energy_density) const
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
  Vector<double> u_loc(dofs_per_cell);
  Vector<double> v_loc(dofs_per_cell);

  cell_energy_density.reinit(mesh.n_active_cells());

  unsigned int cell_index = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    // FIX: must skip artificial/non-owned BEFORE fe_values.reinit(cell)
    if (cell->is_artificial() || !cell->is_locally_owned())
    {
      cell_energy_density[cell_index++] = 0.0;
      continue;
    }

    fe_values.reinit(cell);
    cell->get_dof_indices(dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      u_loc[i] = u[dof_indices[i]];
      v_loc[i] = v[dof_indices[i]];
    }

    double E_cell = 0.0;
    double vol = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      double u_q = 0.0;
      double v_q = 0.0;
      Tensor<1, dim> grad_u_q;

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i, q);
        const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

        u_q += u_loc[i] * phi_i;
        v_q += v_loc[i] * phi_i;
        grad_u_q += u_loc[i] * grad_phi_i;
      }

      const double mu_q = mu.value(fe_values.quadrature_point(q));
      const double density_q = 0.5 * (v_q * v_q + mu_q * grad_u_q.norm_square());

      const double JxW = fe_values.JxW(q);
      E_cell += density_q * JxW;
      vol += JxW;
    }

    cell_energy_density[cell_index++] = (vol > 0.0 ? E_cell / vol : 0.0);
  }
}

/******************************End of energy utilities******************************/

void Wave::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  data_out.add_data_vector(u, "U");
  data_out.add_data_vector(v, "V");

  Vector<double> cell_energy_density;
  compute_cell_energy_density(cell_energy_density);
  data_out.add_data_vector(cell_energy_density, "energy_density");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(output_dir,
                                      "output",
                                      time_step,
                                      MPI_COMM_WORLD,
                                      3);
}

void Wave::solve()
{
  assemble_matrices();

  // Diagnostics (optional)
  if (verbose)
  {
    const double h_min = compute_min_cell_diameter();
    const double c = 1.0;
    const double cfl = c * deltat / h_min;

    if (mpi_rank == 0)
    {
      std::cout << "===============================================\n";
      std::cout << "Diagnostics\n";
      std::cout << "Mesh spacing h (min diameter): " << h_min << "\n";
      std::cout << "Time step dt:                  " << deltat << "\n";
      std::cout << "CFL number (c*dt/h):           " << cfl << "\n";
      std::cout << "Mode (m,n):                    (" << mode_m << "," << mode_n << ")\n";
      std::cout << "===============================================\n";
    }
  }

  // Initial conditions
  {
    VectorTools::interpolate(dof_handler, initial_u, u_owned);
    VectorTools::interpolate(dof_handler, initial_v, v_owned);

    // Enforce BCs at t=0
    std::map<types::global_dof_index, double> bv_u0, bv_v0;
    compute_boundary_values(0.0, bv_u0, bv_v0);

    for (const auto &it : bv_u0)
      if (locally_owned_dofs.is_element(it.first))
        u_owned[it.first] = it.second;

    for (const auto &it : bv_v0)
      if (locally_owned_dofs.is_element(it.first))
        v_owned[it.first] = it.second;

    u = u_owned;
    v = v_owned;

    u.update_ghost_values();
    v.update_ghost_values();

    if (output_interval > 0)
      output(0);
  }

  // Set and store initial energy after ICs/BCs have been applied
  energy_initial = energy();

  // Optional energy logging (rank 0 only)
  std::ofstream energy_out;
  if (energy_log_enabled && mpi_rank == 0)
  {
    energy_out.open(energy_log_file);
    if (!energy_out)
      throw std::runtime_error("Failed to open energy log file: " + energy_log_file);

    energy_out << "step,time,energy,E_over_E0\n";
    energy_out << std::setprecision(16);

    const double E0 = energy_initial;
    const double ratio = (energy_log_normalize && E0 > 0.0) ? (E0 / E0) : 1.0;
    energy_out << 0 << "," << 0.0 << "," << E0 << "," << ratio << "\n";
  }

  auto log_energy = [&](const unsigned int step, const double time, const double E)
  {
    if (!energy_log_enabled || mpi_rank != 0)
      return;
    if ((step % energy_log_stride) != 0)
      return;

    const double E0 = energy_initial;
    const double ratio = (energy_log_normalize && E0 > 0.0) ? (E / E0) : 0.0;

    energy_out << step << "," << time << "," << E << "," << ratio << "\n";
  };

  TrilinosWrappers::MPI::Vector old_u(u_owned);
  TrilinosWrappers::MPI::Vector old_v(v_owned);

  // Use an integer step loop to avoid floating-point drift in (time += dt).
  const double steps_real = T / deltat;
  const unsigned int n_steps = static_cast<unsigned int>(std::llround(steps_real));

  AssertThrow(std::abs(steps_real - static_cast<double>(n_steps)) < 1e-12,
              ExcMessage("T/dt must be an integer (within tolerance) for this solver."));

  for (unsigned int step = 1; step <= n_steps; ++step)
  {
    const double time = step * deltat;

    if (verbose)
    {
      pcout << "n = " << std::setw(6) << step
            << ", t = " << std::setw(16) << std::setprecision(12) << time
            << ":" << std::flush;
    }

    assemble_rhs_u(time, old_u, old_v);
    solve_u();

    assemble_rhs_v(time, old_u, old_v);
    solve_v();

    const double En = energy();

    if (verbose)
      pcout << "  E^n = " << std::setprecision(16) << En << std::endl;

    log_energy(step, time, En);

    if (output_interval > 0 && (step % output_interval == 0))
      output(step);

    old_u = u_owned;
    old_v = v_owned;
  }

  if (energy_out.is_open())
    energy_out.close();
}

// -----------------------------------------------------------------------------
// L2 error computation for time-/space-convergence studies
// -----------------------------------------------------------------------------

double Wave::compute_L2_error_u(const double time) const
{
  ExactSolutionU exact_u(mode_m, mode_n);
  exact_u.set_time(time);

  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  QGaussSimplex<dim> quadrature_error(fe->degree + 2);
  const unsigned int n_q = quadrature_error.size();

  FEValues<dim> fe_values(*fe,
                          quadrature_error,
                          update_values | update_quadrature_points | update_JxW_values);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
  Vector<double> u_loc(dofs_per_cell);

  double local_sum = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned() || cell->is_artificial())
      continue;

    fe_values.reinit(cell);
    cell->get_dof_indices(dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      u_loc[i] = u[dof_indices[i]];

    for (unsigned int q = 0; q < n_q; ++q)
    {
      double uh = 0.0;
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        uh += u_loc[i] * fe_values.shape_value(i, q);

      const double ue = exact_u.value(fe_values.quadrature_point(q));
      const double diff = uh - ue;

      local_sum += diff * diff * fe_values.JxW(q);
    }
  }

  const double global_sum = Utilities::MPI::sum(local_sum, MPI_COMM_WORLD);
  return std::sqrt(global_sum);
}

double Wave::compute_L2_error_v(const double time) const
{
  ExactSolutionV exact_v(mode_m, mode_n);
  exact_v.set_time(time);

  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  QGaussSimplex<dim> quadrature_error(fe->degree + 2);
  const unsigned int n_q = quadrature_error.size();

  FEValues<dim> fe_values(*fe,
                          quadrature_error,
                          update_values | update_quadrature_points | update_JxW_values);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
  Vector<double> v_loc(dofs_per_cell);

  double local_sum = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned() || cell->is_artificial())
      continue;

    fe_values.reinit(cell);
    cell->get_dof_indices(dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      v_loc[i] = v[dof_indices[i]];

    for (unsigned int q = 0; q < n_q; ++q)
    {
      double vh = 0.0;
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        vh += v_loc[i] * fe_values.shape_value(i, q);

      const double ve = exact_v.value(fe_values.quadrature_point(q));
      const double diff = vh - ve;

      local_sum += diff * diff * fe_values.JxW(q);
    }
  }

  const double global_sum = Utilities::MPI::sum(local_sum, MPI_COMM_WORLD);
  return std::sqrt(global_sum);
}
