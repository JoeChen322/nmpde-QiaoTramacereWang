#ifndef WAVE_HPP
#define WAVE_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>

using namespace dealii;

// Class representing the wave equation problem.
class Wave
{
public:
  static constexpr unsigned int dim = 2;

  // mu coefficient (material / wave speed coefficient in stiffness A).
  class FunctionMu : public Function<dim>
  {
  public:
    double value(const Point<dim> & /*p*/,
                 const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };

  // Forcing term f(x,t).
  class ForcingTerm : public Function<dim>
  {
  public:
    double value(const Point<dim> & /*p*/,
                 const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Initial displacement u0(x) = sin(m*pi*x) sin(n*pi*y).
  class InitialValuesU : public Function<dim>
  {
  public:
    InitialValuesU(const unsigned int m_ = 1, const unsigned int n_ = 1)
        : m(m_), n(n_) {}

    void set_mode(const unsigned int m_, const unsigned int n_)
    {
      m = m_;
      n = n_;
    }

    double value(const Point<dim> &p,
                 const unsigned int /*component*/ = 0) const override
    {
      return std::sin(numbers::PI * static_cast<double>(m) * p[0]) *
             std::sin(numbers::PI * static_cast<double>(n) * p[1]);
    }

  private:
    unsigned int m = 1;
    unsigned int n = 1;
  };

  // Initial velocity v0(x) = u_t(x,0).
  class InitialValuesV : public Function<dim>
  {
  public:
    double value(const Point<dim> & /*p*/,
                 const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Dirichlet boundary data g(x,t) for u.
  class BoundaryValuesU : public Function<dim>
  {
  public:
    double value(const Point<dim> & /*p*/,
                 const unsigned int /*component*/ = 0) const override
    {
      return 0.0; // homogeneous Dirichlet for baseline tests
    }
  };

  // Time derivative g_t(x,t) for v=u_t on the boundary.
  class BoundaryValuesV : public Function<dim>
  {
  public:
    double value(const Point<dim> & /*p*/,
                 const unsigned int /*component*/ = 0) const override
    {
      return 0.0; // consistent with g=0 in baseline
    }
  };

  // Exact eigenmode solution for convergence tests (matches InitialValuesU).
  class ExactSolutionU : public Function<dim>
  {
  public:
    ExactSolutionU(const unsigned int m_ = 1, const unsigned int n_ = 1)
        : m(m_), n(n_) {}

    double value(const Point<dim> &p,
                 const unsigned int /*component*/ = 0) const override
    {
      const double omega =
          numbers::PI * std::sqrt(static_cast<double>(m * m + n * n));

      return std::sin(numbers::PI * static_cast<double>(m) * p[0]) *
             std::sin(numbers::PI * static_cast<double>(n) * p[1]) *
             std::cos(omega * this->get_time());
    }

  private:
    unsigned int m = 1;
    unsigned int n = 1;
  };

  class ExactSolutionV : public Function<dim>
  {
  public:
    ExactSolutionV(const unsigned int m_ = 1, const unsigned int n_ = 1)
        : m(m_), n(n_) {}

    double value(const Point<dim> &p,
                 const unsigned int /*component*/ = 0) const override
    {
      const double omega =
          numbers::PI * std::sqrt(static_cast<double>(m * m + n * n));

      return -omega *
             std::sin(numbers::PI * static_cast<double>(m) * p[0]) *
             std::sin(numbers::PI * static_cast<double>(n) * p[1]) *
             std::sin(omega * this->get_time());
    }

  private:
    unsigned int m = 1;
    unsigned int n = 1;
  };

  // Energy access for post-processing/tests
  double get_energy() const { return energy(); }
  double get_initial_energy() const { return energy_initial; }

  // Control printing (important for long dissipation runs)
  void set_verbose(const bool v) { verbose = v; }

  // Set eigenmode for initial condition and exact solution (for studies)
  void set_mode(const unsigned int m, const unsigned int n)
  {
    mode_m = m;
    mode_n = n;
    initial_u.set_mode(m, n);
  }

  // Write energy history as CSV (rank 0 only).
  // CSV columns: step,time,energy,E_over_E0
  void enable_energy_log(const std::string &csv_file,
                         const unsigned int stride = 1,
                         const bool normalize = true)
  {
    energy_log_enabled = true;
    energy_log_file = csv_file;
    energy_log_stride = (stride > 0 ? stride : 1);
    energy_log_normalize = normalize;
  }

  void disable_energy_log() { energy_log_enabled = false; }

  Wave(const std::string &mesh_file_name_,
       const unsigned int &degree_,
       const double &T_,
       const double &deltat_,
       const double &theta_);

  void set_output_interval(const unsigned int k) { output_interval = k; }
  void set_output_directory(const std::string &dir) { output_dir = dir; }

  // Mesh/DoF diagnostics for convergence tests
  double get_h_min() const { return compute_min_cell_diameter(); }
  unsigned long long n_cells() const { return mesh.n_global_active_cells(); }
  types::global_dof_index n_dofs() const { return dof_handler.n_dofs(); }

  void setup();
  void solve();

  // L2 errors against the built-in exact eigenmode (time-dependent).
  double compute_L2_error_u(const double time) const;
  double compute_L2_error_v(const double time) const;

private:
  // Assemble time-independent FE matrices: mass_matrix (M) and stiffness_matrix (A).
  void assemble_matrices();

  // Assemble u-RHS (step-23 style naming):
  // builds rhs_u, forcing_terms, and matrix_u (constrained copy of matrix_u_base).
  void assemble_rhs_u(const double time,
                      const TrilinosWrappers::MPI::Vector &old_u,
                      const TrilinosWrappers::MPI::Vector &old_v);

  // Assemble v-RHS (step-23 style velocity update):
  // builds rhs_v and matrix_v (constrained copy of mass_matrix).
  void assemble_rhs_v(const double time,
                      const TrilinosWrappers::MPI::Vector &old_u,
                      const TrilinosWrappers::MPI::Vector &old_v);

  void solve_u();
  void solve_v();

  void initialize_preconditioner_u();
  void initialize_preconditioner_v();

  void output(const unsigned int &time_step) const;

  double energy() const;

  void compute_cell_energy_density(Vector<double> &cell_energy_density) const;

  double compute_min_cell_diameter() const;

  void compute_boundary_values(const double time,
                               std::map<types::global_dof_index, double> &bv_u,
                               std::map<types::global_dof_index, double> &bv_v) const;

  // MPI
  const unsigned int mpi_size;
  const unsigned int mpi_rank;
  ConditionalOStream pcout;

  // Problem definition
  FunctionMu mu;
  ForcingTerm forcing_term;
  InitialValuesU initial_u;
  InitialValuesV initial_v;
  BoundaryValuesU boundary_u;
  BoundaryValuesV boundary_v;

  // Mode for exact solution (and for convergence experiments)
  unsigned int mode_m = 1;
  unsigned int mode_n = 1;

  // Final time
  const double T;

  // Discretization
  const std::string mesh_file_name;
  const unsigned int degree;
  const double deltat;
  const double theta;

  // Output controls
  unsigned int output_interval = 1;
  std::string output_dir = "./";

  // Verbosity
  bool verbose = true;

  // Mesh
  parallel::fullydistributed::Triangulation<dim> mesh;

  // FE space and quadrature
  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoFs
  DoFHandler<dim> dof_handler;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  // FE matrices (time-independent)
  TrilinosWrappers::SparseMatrix mass_matrix;      // M
  TrilinosWrappers::SparseMatrix stiffness_matrix; // A

  // Time-step matrices (step-23 naming)
  TrilinosWrappers::SparseMatrix matrix_u_base;  // M + theta^2 dt^2 A (unconstrained)
  TrilinosWrappers::SparseMatrix rhs_operator_u; // (M - theta(1-theta) dt^2 A) multiplies u^n in RHS
  TrilinosWrappers::SparseMatrix matrix_u;       // constrained system for u

  TrilinosWrappers::SparseMatrix matrix_v; // constrained system for v (copy of M)

  // RHS vectors
  TrilinosWrappers::MPI::Vector rhs_u;
  TrilinosWrappers::MPI::Vector rhs_v;

  // Unknowns (owned + ghosted)
  TrilinosWrappers::MPI::Vector u_owned;
  TrilinosWrappers::MPI::Vector u; // ghosted

  TrilinosWrappers::MPI::Vector v_owned;
  TrilinosWrappers::MPI::Vector v; // ghosted

  // forcing_terms stores dt * F_theta, where F_theta = (1-theta)F^n + theta F^{n+1}
  TrilinosWrappers::MPI::Vector forcing_terms;

  // Preconditioners
  TrilinosWrappers::PreconditionSSOR preconditioner_u;
  bool preconditioner_u_initialized = false;

  TrilinosWrappers::PreconditionSSOR preconditioner_v;
  bool preconditioner_v_initialized = false;

  // Energy logging
  bool energy_log_enabled = false;
  std::string energy_log_file = "energy.csv";
  unsigned int energy_log_stride = 1;
  bool energy_log_normalize = true;

  // Stored initial energy (set in solve() after ICs are applied)
  double energy_initial = -1.0;
};

#endif
