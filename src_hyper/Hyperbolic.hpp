#ifndef HYPERBOLIC_HPP
#define HYPERBOLIC_HPP
#include <deal.II/base/config.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <memory>

class Hyperbolic
{
public:
  static const unsigned int dim = 2;

  /**
   * @param mesh_file_name_  网格 .msh
   * @param degree_          FE_Q<degree_> 或 FE_SimplexP<degree_>
   * @param T_               终止时间
   * @param dt_              时间步长
   * @param c2_              波速平方 c^2
   */
  Hyperbolic(const std::string &mesh_file_name_,
             const unsigned int degree_,
             const double       T_,
             const double       dt_,
             const double       c2_);

  // ====== 外部可自定义的物理数据 ======
  // f(x,t)
  class ForcingTerm : public dealii::Function<dim>
  {
  public:
    virtual double value(const dealii::Point<dim> &p,
                         unsigned int /*comp*/=0) const override;
  };
  ForcingTerm forcing_term;

  // g(x,t) on boundary
  class BoundaryValues : public dealii::Function<dim>
  {
  public:
    virtual double value(const dealii::Point<dim> &p,
                         unsigned int /*comp*/=0) const override;
  };
  BoundaryValues g_func;

  // u0(x)
  class InitialU : public dealii::Function<dim>
  {
  public:
    virtual double value(const dealii::Point<dim> &p,
                         unsigned int /*comp*/=0) const override;
  };
  InitialU u0_func;

  // u1(x) = ∂_t u(x,0)
  class InitialV : public dealii::Function<dim>
  {
  public:
    virtual double value(const dealii::Point<dim> &p,
                         unsigned int /*comp*/=0) const override;
  };
  InitialV u1_func;

  // ====== 主流程 ======
  void setup();
  void solve();

private:
  // mesh / dofs
  dealii::Triangulation<dim> mesh;
  dealii::DoFHandler<dim>                           dof_handler;
  std::unique_ptr<dealii::FiniteElement<dim>>       fe;
  std::unique_ptr<dealii::Quadrature<dim>>          quadrature;

  // system matrices
  dealii::TrilinosWrappers::SparseMatrix mass_matrix;
  dealii::TrilinosWrappers::SparseMatrix stiffness_matrix;

  // lumped mass diag and inverse
  dealii::TrilinosWrappers::MPI::Vector M_lumped;
  dealii::TrilinosWrappers::MPI::Vector Minv_lumped;
  dealii::TrilinosWrappers::MPI::Vector ones;

  // state vectors
  // u^n (at integer steps)
  dealii::TrilinosWrappers::MPI::Vector u_owned;
  dealii::TrilinosWrappers::MPI::Vector u;          // ghosted for output

  // v^{n+1/2} (velocity at half step)
  dealii::TrilinosWrappers::MPI::Vector vhalf_owned;
  dealii::TrilinosWrappers::MPI::Vector vhalf;      // ghosted for output

  // helper
  dealii::TrilinosWrappers::MPI::Vector force;
  dealii::TrilinosWrappers::MPI::Vector Ku;

  // dof sets
  dealii::IndexSet locally_owned;
  dealii::IndexSet locally_relevant;

  // parameters
  const std::string  mesh_file_name;
  const unsigned int r;
  const double       T;
  const double       dt;
  const double       c2;

  double time = 0.0;

  dealii::ConditionalOStream pcout;

private:
  void assemble_matrices();
  void build_lumped();
  void assemble_force(double t);
  void apply_dirichlet(dealii::TrilinosWrappers::MPI::Vector &vec_u,
                       double t);
  void apply_dirichlet_velocity(dealii::TrilinosWrappers::MPI::Vector &vec_v,
                                double t);

  void initialize_ic(); // set u^0, v^{1/2}
  void do_timestep(unsigned int step);
  void output(unsigned int step) const;
};

#endif
