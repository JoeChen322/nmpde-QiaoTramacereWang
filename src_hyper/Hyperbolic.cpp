#include "Hyperbolic.hpp"
#include <deal.II/base/function_lib.h>
#include <cmath>
#include <unistd.h>

using namespace dealii;

// ================== 默认物理数据 ==================

// f(x,t)
double Hyperbolic::ForcingTerm::value(const Point<dim> &/*p*/,
                                      unsigned int) const
{
  return 0.0; // 默认无体力
}

// g(x,t) Dirichlet
double Hyperbolic::BoundaryValues::value(const Point<dim> &/*p*/,
                                         unsigned int) const
{
  return 0.0; // 默认固定边界 u=0
}

// u0(x)
double Hyperbolic::InitialU::value(const Point<dim> &p,
                                   unsigned int) const
{
  // 给一个高斯包，演示用
  //const double dx = p(0)-0.5;
  //const double dy = p(1)-0.5;
  return std::sin(numbers::PI * p[0]) *
         std::sin(numbers::PI * p[1]);
}

// u1(x) = ∂_t u(x,0)
double Hyperbolic::InitialV::value(const Point<dim> & /*p*/,
                                   unsigned int) const
{
  return 0.0; // 初始速度默认0
}

// =================================================

Hyperbolic::Hyperbolic(const std::string &mesh_file_name_,
                       const unsigned int degree_,
                       const double       T_,
                       const double       dt_,
                       const double       c2_)
  : mesh(
      typename dealii::Triangulation<dim>::MeshSmoothing(
        dealii::Triangulation<dim>::limit_level_difference_at_vertices))
  , dof_handler(mesh)
  , mesh_file_name(mesh_file_name_)
  , r(degree_)
  , T(T_)
  , dt(dt_)
  , c2(c2_)
  , pcout(std::cout, /*always print, single rank*/ true)
{}


// -------------------------------------------------
void Hyperbolic::setup()
{
  // 1. 读网格
  {
    char cwd[512]; getcwd(cwd, sizeof(cwd));
    pcout << "CWD: " << cwd << std::endl;
    pcout << "Opening mesh: " << mesh_file_name << std::endl;

    std::ifstream in(mesh_file_name);
    AssertThrow(in.good(),
                ExcMessage("Cannot open mesh file: " + mesh_file_name));

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh);
    grid_in.read_msh(in);

    pcout << "  n_active_cells = " << mesh.n_active_cells() << std::endl;
  }

  // 2. 选择有限元
  if (mesh.all_reference_cells_are_hyper_cube())
    {
      pcout << "quadrilateral mesh -> FE_Q(" << r << ")\n";
      fe = std::make_unique<FE_Q<dim>>(r);
      quadrature = std::make_unique<QGauss<dim>>(r+1);
    }
  else
    {
      pcout << "simplex mesh -> FE_SimplexP(" << r << ")\n";
      fe = std::make_unique<FE_SimplexP<dim>>(r);
      quadrature = std::make_unique<QGaussSimplex<dim>>(r+1);
    }

  dof_handler.reinit(mesh);
  dof_handler.distribute_dofs(*fe);

  locally_owned    = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant);

  pcout << "  n_dofs = " << dof_handler.n_dofs() << std::endl;

  TrilinosWrappers::SparsityPattern sp(locally_owned, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler, sp);
  sp.compress();

  mass_matrix.reinit(sp);
  stiffness_matrix.reinit(sp);

  u_owned.reinit(locally_owned, MPI_COMM_WORLD);
  vhalf_owned.reinit(locally_owned, MPI_COMM_WORLD);

  u.reinit(locally_owned, locally_relevant, MPI_COMM_WORLD);
  vhalf.reinit(locally_owned, locally_relevant, MPI_COMM_WORLD);

  force.reinit(locally_owned, MPI_COMM_WORLD);
  Ku.reinit(locally_owned, MPI_COMM_WORLD);

  ones.reinit(locally_owned, MPI_COMM_WORLD);
  M_lumped.reinit(locally_owned, MPI_COMM_WORLD);
  Minv_lumped.reinit(locally_owned, MPI_COMM_WORLD);

  assemble_matrices();
  build_lumped();
}

// -------------------------------------------------
void Hyperbolic::assemble_matrices()
{
  pcout << "Assembling M and K ..." << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values |
                          update_gradients |
                          update_JxW_values);

  FullMatrix<double> cell_M(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_K(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_M = 0.0;
      cell_K = 0.0;

      for (unsigned int q=0; q<n_q; ++q)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              cell_M(i,j) += fe_values.shape_value(i,q) *
                             fe_values.shape_value(j,q) *
                             fe_values.JxW(q);

              cell_K(i,j) += c2 *
                             (fe_values.shape_grad(i,q) *
                              fe_values.shape_grad(j,q)) *
                             fe_values.JxW(q);
            }

      cell->get_dof_indices(dof_indices);
      mass_matrix.add(dof_indices, cell_M);
      stiffness_matrix.add(dof_indices, cell_K);
    }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);
}

// -------------------------------------------------
void Hyperbolic::build_lumped()
{
  pcout << "Building lumped mass diag (explicit) ..." << std::endl;

  M_lumped = 0.0;
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values | update_JxW_values);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(dof_indices);

      for (unsigned int q=0; q<n_q; ++q)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            double m_ii = fe_values.shape_value(i,q) *
                          fe_values.shape_value(i,q) *
                          fe_values.JxW(q);
            M_lumped[dof_indices[i]] += m_ii;
          }
    }

  // inverse
  for (unsigned int i=0; i<M_lumped.size(); ++i)
    Minv_lumped[i] = (M_lumped[i]>1e-20 ? 1.0/M_lumped[i] : 0.0);

  pcout << "M_lumped range = [" << *std::min_element(M_lumped.begin(), M_lumped.end())
        << ", " << *std::max_element(M_lumped.begin(), M_lumped.end()) << "]" << std::endl;
}


// -------------------------------------------------
void Hyperbolic::assemble_force(const double t)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe, *quadrature,
                          update_values |
                          update_quadrature_points |
                          update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  force = 0.0;
  forcing_term.set_time(t);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_rhs = 0.0;

      for (unsigned int q=0; q<n_q; ++q)
        {
          const double fq = forcing_term.value(fe_values.quadrature_point(q));
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            cell_rhs(i) += fq *
                           fe_values.shape_value(i,q) *
                           fe_values.JxW(q);
        }

      cell->get_dof_indices(dof_indices);
      force.add(dof_indices, cell_rhs);
    }

  force.compress(VectorOperation::add);
}

// -------------------------------------------------
// 把 u 的Dirichlet自由度强制成 g(x,t)
void Hyperbolic::apply_dirichlet(TrilinosWrappers::MPI::Vector &vec_u,
                                 const double t)
{
  g_func.set_time(t);

  std::map<types::boundary_id,const Function<dim>*> bfuncs;
  // 假设所有边界id都用同一个g
  // 如果网格里不是0..3，按需改
  for (unsigned int bid=0; bid<10; ++bid)
    bfuncs[bid] = &g_func;

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           bfuncs,
                                           boundary_values);

  for (auto &p : boundary_values)
    vec_u[p.first] = p.second;

  vec_u.compress(VectorOperation::insert);
}

// -------------------------------------------------
// 速度边界同理，用 g_t(x,t) ≈ (g(x,t+dt/2)-g(x,t-dt/2))/dt
void Hyperbolic::apply_dirichlet_velocity(TrilinosWrappers::MPI::Vector &vec_v,
                                          const double t)
{
  // 简单差分近似 ∂_t g
  class GtApprox : public Function<dim>
  {
  public:
    GtApprox(const Function<dim> &g_, double t_, double dt_)
      : g(g_), t(t_), dt(dt_) {}
    virtual double value(const Point<dim> &p,
                         unsigned int /*c*/=0) const override
    {
      const_cast<Function<dim>&>(g).set_time(t+0.5*dt);
      const double gp = g.value(p);
      const_cast<Function<dim>&>(g).set_time(t-0.5*dt);
      const double gm = g.value(p);
      // central difference
      return (gp-gm)/dt;
    }
  private:
    const Function<dim> &g;
    double t, dt;
  };

  GtApprox gt(g_func,t,dt);

  std::map<types::boundary_id,const Function<dim>*> bfuncs;
  for (unsigned int bid=0; bid<10; ++bid)
    bfuncs[bid] = &gt;

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           bfuncs,
                                           boundary_values);

  for (auto &p : boundary_values)
    vec_v[p.first] = p.second;

  vec_v.compress(VectorOperation::insert);
}

// -------------------------------------------------
// 初始化 u^0, v^{1/2}
void Hyperbolic::initialize_ic()
{
  time = 0.0;

  // u^0 = u0(x)
  VectorTools::interpolate(dof_handler, u0_func, u_owned);
  u = u_owned;

  // v^0 = u1(x)
  TrilinosWrappers::MPI::Vector v0_owned(locally_owned, MPI_COMM_WORLD);
  VectorTools::interpolate(dof_handler, u1_func, v0_owned);

  // 计算 a^0 = M^{-1}( f^0 - K u^0 )
  assemble_force(time);
  stiffness_matrix.vmult(Ku, u_owned);

  TrilinosWrappers::MPI::Vector a0(locally_owned, MPI_COMM_WORLD);
  for (auto idx : locally_owned)
    {
      const double rhs = force[idx] - Ku[idx];
      a0[idx] = Minv_lumped[idx] * rhs;
    }
  a0.compress(VectorOperation::insert);

  // leapfrog 需要 v^{1/2} = v0 + 0.5*dt*a0
  vhalf_owned.reinit(locally_owned, MPI_COMM_WORLD);
  for (auto idx : locally_owned)
    vhalf_owned[idx] = v0_owned[idx] + 0.5*dt*a0[idx];
  vhalf_owned.compress(VectorOperation::insert);

  // 把Dirichlet条件投影进去
  apply_dirichlet(u_owned, time);
  apply_dirichlet_velocity(vhalf_owned, time+0.5*dt);

  u     = u_owned;
  vhalf = vhalf_owned;

  output(0);
}

// -------------------------------------------------
void Hyperbolic::do_timestep(const unsigned int step)
{
  // 1) 位置步: u^{n+1} = u^n + dt * v^{n+1/2}
  for (auto idx : locally_owned)
    u_owned[idx] += dt * vhalf_owned[idx];

  time += dt;

  // Enforce boundary on displacement at t^{n+1}
  apply_dirichlet(u_owned, time);
  u = u_owned;

  // 2) 组装 f^{n+1} and Ku^{n+1}
  assemble_force(time);
  stiffness_matrix.vmult(Ku, u_owned);

  // 3) 加速度 a^{n+1} = M^{-1}( f^{n+1} - K u^{n+1} )
  for (auto idx : locally_owned)
    {
      const double rhs = force[idx] - Ku[idx];
      const double acc = Minv_lumped[idx] * rhs;

      // 4) 速度半步: v^{n+3/2} = v^{n+1/2} + dt * a^{n+1}
      vhalf_owned[idx] += dt * acc;
    }

  // enforce boundary velocity at t^{n+1+1/2}
  apply_dirichlet_velocity(vhalf_owned, time+0.5*dt);

  vhalf = vhalf_owned;

  output(step);
}

// -------------------------------------------------
void Hyperbolic::output(const unsigned int step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, u,     "u");
  data_out.add_data_vector(dof_handler, vhalf, "v_half");
  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./",
                                      "wave",
                                      step,
                                      MPI_COMM_WORLD,
                                      3);
}

// -------------------------------------------------
void Hyperbolic::solve()
{
  pcout << "=========== Hyperbolic wave solve() ===========" << std::endl;
  initialize_ic();

  unsigned int step = 0;
  while (time < T-1e-14)
    {
      ++step;
      pcout << "step " << step << "  t=" << time+dt << std::endl;
      do_timestep(step);
    }
}
