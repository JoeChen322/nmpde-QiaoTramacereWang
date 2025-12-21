#include "Wave.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static double safe_rate(const double e_coarse,
                        const double e_fine,
                        const double h_coarse,
                        const double h_fine)
{
  // p = log(e_c/e_f) / log(h_c/h_f)
  if (e_coarse <= 0.0 || e_fine <= 0.0 || h_coarse <= 0.0 || h_fine <= 0.0)
    return 0.0;

  const double denom = std::log(h_coarse / h_fine);
  if (std::abs(denom) < 1e-30)
    return 0.0;

  return std::log(e_coarse / e_fine) / denom;
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  // Usage:
  //   SpaceConvergence <mesh_dir> <prefix> <degree> <T> <theta> <dt> <N1> <N2> ...
  //
  // Example:
  //   mpirun -np 4 ./SpaceConvergence ../mesh mesh-square 1 1.0 0.5 0.001 20 40 80 160
  //
  if (argc < 8)
  {
    if (mpi_rank == 0)
    {
      std::cerr << "Usage:\n"
                << "  SpaceConvergence <mesh_dir> <prefix> <degree> <T> <theta> <dt> <N1> <N2> ...\n\n"
                << "Example:\n"
                << "  mpirun -np 4 ./SpaceConvergence ../mesh mesh-square 1 1.0 0.5 0.001 20 40 80 160\n";
    }
    return 1;
  }

  const std::string mesh_dir = argv[1];
  const std::string prefix = argv[2];
  const unsigned int degree = static_cast<unsigned int>(std::stoi(argv[3]));
  const double T = std::stod(argv[4]);
  const double theta = std::stod(argv[5]);
  const double dt = std::stod(argv[6]);

  std::vector<int> Ns;
  for (int i = 7; i < argc; ++i)
    Ns.push_back(std::stoi(argv[i]));

  if (mpi_rank == 0)
  {
    std::cout << "Space convergence test\n"
              << "  mesh_dir = " << mesh_dir << "\n"
              << "  prefix   = " << prefix << "\n"
              << "  degree   = " << degree << "\n"
              << "  T        = " << T << "\n"
              << "  theta    = " << theta << "\n"
              << "  dt       = " << dt << "  (choose small so time error is negligible)\n"
              << "  Ns       = ";
    for (auto n : Ns)
      std::cout << n << " ";
    std::cout << "\n\n";

    std::cout << "NOTE: Rates are computed using h_nom = 1/N (robust for mesh-square-N family).\n"
              << "      We still record h_min for diagnostics only.\n\n";
  }

  const unsigned int L = Ns.size();
  std::vector<std::string> mesh_files(L);

  for (unsigned int k = 0; k < L; ++k)
    mesh_files[k] = mesh_dir + "/" + prefix + "-" + std::to_string(Ns[k]) + ".msh";

  // We store both:
  //  - h_min: diagnostic from mesh geometry (can be noisy on unstructured meshes)
  //  - h_nom: nominal global size used for rates, h_nom = 1/N
  std::vector<double> h_min(L, 0.0), h_nom(L, 0.0);
  std::vector<double> err_u(L, 0.0), err_v(L, 0.0);
  std::vector<double> rate_u(L, 0.0), rate_v(L, 0.0);
  std::vector<unsigned long long> cells(L, 0ULL);
  std::vector<types::global_dof_index> dofs(L, 0);

  for (unsigned int k = 0; k < L; ++k)
  {
    if (mpi_rank == 0)
      std::cout << "-----------------------------------------------\n";

    const std::string &mesh_file = mesh_files[k];

    Wave problem(mesh_file, degree, T, dt, theta);
    problem.set_output_interval(0); // disable VTU output for convergence runs
    problem.set_output_directory("./");

    problem.setup();

    // Record mesh/dof info before solve
    h_min[k] = problem.get_h_min();
    cells[k] = problem.n_cells();
    dofs[k] = problem.n_dofs();

    // Nominal mesh size for this mesh family:
    // If your domain length is 2 instead of 1, the true h is (2/N),
    // but the constant cancels in rate computations. Using 1/N is fine.
    h_nom[k] = 1.0 / static_cast<double>(Ns[k]);

    problem.solve();

    // Errors at final time
    err_u[k] = problem.compute_L2_error_u(T);
    err_v[k] = problem.compute_L2_error_v(T);

    if (k > 0)
    {
      // IMPORTANT: use h_nom for rates
      rate_u[k] = safe_rate(err_u[k - 1], err_u[k], h_nom[k - 1], h_nom[k]);
      rate_v[k] = safe_rate(err_v[k - 1], err_v[k], h_nom[k - 1], h_nom[k]);
    }

    if (mpi_rank == 0)
    {
      std::cout << std::setprecision(16);
      std::cout << "mesh = " << mesh_file << "\n"
                << "  N=" << Ns[k]
                << " | cells=" << cells[k]
                << " | dofs=" << dofs[k]
                << " | h_min=" << h_min[k]
                << " | h_nom=1/N=" << h_nom[k] << "\n"
                << "  ||u-ue||_L2 = " << err_u[k]
                << " | ||v-ve||_L2 = " << err_v[k] << "\n";
      if (k > 0)
        std::cout << "  rates (using h_nom): p_u = " << rate_u[k]
                  << ", p_v = " << rate_v[k] << "\n";
    }
  }

  // Write CSV on rank 0
  if (mpi_rank == 0)
  {
    std::ofstream csv("space_convergence.csv");
    csv << "k,N,mesh_file,cells,dofs,h_min,h_nom,dt,T,theta,err_u,rate_u,err_v,rate_v\n";
    csv << std::setprecision(16);

    for (unsigned int k = 0; k < L; ++k)
    {
      csv << k << ","
          << Ns[k] << ","
          << "\"" << mesh_files[k] << "\"" << ","
          << cells[k] << ","
          << dofs[k] << ","
          << h_min[k] << ","
          << h_nom[k] << ","
          << dt << ","
          << T << ","
          << theta << ","
          << err_u[k] << ","
          << rate_u[k] << ","
          << err_v[k] << ","
          << rate_v[k] << "\n";
    }

    std::cout << "\nWrote: space_convergence.csv\n";
  }

  return 0;
}
