#include "../include/Wave.hpp"
#include "../include/IOUtils.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static void print_usage(std::ostream &os)
{
  os << "Usage:\n"
     << "  SpaceConvergence <mesh_dir> <prefix> <degree> <T> <theta> <dt> <N1> <N2> ...\n\n"
     << "Example:\n"
     << "  mpirun -np 4 ./SpaceConvergence ../meshes mesh-square 1 1.0 0.5 0.01 8 16 32 64 128\n\n"
     << "Defaults (if omitted):\n"
     << "  mesh_dir=../meshes, prefix=mesh-square, degree=1, T=1.0, theta=0.5, dt=0.01, Ns={8,16,32,64,128}\n";
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  // ----------------------------
  // Defaults (run with no args)
  // ----------------------------
  std::string mesh_dir = "../meshes";
  std::string prefix = "mesh-square";
  unsigned int degree = 1;
  double T = 1.0;
  double theta = 0.5;
  double dt = 0.01; // choose moderate dt
  std::vector<int> Ns = {8, 16, 32, 64, 128};

  // ----------------------------
  // CLI parsing (same format as before)
  // SpaceConvergence <mesh_dir> <prefix> <degree> <T> <theta> <dt> <N1> <N2> ...
  // ----------------------------
  if (argc > 1)
  {
    const std::string arg1 = argv[1];
    if (arg1 == "-h" || arg1 == "--help")
    {
      if (mpi_rank == 0)
        print_usage(std::cout);
      return 0;
    }
  }

  // Override defaults from CLI
  if (argc > 1)
    mesh_dir = argv[1];
  if (argc > 2)
    prefix = argv[2];
  if (argc > 3)
    degree = static_cast<unsigned int>(std::stoi(argv[3]));
  if (argc > 4)
    T = std::stod(argv[4]);
  if (argc > 5)
    theta = std::stod(argv[5]);
  if (argc > 6)
    dt = std::stod(argv[6]);

  // If the user provides Ns explicitly, override defaults.
  if (argc > 7)
  {
    Ns.clear();
    for (int i = 7; i < argc; ++i)
      Ns.push_back(std::stoi(argv[i]));
  }

  if (Ns.empty())
  {
    if (mpi_rank == 0)
      std::cerr << "Error: Ns list is empty.\n";
    return 1;
  }

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
    problem.set_output_interval(0);
    problem.set_output_directory("./");

    problem.setup();

    // Record mesh/dof info before solve
    h_min[k] = problem.get_h_min();
    cells[k] = problem.n_cells();
    dofs[k] = problem.n_dofs();

    // Nominal mesh size for this mesh family:
    // Using 1/N is robust for rate computations (constant factors cancel).
    h_nom[k] = 1.0 / static_cast<double>(Ns[k]);

    problem.solve();

    // Errors at final time
    err_u[k] = problem.compute_L2_error_u(T);
    err_v[k] = problem.compute_L2_error_v(T);

    if (k > 0)
    {
      rate_u[k] = io_utils::safe_rate(err_u[k - 1], err_u[k], h_nom[k - 1], h_nom[k]);
      rate_v[k] = io_utils::safe_rate(err_v[k - 1], err_v[k], h_nom[k - 1], h_nom[k]);
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
  const std::string out_base = "../results";
  const std::string out_dir = out_base + "/convergence";

  // All ranks must participate (MPI-safe)
  io_utils::ensure_directory_exists(out_base, mpi_rank);
  io_utils::ensure_directory_exists(out_dir, mpi_rank);

  if (mpi_rank == 0)
  {

    std::ofstream csv(out_dir + "/space_convergence.csv");

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

    std::cout << "\nWrote: " << out_dir + "/space_convergence.csv\n";
  }

  return 0;
}
