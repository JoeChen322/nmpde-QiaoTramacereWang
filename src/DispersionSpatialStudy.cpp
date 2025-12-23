#include "../include/Wave.hpp"
#include "../include/IOUtils.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;

static void print_usage(std::ostream &os)
{
  os << "Usage:\n"
     << "  DispersionSpatialStudy <mesh_dir> <prefix> <degree> <T> <dt> <m> <n> <N1> <N2> ...\n\n"
     << "Example:\n"
     << "  mpirun -np 4 ./DispersionSpatialStudy ../meshes mesh-square 1 10.0 0.001 1 1 8 16 32 64 128\n\n"
     << "Defaults (if omitted):\n"
     << "  mesh_dir=../meshes, prefix=mesh-square, degree=1, T=10.0, dt=0.001, m=1, n=1, Ns={8,16,32,64,128}\n";
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  // Defaults
  std::string mesh_dir = "../meshes";
  std::string prefix = "mesh-square";
  unsigned int degree = 1;
  double T = 10.0;
  double dt = 0.01;  // set default dt a little moderate for running
  unsigned int m = 2; // More apprent
  unsigned int n = 2;
  std::vector<int> Ns = {4, 8, 16, 32, 64};

  const double theta = 0.5;      // non-dissipative for oscillatory modes
  const unsigned int stride = 1; // modal sampling stride

  // CLI parsing
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

  if (argc > 1)
    mesh_dir = argv[1];
  if (argc > 2)
    prefix = argv[2];
  if (argc > 3)
    degree = static_cast<unsigned int>(std::stoi(argv[3]));
  if (argc > 4)
    T = std::stod(argv[4]);
  if (argc > 5)
    dt = std::stod(argv[5]);
  if (argc > 6)
    m = static_cast<unsigned int>(std::stoi(argv[6]));
  if (argc > 7)
    n = static_cast<unsigned int>(std::stoi(argv[7]));

  if (argc > 8)
  {
    Ns.clear();
    for (int i = 8; i < argc; ++i)
      Ns.push_back(std::stoi(argv[i]));
  }

  if (Ns.empty())
  {
    if (mpi_rank == 0)
      std::cerr << "Error: Ns list is empty.\n";
    return 1;
  }

  if (!io_utils::divides_T(T, dt))
  {
    if (mpi_rank == 0)
      std::cerr << "Error: dt does not divide T.\n";
    return 1;
  }

  if (mpi_rank == 0)
  {
    std::cout << "Spatial dispersion study (theta=0.5, tiny dt)\n"
              << "  mesh_dir = " << mesh_dir << "\n"
              << "  prefix   = " << prefix << "\n"
              << "  degree   = " << degree << "\n"
              << "  mode (m,n)=(" << m << "," << n << ")\n"
              << "  dt=" << dt << ", T=" << T << "\n"
              << "  Ns       = ";
    for (auto Nval : Ns)
      std::cout << Nval << " ";
    std::cout << "\n\n";
  }

  // Output directory (MPI-safe)
  const std::string out_base = "../results";
  const std::string out_dir = out_base + "/dispersion";
  const std::string modal_dir = out_dir + "/modal";

  io_utils::ensure_directory_exists(out_base, mpi_rank);
  io_utils::ensure_directory_exists(out_dir, mpi_rank);
  io_utils::ensure_directory_exists(modal_dir, mpi_rank);

  std::ofstream csv;
  if (mpi_rank == 0)
  {
    const std::string csv_path = out_dir + "/dispersion_spatial.csv";
    csv.open(csv_path);

    csv << "k,N,mesh_file,h_min,h_nom,cells,dofs,dt,T,theta,"
           "omega_exact,omega_semi,omega_num,omega_semi_dt,"
           "ratio_semi_exact,ratio_num_semi,modal_csv\n";
    csv << std::setprecision(16);
  }

  for (unsigned int k = 0; k < Ns.size(); ++k)
  {
    const std::string mesh_file =
        mesh_dir + "/" + prefix + "-" + std::to_string(Ns[k]) + ".msh";

    Wave problem(mesh_file, degree, T, dt, theta);
    problem.set_verbose(false);
    problem.set_output_interval(0);
    problem.set_output_directory(".");
    problem.set_mode(m, n);

    // Enable modal sampling so omega_semi and omega_num are computed.
    const std::string modal_csv =
        modal_dir + "/modal_spatial_m" + std::to_string(m) +
        "_n" + std::to_string(n) +
        "_N" + std::to_string(Ns[k]) +
        "_dt" + io_utils::tag_double(dt) +
        "_T" + io_utils::tag_double(T) + ".csv";

    problem.enable_modal_log(modal_csv, stride, /*include_exact=*/false);

    problem.setup();

    const double hmin = problem.get_h_min();
    const double h_nom = 1.0 / static_cast<double>(Ns[k]); // robust for mesh-square-N family
    const auto cells = problem.n_cells();
    const auto dofs = problem.n_dofs();

    problem.solve();

    const double omega_exact = problem.get_omega_exact();
    const double omega_semi = problem.get_omega_semi();
    const double omega_num = problem.get_omega_num();

    const double x = omega_semi * dt;

    const double ratio_semi = (omega_exact > 0.0) ? (omega_semi / omega_exact) : 0.0;
    const double ratio_num_semi = (omega_semi > 0.0) ? (omega_num / omega_semi) : 0.0;

    if (mpi_rank == 0)
    {
      std::cout << "N=" << Ns[k]
                << "  h_nom=" << h_nom
                << "  omega_semi/omega=" << ratio_semi
                << "  omega_num/omega_semi=" << ratio_num_semi
                << "  x=omega_semi*dt=" << x
                << "  (modal: " << modal_csv << ")\n";

      csv << k << "," << Ns[k] << ","
          << "\"" << mesh_file << "\"" << ","
          << hmin << ","
          << h_nom << ","
          << cells << ","
          << dofs << ","
          << dt << ","
          << T << ","
          << theta << ","
          << omega_exact << ","
          << omega_semi << ","
          << omega_num << ","
          << x << ","
          << ratio_semi << ","
          << ratio_num_semi << ","
          << "\"" << modal_csv << "\"\n";
    }
  }

  if (mpi_rank == 0)
  {
    csv.close();
    std::cout << "\nWrote: " << out_dir + "/dispersion_spatial.csv\n";
  }

  return 0;
}
