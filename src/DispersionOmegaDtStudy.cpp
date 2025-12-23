#include "../include/Wave.hpp"
#include "../include/IOUtils.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  // CLI:
  // DispersionOmegaDtStudy [mesh] [degree] [T] [stride]
  std::string mesh_file = "../meshes/mesh-square-64.msh";
  unsigned int degree = 1;
  double T = 10.0;         // must be divisible by dt list
  unsigned int stride = 10; // output stride for modal log

  if (argc > 1)
    mesh_file = argv[1];
  if (argc > 2)
    degree = static_cast<unsigned int>(std::stoi(argv[2]));
  if (argc > 3)
    T = std::stod(argv[3]);
  if (argc > 4)
    stride = static_cast<unsigned int>(std::stoi(argv[4]));

  // Dispersion-only strategy: theta fixed to 0.5
  const double theta = 0.5;

  // Choose dt values
  const std::vector<double> dts = {0.2, 0.1, 0.05, 0.025};

  // A few modes
  const std::vector<std::pair<unsigned int, unsigned int>> modes = {
      {1, 1},
      {2, 2},
      {4, 4}};

  for (double dt : dts)
    if (!io_utils::divides_T(T, dt))
    {
      if (mpi_rank == 0)
        std::cerr << "Error: dt=" << dt << " does not divide T=" << T << "\n";
      return 1;
    }

  const std::string base_dir = "../results/dispersion";
  const std::string modal_dir = base_dir + "/modal";
  io_utils::ensure_directory_exists("../results", mpi_rank);
  io_utils::ensure_directory_exists(base_dir, mpi_rank);
  io_utils::ensure_directory_exists(modal_dir, mpi_rank);

  if (mpi_rank == 0)
  {
    std::cout << "Time dispersion study: omega_num/omega_semi vs (omega_semi*dt) (theta=0.5)\n"
              << "  mesh   = " << mesh_file << "\n"
              << "  degree = " << degree << "\n"
              << "  T      = " << T << "\n"
              << "  stride = " << stride << "\n"
              << "  outdir = " << base_dir << "\n\n";
  }

  std::ofstream summary;
  if (mpi_rank == 0)
  {
    summary.open(base_dir + "/dispersion_summary.csv");
    summary << "mesh,degree,m,n,theta,dt,T,"
               "omega_exact,omega_semi,omega_semi_dt,omega_num,"
               "ratio_num_semi,ratio_semi_exact,ratio_num_exact,"
               "phase_drift_T,ratio_pred_theta05,modal_csv\n";
    summary << std::setprecision(16);
  }

  for (const auto &mn : modes)
  {
    const unsigned int m = mn.first;
    const unsigned int n = mn.second;

    for (double dt : dts)
    {
      if (mpi_rank == 0)
      {
        std::cout << "-----------------------------------------------\n"
                  << "Case: (m,n)=(" << m << "," << n << "), dt=" << dt << "\n";
      }

      Wave problem(mesh_file, degree, T, dt, theta);

      problem.set_verbose(false);
      problem.set_output_interval(0);
      problem.set_output_directory(".");
      problem.set_mode(m, n);

      const std::string modal_csv_rel =
          "modal_m" + std::to_string(m) +
          "_n" + std::to_string(n) +
          "_dt" + io_utils::tag_double(dt) +
          "_T" + io_utils::tag_double(T) + ".csv";

      const std::string modal_csv = modal_dir + "/" + modal_csv_rel;

      problem.enable_modal_log(modal_csv, stride, /*include_exact=*/true);

      problem.setup();
      problem.solve();

      const double omega_exact = problem.get_omega_exact();
      const double omega_semi = problem.get_omega_semi();
      const double omega_num = problem.get_omega_num();
      const double driftT = problem.get_phase_drift_T();

      const double x = omega_semi * dt;

      const double ratio_num_semi = (omega_semi > 0.0) ? (omega_num / omega_semi) : 0.0;
      const double ratio_semi_exact = (omega_exact > 0.0) ? (omega_semi / omega_exact) : 0.0;
      const double ratio_num_exact = (omega_exact > 0.0) ? (omega_num / omega_exact) : 0.0;

      // Theory overlay for theta=0.5: omega_dt/omega_h = (2/x) atan(x/2)
      double ratio_pred = 1.0;
      if (x > 1e-14)
        ratio_pred = (2.0 / x) * std::atan(x / 2.0);

      if (mpi_rank == 0)
      {
        std::cout << std::setprecision(16)
                  << "  omega_exact=" << omega_exact
                  << "  omega_semi=" << omega_semi
                  << "  omega_num=" << omega_num << "\n"
                  << "  x=omega_semi*dt=" << x
                  << "  omega_num/omega_semi=" << ratio_num_semi
                  << "  pred(theta=0.5)=" << ratio_pred << "\n"
                  << "  phase_drift(T)=" << driftT << "\n"
                  << "  wrote: " << modal_csv << "\n";

        summary << "\"" << mesh_file << "\"" << ","
                << degree << ","
                << m << "," << n << ","
                << theta << ","
                << dt << ","
                << T << ","
                << omega_exact << ","
                << omega_semi << ","
                << x << ","
                << omega_num << ","
                << ratio_num_semi << ","
                << ratio_semi_exact << ","
                << ratio_num_exact << ","
                << driftT << ","
                << ratio_pred << ","
                << "\"" << modal_csv << "\"\n";
      }
    }
  }

  if (mpi_rank == 0)
  {
    summary.close();
    std::cout << "\nWrote: " << (base_dir + "/dispersion_summary.csv") << "\n";
  }

  return 0;
}
