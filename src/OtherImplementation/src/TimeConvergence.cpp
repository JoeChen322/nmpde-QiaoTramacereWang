#include "Wave.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static bool divides_T(const double T, const double dt, const double tol = 1e-12)
{
  const double n = T / dt;
  return std::abs(n - std::round(n)) < tol;
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  // Defaults (override via CLI if you want)
  std::string  mesh_file = "../mesh/mesh-square-40.msh";
  unsigned int degree    = 1;
  double       T         = 1.0;
  double       theta     = 0.5;
  double       dt0       = 0.1;   // coarsest dt
  unsigned int levels    = 5;     // number of halvings

  // CLI: TimeConvergence [mesh] [degree] [T] [theta] [dt0] [levels]
  if (argc > 1) mesh_file = argv[1];
  if (argc > 2) degree    = static_cast<unsigned int>(std::stoi(argv[2]));
  if (argc > 3) T         = std::stod(argv[3]);
  if (argc > 4) theta     = std::stod(argv[4]);
  if (argc > 5) dt0       = std::stod(argv[5]);
  if (argc > 6) levels    = static_cast<unsigned int>(std::stoi(argv[6]));

  // Build dt list: dt_k = dt0 / 2^k
  std::vector<double> dts;
  dts.reserve(levels);
  for (unsigned int k = 0; k < levels; ++k)
    dts.push_back(dt0 / std::pow(2.0, static_cast<double>(k)));

  // Safety: require dt divides T for clean final-time comparison (since solver uses fixed dt).
  for (double dt : dts)
    {
      if (!divides_T(T, dt))
        {
          if (mpi_rank == 0)
            {
              std::cerr << "Error: dt = " << dt << " does not divide T = " << T
                        << " (choose dt0 and levels so that all dt_k divide T).\n";
            }
          return 1;
        }
    }

  std::vector<double> err_u(levels, 0.0), err_v(levels, 0.0);
  std::vector<double> rate_u(levels, 0.0), rate_v(levels, 0.0);

  if (mpi_rank == 0)
    {
      std::cout << "Time convergence test (theta-method)\n";
      std::cout << "  mesh  = " << mesh_file << "\n";
      std::cout << "  degree= " << degree << "\n";
      std::cout << "  T     = " << T << "\n";
      std::cout << "  theta = " << theta << "\n";
      std::cout << "  dt0   = " << dt0 << ", levels = " << levels << "\n\n";
    }

  for (unsigned int k = 0; k < levels; ++k)
    {
      const double dt = dts[k];

      if (mpi_rank == 0)
        std::cout << "-----------------------------------------------\n";

      Wave problem(mesh_file, degree, T, dt, theta);
      problem.set_output_interval(0);     // disable VTU output for convergence runs
      problem.set_output_directory("./"); // no directory creation needed

      problem.setup();
      problem.solve();

      err_u[k] = problem.compute_L2_error_u(T);
      err_v[k] = problem.compute_L2_error_v(T);

      if (k > 0)
        {
          // dt halves each level => rate = log(e_k-1 / e_k) / log(2)
          rate_u[k] = std::log(err_u[k - 1] / err_u[k]) / std::log(2.0);
          rate_v[k] = std::log(err_v[k - 1] / err_v[k]) / std::log(2.0);
        }

      if (mpi_rank == 0)
        {
          std::cout << std::setprecision(16);
          std::cout << "dt = " << dt
                    << " | ||u-ue||_L2 = " << err_u[k]
                    << " | ||v-ve||_L2 = " << err_v[k] << "\n";
          if (k > 0)
            std::cout << "        rates: p_u = " << rate_u[k]
                      << ", p_v = " << rate_v[k] << "\n";
        }
    }

  // Write CSV (rank 0)
  if (mpi_rank == 0)
    {
      std::ofstream csv("time_convergence.csv");
      csv << "level,dt,err_u,rate_u,err_v,rate_v\n";
      csv << std::setprecision(16);
      for (unsigned int k = 0; k < levels; ++k)
        csv << k << "," << dts[k] << "," << err_u[k] << "," << rate_u[k] << ","
            << err_v[k] << "," << rate_v[k] << "\n";

      std::cout << "\nWrote: time_convergence.csv\n";
    }

  return 0;
}
