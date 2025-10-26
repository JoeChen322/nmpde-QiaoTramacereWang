#include "Hyperbolic.hpp"
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

using namespace dealii;   // 👈 这一行很关键！

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file = "../mesh/mesh-square-2.msh";
  const unsigned int degree    = 2;
  const double       T         = 1.0;
  const double       dt        = 0.002;
  const double       c2        = 1.0;

  Hyperbolic problem(mesh_file, degree, T, dt, c2);
  problem.setup();
  problem.solve();

  return 0;
}
