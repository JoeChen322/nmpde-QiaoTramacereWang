#include "../include/Wave.hpp"

// Main function.
int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name = "../meshes/mesh-square-40.msh";
  const unsigned int degree         = 1;

  const double T      = 1.0;
  const double deltat = 0.05;
  const double theta  = 0.5;

  Wave problem(mesh_file_name, degree, T, deltat, theta);

  // Optional controls (default: write every step to "./")
  problem.set_output_interval(1);
  problem.set_output_directory("./");

  problem.setup();
  problem.solve();

  return 0;
}
