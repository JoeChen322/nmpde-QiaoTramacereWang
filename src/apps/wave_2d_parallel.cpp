#include "CGWaveSolver2DParallel.hpp"
#include "ProblemBase.hpp"

#include <deal.II/base/utilities.h>

#include <cmath>
#include <iostream>
#include <memory>

using namespace WaveEquation;

/**
 * @brief Simple Gaussian pulse problem for testing parallel solver
 */
class GaussianPulse2D : public ProblemBase<2>
{
public:
    GaussianPulse2D() = default;
    
    double initial_displacement(const dealii::Point<2> &p) const override
    {
        const double x = p[0];
        const double y = p[1];
        const double r2 = x*x + y*y;
        const double sigma = 0.1;
        return std::exp(-r2 / (2.0 * sigma * sigma));
    }
    
    double initial_velocity(const dealii::Point<2> &/*p*/) const override
    {
        return 0.0;
    }
    
    double source_term(const dealii::Point<2> &/*p*/, double /*t*/) const override
    {
        return 0.0;
    }
    
    double boundary_value(const dealii::Point<2> &/*p*/, double /*t*/) const override
    {
        return 0.0;  // Homogeneous Neumann boundary conditions
    }
};

int main(int argc, char *argv[])
{
    // Initialize MPI
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    
    // Check command-line arguments
    bool use_mpi = true;
    if (argc > 1)
    {
        std::string arg(argv[1]);
        if (arg == "--serial" || arg == "-s")
        {
            use_mpi = false;
        }
    }
    
    const unsigned int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    
    try
    {
        // Print mode
        if (mpi_rank == 0)
        {
            std::cout << "========================================" << std::endl;
            std::cout << "2D Wave Solver Parallel Test" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Mode: " << (use_mpi ? "PARALLEL (MPI)" : "SERIAL") << std::endl;
            
            if (use_mpi)
            {
                std::cout << "MPI Processes: " 
                         << dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) 
                         << std::endl;
            }
            
            std::cout << "Usage: " << argv[0] << " [--serial|-s]" << std::endl;
            std::cout << "  Default: parallel mode" << std::endl;
            std::cout << "  --serial: force serial mode" << std::endl;
            std::cout << std::endl;
        }
        
        // Create solver
        CGWaveSolver2DParallel solver(use_mpi);
        
        // Create problem
        auto problem = std::make_unique<GaussianPulse2D>();
        solver.set_problem(std::move(problem));
        
        // Configure solver
        solver.set_mesh_file("../src/core/Meshes/square_level_3.msh");
        solver.set_wave_speed(1.0);
        solver.set_time_step(0.01);
        solver.set_final_time(1.0);
        solver.set_output_interval(10);
        solver.set_theta(0.5);  // Crank-Nicolson
        
        // Run simulation
        solver.setup();
        solver.run();
        
        if (mpi_rank == 0)
        {
            std::cout << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Simulation completed successfully!" << std::endl;
            std::cout << "========================================" << std::endl;
        }
    }
    catch (std::exception &exc)
    {
        if (mpi_rank == 0)
        {
            std::cerr << std::endl << std::endl
                     << "----------------------------------------------------"
                     << std::endl;
            std::cerr << "Exception on processing: " << std::endl
                     << exc.what() << std::endl
                     << "Aborting!" << std::endl
                     << "----------------------------------------------------"
                     << std::endl;
        }
        return 1;
    }
    catch (...)
    {
        if (mpi_rank == 0)
        {
            std::cerr << std::endl << std::endl
                     << "----------------------------------------------------"
                     << std::endl;
            std::cerr << "Unknown exception!" << std::endl
                     << "Aborting!" << std::endl
                     << "----------------------------------------------------"
                     << std::endl;
        }
        return 1;
    }
    
    return 0;
}
