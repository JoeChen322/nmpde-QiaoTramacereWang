#include "CGWaveSolver2D.hpp"
#include "ProblemBase.hpp"

#include <iostream>
#include <memory>

// Include the 2D Gaussian pulse problem
#include "../../src/core/Problems/Gaussian2DPulse.cpp"

int main()
{
    try
    {
        std::cout << "Testing 2D Wave Equation Solver (Triangular Elements)" << std::endl;
        std::cout << "======================================================" << std::endl;
        std::cout << std::endl;
        
        // Create 2D solver
        WaveEquation::CGWaveSolver2D solver;
        
        // Create and set problem
        auto problem = std::make_unique<WaveEquation::Gaussian2DPulse<2>>();
        solver.set_problem(std::move(problem));
        
        // Configure solver - use one of the generated meshes
        // Path relative to build directory
        solver.set_mesh_file("../src/core/Meshes/square_level_4.msh");
        solver.set_final_time(6.0);
        solver.set_time_step(0.01);
        solver.set_output_interval(1);  // Output every step like 1D
        solver.set_wave_speed(1.0);
        
        // Run simulation
        solver.setup();
        solver.run();
        
        std::cout << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    
    return 0;
}
