#include "CGWaveSolver.hpp"
#include "ProblemBase.hpp"

#include <iostream>
#include <memory>

// Include the Gaussian pulse problem implementation
#include "../../src/core/Problems/GaussianPulse.cpp"

int main()
{
    try
    {
        std::cout << "Testing 1D Wave Equation Solver" << std::endl;
        std::cout << "===============================" << std::endl;
        
        // Create 1D solver
        WaveEquation::CGWaveSolver<1> solver;
        
        // Create and set problem
        auto problem = std::make_unique<WaveEquation::GaussianPulse<1>>();
        solver.set_problem(std::move(problem));
        
        // Run simulation
        solver.setup();
        solver.run();
        
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