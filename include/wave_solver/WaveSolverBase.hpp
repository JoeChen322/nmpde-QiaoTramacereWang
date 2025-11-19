#pragma once

#include <iostream>
#include <memory>

namespace WaveEquation
{
    /**
     * @brief Base class for wave equation solvers
     * 
     * Defines the interface that all wave solvers must implement.
     * This enables polymorphic behavior and testing through base pointers.
     */
    template <int dim>
    class WaveSolverBase
    {
    public:
        virtual ~WaveSolverBase() = default;

        // Main interface
        virtual void setup() = 0;
        virtual void run() = 0;
        
        // Analysis
        virtual double compute_energy() const = 0;
        
        // Output
        virtual void output_results(unsigned int step, double time) const = 0;

    protected:
        // Output stream for logging (accessible to derived classes)
        std::ostream &pcout = std::cout;
    };

} // namespace WaveEquation