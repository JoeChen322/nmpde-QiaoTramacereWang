#pragma once

#include <deal.II/base/function.h>

namespace WaveEquation
{
    template <int dim>
    class ProblemBase : public dealii::Function<dim>
    {
    public:
        virtual ~ProblemBase() = default;
        
        // Initial conditions
        virtual double initial_displacement(const dealii::Point<dim> &p) const = 0;
        virtual double initial_velocity(const dealii::Point<dim> &p) const = 0;
        
        // Boundary conditions
        virtual double boundary_value(const dealii::Point<dim> &p, double t) const = 0;
        
        // Source term (right-hand side)
        virtual double source_term(const dealii::Point<dim> &p, double t) const = 0;
    };

} // namespace WaveEquation