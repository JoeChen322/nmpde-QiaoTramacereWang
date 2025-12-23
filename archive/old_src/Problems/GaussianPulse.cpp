#include "ProblemBase.hpp"

namespace WaveEquation
{
    template <int dim>
    class GaussianPulse : public ProblemBase<dim>
    {
    public:
        virtual double initial_displacement(const dealii::Point<dim> &p) const override
        {
            // d'Alembert solution for symmetric splitting:
            // u(x,t) = 1/2[g(x-ct) + g(x+ct)]
            // This requires g(x,0) = g(x) and h(x,0) = 0
            // The Gaussian will split into two half-amplitude waves traveling in opposite directions
            const double x = p[0];
            const double sigma = 0.1;
            return std::exp(-(x * x) / (2.0 * sigma * sigma));
        }
        
        virtual double initial_velocity(const dealii::Point<dim> &/*p*/) const override
        {
            // Zero initial velocity for symmetric splitting
            // With u(x,0) = g(x) and v(x,0) = 0, d'Alembert gives:
            // u(x,t) = 1/2[g(x-ct) + g(x+ct)]
            // This creates two waves of equal amplitude moving in opposite directions
            return 0.0;
        }
        
        virtual double boundary_value(const dealii::Point<dim> &/*p*/, double /*t*/) const override
        {
            // Homogeneous Dirichlet boundary conditions
            return 0.0;
        }
        
        virtual double source_term(const dealii::Point<dim> &/*p*/, double /*t*/) const override
        {
            // No external forcing
            return 0.0;
        }
        
        virtual double value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
        {
            // For dealii::Function interface - used for boundary conditions
            return boundary_value(p, this->get_time());
        }
    };

} // namespace WaveEquation