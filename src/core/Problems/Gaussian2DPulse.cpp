#include "ProblemBase.hpp"
#include <cmath>

namespace WaveEquation
{
    template <int dim>
    class Gaussian2DPulse : public ProblemBase<dim>
    {
    public:
        virtual double initial_displacement(const dealii::Point<dim> &p) const override
        {
            // 2D radially symmetric Gaussian pulse centered at origin
            const double r_squared = p.square();
            const double sigma = 0.2;  // Width of the pulse
            return std::exp(-r_squared / (2.0 * sigma * sigma));
        }
        
        virtual double initial_velocity(const dealii::Point<dim> &/*p*/) const override
        {
            // Zero initial velocity for symmetric expansion
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
