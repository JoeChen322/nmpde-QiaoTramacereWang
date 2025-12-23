#include "ProblemBase.hpp"
#include <cmath>

namespace WaveEquation
{
    /**
     * Method of Manufactured Solutions for 1D Wave Equation with Neumann BCs
     * 
     * Wave equation: u_tt = c² u_xx + f(x,t)
     * Domain: [-1, 1]
     * Boundary conditions: u_x(-1,t) = 0, u_x(1,t) = 0 (Neumann/reflecting)
     * 
     * We choose a manufactured solution that satisfies Neumann boundary conditions:
     * u_exact(x,t) = cos(π*x) * cos(ω*t)
     * 
     * where ω = c*π for the domain [-1, 1]
     * 
     * Verification of Neumann BCs:
     * u_x = -π * sin(π*x) * cos(ω*t)
     * u_x(-1,t) = -π * sin(-π) * cos(ω*t) = 0 ✓
     * u_x(+1,t) = -π * sin(π) * cos(ω*t) = 0 ✓
     * 
     * Derivatives:
     * u_tt = -ω² * cos(π*x) * cos(ω*t)
     * u_xx = -π² * cos(π*x) * cos(ω*t)
     * 
     * Therefore, the source term needed is:
     * f(x,t) = u_tt - c² * u_xx
     *        = -ω² * cos(π*x) * cos(ω*t) + c² * π² * cos(π*x) * cos(ω*t)
     *        = cos(π*x) * cos(ω*t) * (c² * π² - ω²)
     * 
     * With ω = c*π, this simplifies to:
     * f(x,t) = 0
     * 
     * So our manufactured solution is actually an exact solution of the homogeneous wave equation!
     * This is perfect for testing with Neumann (reflecting) boundary conditions.
     */
    template <int dim>
    class ManufacturedSolution : public ProblemBase<dim>
    {
    public:
        ManufacturedSolution(double wave_speed = 1.0)
            : wave_speed_(wave_speed)
            , omega_(wave_speed * M_PI)
        {
        }
        
        // Exact solution: u(x,t) = cos(π*x) * cos(ω*t)
        double exact_solution(const dealii::Point<dim> &p, double t) const
        {
            const double x = p[0];
            return std::cos(M_PI * x) * std::cos(omega_ * t);
        }
        
        // Time derivative: u_t(x,t) = -ω * cos(π*x) * sin(ω*t)
        double exact_velocity(const dealii::Point<dim> &p, double t) const
        {
            const double x = p[0];
            return -omega_ * std::cos(M_PI * x) * std::sin(omega_ * t);
        }
        
        // Initial displacement: u(x,0) = cos(π*x)
        virtual double initial_displacement(const dealii::Point<dim> &p) const override
        {
            return exact_solution(p, 0.0);
        }
        
        // Initial velocity: u_t(x,0) = 0
        virtual double initial_velocity(const dealii::Point<dim> &p) const override
        {
            return exact_velocity(p, 0.0);
        }
        
        // Boundary conditions: u_x(±1,t) = 0 (Neumann - automatically satisfied)
        virtual double boundary_value(const dealii::Point<dim> &p, double t) const override
        {
            return exact_solution(p, t);
        }
        
        // Source term: f(x,t) = 0 (since our manufactured solution is exact)
        virtual double source_term(const dealii::Point<dim> &p, double t) const override
        {
            (void)p;
            (void)t;
            return 0.0;
        }
        
        // For dealii::Function interface
        virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
        {
            (void)component;
            return boundary_value(p, this->get_time());
        }
        
    private:
        double wave_speed_;
        double omega_;  // Angular frequency = c*π
    };
    
    /**
     * Alternative manufactured solution with non-zero source term
     * 
     * u_exact(x,t) = cos(π*x) * sin(π*t)
     * 
     * Verification of Neumann BCs:
     * u_x = -π * sin(π*x) * sin(π*t)
     * u_x(-1,t) = -π * sin(-π) * sin(π*t) = 0 ✓
     * u_x(+1,t) = -π * sin(π) * sin(π*t) = 0 ✓
     * 
     * Derivatives:
     * u_tt = -π² * cos(π*x) * sin(π*t)
     * u_xx = -π² * cos(π*x) * sin(π*t)
     * 
     * Source term:
     * f(x,t) = u_tt - c² * u_xx
     *        = -π² * cos(π*x) * sin(π*t) + c² * π² * cos(π*x) * sin(π*t)
     *        = π² * (c² - 1) * cos(π*x) * sin(π*t)
     */
    template <int dim>
    class ManufacturedSolutionWithSource : public ProblemBase<dim>
    {
    public:
        ManufacturedSolutionWithSource(double wave_speed = 1.0)
            : wave_speed_(wave_speed)
        {
        }
        
        // Exact solution: u(x,t) = cos(π*x) * sin(π*t)
        double exact_solution(const dealii::Point<dim> &p, double t) const
        {
            const double x = p[0];
            return std::cos(M_PI * x) * std::sin(M_PI * t);
        }
        
        // Time derivative: u_t(x,t) = π * cos(π*x) * cos(π*t)
        double exact_velocity(const dealii::Point<dim> &p, double t) const
        {
            const double x = p[0];
            return M_PI * std::cos(M_PI * x) * std::cos(M_PI * t);
        }
        
        virtual double initial_displacement(const dealii::Point<dim> &p) const override
        {
            return exact_solution(p, 0.0);
        }
        
        virtual double initial_velocity(const dealii::Point<dim> &p) const override
        {
            return exact_velocity(p, 0.0);
        }
        
        virtual double boundary_value(const dealii::Point<dim> &p, double t) const override
        {
            return exact_solution(p, t);
        }
        
        // Source term: f(x,t) = π² * (c² - 1) * cos(π*x) * sin(π*t)
        virtual double source_term(const dealii::Point<dim> &p, double t) const override
        {
            const double x = p[0];
            const double c2 = wave_speed_ * wave_speed_;
            return M_PI * M_PI * (c2 - 1.0) * std::cos(M_PI * x) * std::sin(M_PI * t);
        }
        
        virtual double value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
        {
            (void)component;
            return boundary_value(p, this->get_time());
        }
        
    private:
        double wave_speed_;
    };

} // namespace WaveEquation
