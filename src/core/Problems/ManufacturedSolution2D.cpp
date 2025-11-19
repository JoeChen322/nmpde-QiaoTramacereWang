#include "ProblemBase.hpp"
#include <cmath>

namespace WaveEquation
{
    /**
     * Method of Manufactured Solutions for 2D Wave Equation with Neumann BCs
     * 
     * Wave equation: u_tt = c² ∇²u + f(x,y,t)
     * Domain: [-1, 1] × [-1, 1]
     * Boundary conditions: ∂u/∂n = 0 on all boundaries (Neumann/reflecting)
     * 
     * We choose a manufactured solution that satisfies Neumann boundary conditions:
     * u_exact(x,y,t) = cos(π*x) * cos(π*y) * cos(ω*t)
     * 
     * where ω = c*π*√2 for mode (1,1)
     * 
     * Verification of Neumann BCs:
     * ∂u/∂x = -π*sin(π*x)*cos(π*y)*cos(ω*t)
     * At x = ±1: sin(±π) = 0 ✓
     * 
     * ∂u/∂y = -π*cos(π*x)*sin(π*y)*cos(ω*t)
     * At y = ±1: sin(±π) = 0 ✓
     * 
     * Derivatives:
     * u_tt = -ω² * cos(π*x) * cos(π*y) * cos(ω*t)
     * u_xx = -π² * cos(π*x) * cos(π*y) * cos(ω*t)
     * u_yy = -π² * cos(π*x) * cos(π*y) * cos(ω*t)
     * ∇²u = u_xx + u_yy = -2π² * cos(π*x) * cos(π*y) * cos(ω*t)
     * 
     * Therefore, the source term needed is:
     * f(x,y,t) = u_tt - c² * ∇²u
     *          = -ω² * cos(π*x) * cos(π*y) * cos(ω*t) + c² * 2π² * cos(π*x) * cos(π*y) * cos(ω*t)
     *          = cos(π*x) * cos(π*y) * cos(ω*t) * (2c²π² - ω²)
     * 
     * With ω = c*π*√2, we have ω² = 2c²π², so:
     * f(x,y,t) = 0
     * 
     * So our manufactured solution is actually an exact solution of the homogeneous wave equation!
     * This is perfect for testing with Neumann (reflecting) boundary conditions.
     * 
     * Period: T = 2π/ω = 2π/(c*π*√2) = √2/c ≈ 1.414 seconds (for c=1)
     */
    template <int dim>
    class ManufacturedSolution2D : public ProblemBase<dim>
    {
    public:
        static_assert(dim == 2, "ManufacturedSolution2D only works in 2D");
        
        ManufacturedSolution2D(double wave_speed = 1.0)
            : wave_speed_(wave_speed)
            , k_(1.0)  // Mode number in x-direction
            , m_(1.0)  // Mode number in y-direction
            , omega_(wave_speed * M_PI * std::sqrt(k_*k_ + m_*m_))  // Natural frequency
        {
        }
        
        // Exact solution: u(x,y,t) = cos(π*x) * cos(π*y) * cos(ω*t)
        double exact_solution(const dealii::Point<2> &p, double t) const
        {
            const double x = p[0];
            const double y = p[1];
            return std::cos(M_PI * k_ * x) * 
                   std::cos(M_PI * m_ * y) * 
                   std::cos(omega_ * t);
        }
        
        // Time derivative: u_t(x,y,t) = -ω * cos(π*x) * cos(π*y) * sin(ω*t)
        double exact_velocity(const dealii::Point<2> &p, double t) const
        {
            const double x = p[0];
            const double y = p[1];
            return -omega_ * 
                   std::cos(M_PI * k_ * x) * 
                   std::cos(M_PI * m_ * y) * 
                   std::sin(omega_ * t);
        }
        
        // Initial displacement: u(x,y,0) = cos(π*x) * cos(π*y)
        virtual double initial_displacement(const dealii::Point<2> &p) const override
        {
            return exact_solution(p, 0.0);
        }
        
        // Initial velocity: u_t(x,y,0) = 0
        virtual double initial_velocity(const dealii::Point<2> &p) const override
        {
            return exact_velocity(p, 0.0);
        }
        
        // Boundary conditions: ∂u/∂n = 0 (Neumann - automatically satisfied)
        virtual double boundary_value(const dealii::Point<2> &p, double t) const override
        {
            return exact_solution(p, t);
        }
        
        // Source term: f(x,y,t) = 0 (since our manufactured solution is exact)
        virtual double source_term(const dealii::Point<2> &/*p*/, double /*t*/) const override
        {
            return 0.0;
        }
        
        // For dealii::Function interface
        virtual double value(const dealii::Point<2> &p, const unsigned int /*component*/) const override
        {
            return boundary_value(p, this->get_time());
        }
        
        // Getters for testing
        double get_wave_speed() const { return wave_speed_; }
        double get_omega() const { return omega_; }
        double get_period() const { return 2.0 * M_PI / omega_; }
        
    private:
        double wave_speed_;
        double k_;  // Mode number in x-direction
        double m_;  // Mode number in y-direction
        double omega_;  // Angular frequency = c*π*√(k² + m²)
    };
    
    // Explicit instantiation for 2D
    template class ManufacturedSolution2D<2>;

} // namespace WaveEquation
