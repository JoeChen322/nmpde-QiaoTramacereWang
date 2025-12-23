#ifndef ERROR_COMPUTER_HPP
#define ERROR_COMPUTER_HPP

#include "ProblemBase.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/lac/vector.h>

#include <cmath>

namespace WaveEquation
{
namespace Utilities
{

/**
 * @brief Helper class to compute L2 error norms for convergence testing
 * 
 * This class provides static methods to compute the L2 norm of the error
 * between a numerical solution and an exact solution from a ProblemBase.
 * 
 * @tparam dim Spatial dimension (1 or 2)
 */
template <int dim>
class ErrorComputer
{
public:
    /**
     * @brief Compute L2 error between numerical and exact solution (1D case)
     * 
     * Uses standard quadrilateral/interval elements with QGauss quadrature.
     * 
     * @param dof_handler DoF handler for the numerical solution
     * @param numerical_solution Vector containing the numerical solution values
     * @param exact_problem Problem definition containing the exact solution
     * @param time Current time for evaluating the exact solution
     * @return L2 norm of the error
     */
    template <typename ProblemType>
    static double compute_l2_error(
        const dealii::DoFHandler<dim> &dof_handler,
        const dealii::Vector<double> &numerical_solution,
        const ProblemType &exact_problem,
        double time)
    {
        static_assert(dim == 1, "This specialization is for 1D only. Use the 2D version for dim=2.");
        
        dealii::QGauss<dim> quadrature(3);  // Use 3-point Gauss quadrature
        dealii::FEValues<dim> fe_values(dof_handler.get_fe(),
                                       quadrature,
                                       dealii::update_values |
                                       dealii::update_quadrature_points |
                                       dealii::update_JxW_values);
        
        const unsigned int n_q_points = quadrature.size();
        std::vector<double> numerical_values(n_q_points);
        
        double l2_error_squared = 0.0;
        
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            fe_values.get_function_values(numerical_solution, numerical_values);
            
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const dealii::Point<dim> &q_point = fe_values.quadrature_point(q);
                const double exact_value = exact_problem.exact_solution(q_point, time);
                const double error = numerical_values[q] - exact_value;
                
                l2_error_squared += error * error * fe_values.JxW(q);
            }
        }
        
        return std::sqrt(l2_error_squared);
    }
};

/**
 * @brief Specialization of ErrorComputer for 2D simplex elements
 * 
 * Uses triangular elements with QGaussSimplex quadrature and MappingFE.
 */
template <>
class ErrorComputer<2>
{
public:
    /**
     * @brief Compute L2 error between numerical and exact solution (2D case)
     * 
     * Uses simplex (triangular) elements with QGaussSimplex quadrature.
     * 
     * @param dof_handler DoF handler for the numerical solution
     * @param fe Finite element used for the solution
     * @param numerical_solution Vector containing the numerical solution values
     * @param exact_problem Problem definition containing the exact solution
     * @param time Current time for evaluating the exact solution
     * @return L2 norm of the error
     */
    template <typename ProblemType>
    static double compute_l2_error(
        const dealii::DoFHandler<2> &dof_handler,
        const dealii::FiniteElement<2> &fe,
        const dealii::Vector<double> &numerical_solution,
        const ProblemType &exact_problem,
        double time)
    {
        dealii::QGaussSimplex<2> quadrature(3);  // Use 3-point Gauss quadrature for triangles
        dealii::MappingFE<2> mapping(fe);  // Proper mapping for simplex elements
        dealii::FEValues<2> fe_values(mapping,
                                     fe,
                                     quadrature,
                                     dealii::update_values |
                                     dealii::update_quadrature_points |
                                     dealii::update_JxW_values);
        
        const unsigned int n_q_points = quadrature.size();
        std::vector<double> numerical_values(n_q_points);
        
        double l2_error_squared = 0.0;
        
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            fe_values.get_function_values(numerical_solution, numerical_values);
            
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const dealii::Point<2> &q_point = fe_values.quadrature_point(q);
                const double exact_value = exact_problem.exact_solution(q_point, time);
                const double error = numerical_values[q] - exact_value;
                
                l2_error_squared += error * error * fe_values.JxW(q);
            }
        }
        
        return std::sqrt(l2_error_squared);
    }
};

} // namespace Utilities
} // namespace WaveEquation

#endif // ERROR_COMPUTER_HPP
