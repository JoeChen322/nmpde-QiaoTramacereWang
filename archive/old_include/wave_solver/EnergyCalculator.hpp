#pragma once

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

namespace WaveEquation
{
namespace Utilities
{
    /**
     * @brief Utility class for energy calculations in wave equation
     */
    class EnergyCalculator
    {
    public:
        /**
         * @brief Compute total energy (kinetic + potential) for the wave equation
         * @param velocity Velocity field vector
         * @param displacement Displacement field vector
         * @param mass_matrix Mass matrix M
         * @param stiffness_matrix Stiffness matrix K
         * @return Total energy E = 0.5 * (v^T M v + u^T K u)
         */
        static double compute_total_energy(
            const dealii::Vector<double> &velocity,
            const dealii::Vector<double> &displacement,
            const dealii::SparseMatrix<double> &mass_matrix,
            const dealii::SparseMatrix<double> &stiffness_matrix);
        
        /**
         * @brief Compute kinetic energy
         * @param velocity Velocity field vector
         * @param mass_matrix Mass matrix M
         * @return Kinetic energy E_k = 0.5 * v^T M v
         */
        static double compute_kinetic_energy(
            const dealii::Vector<double> &velocity,
            const dealii::SparseMatrix<double> &mass_matrix);
        
        /**
         * @brief Compute potential energy
         * @param displacement Displacement field vector
         * @param stiffness_matrix Stiffness matrix K
         * @return Potential energy E_p = 0.5 * u^T K u
         */
        static double compute_potential_energy(
            const dealii::Vector<double> &displacement,
            const dealii::SparseMatrix<double> &stiffness_matrix);
    };

} // namespace Utilities
} // namespace WaveEquation
