#include "EnergyCalculator.hpp"

namespace WaveEquation
{
namespace Utilities
{
    double EnergyCalculator::compute_total_energy(
        const dealii::Vector<double> &velocity,
        const dealii::Vector<double> &displacement,
        const dealii::SparseMatrix<double> &mass_matrix,
        const dealii::SparseMatrix<double> &stiffness_matrix)
    {
        double kinetic = compute_kinetic_energy(velocity, mass_matrix);
        double potential = compute_potential_energy(displacement, stiffness_matrix);
        
        return kinetic + potential;
    }

    double EnergyCalculator::compute_kinetic_energy(
        const dealii::Vector<double> &velocity,
        const dealii::SparseMatrix<double> &mass_matrix)
    {
        // Kinetic energy: E_k = 0.5 * v^T M v
        dealii::Vector<double> tmp(velocity.size());
        mass_matrix.vmult(tmp, velocity);
        
        return 0.5 * (velocity * tmp);
    }

    double EnergyCalculator::compute_potential_energy(
        const dealii::Vector<double> &displacement,
        const dealii::SparseMatrix<double> &stiffness_matrix)
    {
        // Potential energy: E_p = 0.5 * u^T K u
        dealii::Vector<double> tmp(displacement.size());
        stiffness_matrix.vmult(tmp, displacement);
        
        return 0.5 * (displacement * tmp);
    }

} // namespace Utilities
} // namespace WaveEquation
