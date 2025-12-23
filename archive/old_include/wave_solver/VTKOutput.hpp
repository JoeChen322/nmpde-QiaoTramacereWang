#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <string>
#include <vector>

namespace WaveEquation
{
namespace Utilities
{
    /**
     * @brief Utility class for VTK output of wave equation solutions
     */
    template <int dim>
    class VTKOutput
    {
    public:
        /**
         * @brief Write VTK file with displacement and velocity fields
         * @param dof_handler The DoF handler
         * @param displacement Displacement field vector
         * @param velocity Velocity field vector
         * @param step Current time step number
         * @param filename_base Base name for output file (default: "solution")
         * @param output_dir Output directory path (default: ".")
         */
        static void write_vtk(
            const dealii::DoFHandler<dim> &dof_handler,
            const dealii::Vector<double> &displacement,
            const dealii::Vector<double> &velocity,
            unsigned int step,
            const std::string &filename_base = "solution",
            const std::string &output_dir = ".");
        
        /**
         * @brief Write simple text output for 1D solutions
         * @param solution Solution vector
         * @param step Current time step number
         * @param time Current simulation time
         * @param filename_base Base name for output file (default: "output")
         * @param output_dir Output directory path (default: ".")
         */
        static void write_text_1d(
            const dealii::Vector<double> &solution,
            unsigned int step,
            double time,
            const std::string &filename_base = "output",
            const std::string &output_dir = ".");
        
        /**
         * @brief Write simple text output for 2D solutions
         * @param dof_handler The DoF handler
         * @param displacement Displacement field vector
         * @param velocity Velocity field vector
         * @param step Current time step number
         * @param time Current simulation time
         * @param filename_base Base name for output file (default: "output_2d")
         * @param output_dir Output directory path (default: ".")
         */
        static void write_text_2d(
            const dealii::DoFHandler<dim> &dof_handler,
            const dealii::Vector<double> &displacement,
            const dealii::Vector<double> &velocity,
            unsigned int step,
            double time,
            const std::string &filename_base = "output_2d",
            const std::string &output_dir = ".");
    };

} // namespace Utilities
} // namespace WaveEquation
