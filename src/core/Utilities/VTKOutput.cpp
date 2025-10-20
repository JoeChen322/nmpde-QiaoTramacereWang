#include "VTKOutput.hpp"

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <string>

namespace WaveEquation
{
namespace Utilities
{
    template <int dim>
    void VTKOutput<dim>::write_vtk(
        const dealii::DoFHandler<dim> &dof_handler,
        const dealii::Vector<double> &displacement,
        const dealii::Vector<double> &velocity,
        unsigned int step,
        double time,
        const std::string &filename_base)
    {
        // VTK output for ParaView visualization
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        
        // Add displacement field
        data_out.add_data_vector(displacement, "displacement");
        
        // Add velocity field
        data_out.add_data_vector(velocity, "velocity");
        
        data_out.build_patches();
        
        // Write VTK file
        std::string filename = filename_base + "_" + std::to_string(step) + ".vtk";
        std::ofstream output(filename);
        data_out.write_vtk(output);
        output.close();
    }

    template <int dim>
    void VTKOutput<dim>::write_text_1d(
        const dealii::Vector<double> &solution,
        unsigned int step,
        double time,
        const std::string &filename_base)
    {
        std::string filename = filename_base + "_step_" + std::to_string(step) + ".txt";
        std::ofstream out(filename);
        
        out << "# Step: " << step << ", Time: " << time << std::endl;
        out << "# Node Position, Displacement" << std::endl;
        
        // For 1D, we can output the solution at each node
        for (unsigned int i = 0; i < solution.size(); ++i)
        {
            // In 1D, node positions are equally spaced in [-1, 1]
            double x = -1.0 + 2.0 * i / (solution.size() - 1);
            out << x << " " << solution[i] << std::endl;
        }
        
        out.close();
    }

    // Explicit instantiation for 1D and 2D
    template class VTKOutput<1>;
    template class VTKOutput<2>;

} // namespace Utilities
} // namespace WaveEquation
