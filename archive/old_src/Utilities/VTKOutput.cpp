#include "VTKOutput.hpp"

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_fe.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace WaveEquation
{
namespace Utilities
{
    // Helper function to create directory recursively if it doesn't exist
    static void create_directory_if_needed(const std::string &dir_path)
    {
        if (dir_path.empty() || dir_path == ".")
            return;
            
        struct stat st;
        if (stat(dir_path.c_str(), &st) == 0)
        {
            // Directory already exists
            return;
        }
        
        // Find parent directory
        size_t pos = dir_path.find_last_of('/');
        if (pos != std::string::npos)
        {
            std::string parent = dir_path.substr(0, pos);
            create_directory_if_needed(parent);
        }
        
        // Create this directory
        mkdir(dir_path.c_str(), 0755);
    }

    template <int dim>
    void VTKOutput<dim>::write_vtk(
        const dealii::DoFHandler<dim> &dof_handler,
        const dealii::Vector<double> &displacement,
        const dealii::Vector<double> &velocity,
        unsigned int step,
        const std::string &filename_base,
        const std::string &output_dir)
    {
        // Create output directory if needed
        create_directory_if_needed(output_dir);
        
        // VTK output for ParaView visualization
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        
        // Add displacement field
        data_out.add_data_vector(displacement, "displacement");
        
        // Add velocity field
        data_out.add_data_vector(velocity, "velocity");
        
        data_out.build_patches();
        
        // Write VTK file
        std::string filename = output_dir + "/" + filename_base + "_" + std::to_string(step) + ".vtk";
        std::ofstream output(filename);
        data_out.write_vtk(output);
        output.close();
    }

    template <int dim>
    void VTKOutput<dim>::write_text_1d(
        const dealii::Vector<double> &solution,
        unsigned int step,
        double time,
        const std::string &filename_base,
        const std::string &output_dir)
    {
        // Create output directory if needed
        create_directory_if_needed(output_dir);
        
        std::string filename = output_dir + "/" + filename_base + "_step_" + std::to_string(step) + ".txt";
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

    template <int dim>
    void VTKOutput<dim>::write_text_2d(
        const dealii::DoFHandler<dim> &dof_handler,
        const dealii::Vector<double> &displacement,
        const dealii::Vector<double> &velocity,
        unsigned int step,
        double time,
        const std::string &filename_base,
        const std::string &output_dir)
    {
        // Create output directory if needed
        create_directory_if_needed(output_dir);
        
        std::string filename = output_dir + "/" + filename_base + "_step_" + std::to_string(step) + ".txt";
        std::ofstream out(filename);
        
        out << "# Step: " << step << ", Time: " << time << std::endl;
        out << "# X Y Displacement Velocity" << std::endl;
        
        // Output data at each DoF location (iterate over all DoFs)
        const unsigned int n_dofs = dof_handler.n_dofs();
        
        // For simplex elements, we need to get the support points
        std::vector<dealii::Point<dim>> support_points(n_dofs);
        dealii::DoFTools::map_dofs_to_support_points(
            dealii::MappingFE<dim>(dof_handler.get_fe()),
            dof_handler,
            support_points);
        
        // Write each DoF with its position and values
        for (unsigned int i = 0; i < n_dofs; ++i)
        {
            out << support_points[i][0] << " " 
                << support_points[i][1] << " " 
                << displacement[i] << " " 
                << velocity[i] << std::endl;
        }
        
        out.close();
    }

    // Explicit instantiation for 1D and 2D
    template class VTKOutput<1>;
    template class VTKOutput<2>;

} // namespace Utilities
} // namespace WaveEquation
