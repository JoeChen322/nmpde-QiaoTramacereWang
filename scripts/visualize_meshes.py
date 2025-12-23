import gmsh
import os

def visualize_mesh_gmsh(mesh_filename):
    """Visualize mesh using Gmsh's built-in GUI"""
    
    if not os.path.exists(mesh_filename):
        print(f"Error: Mesh file '{mesh_filename}' not found!")
        return
    
    # Initialize Gmsh
    gmsh.initialize()
    
    # Load the mesh file
    gmsh.open(mesh_filename)
    
    # Set some visualization options
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)  # Show faces
    gmsh.option.setNumber("Mesh.Lines", 1)         # Show lines
    gmsh.option.setNumber("Mesh.Points", 1)        # Show points
    gmsh.option.setNumber("Mesh.ColorCarousel", 2) # Color by element type
    
    print(f"Visualizing mesh: {mesh_filename}")
    print("Close the Gmsh window to continue...")
    
    # Run the GUI
    gmsh.fltk.run()
    
    # Finalize
    gmsh.finalize()

def visualize_multiple_meshes(mesh_files):
    """Visualize multiple mesh files sequentially"""
    
    for mesh_file in mesh_files:
        if os.path.exists(mesh_file):
            print(f"\nVisualizing: {mesh_file}")
            visualize_mesh_gmsh(mesh_file)
        else:
            print(f"Mesh file not found: {mesh_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize mesh files using Gmsh GUI")
    parser.add_argument("meshes", nargs="*", help="Mesh files to visualize (default: square_level_*.msh)")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing mesh files")
    args = parser.parse_args()
    
    if args.meshes:
        mesh_files = args.meshes
    else:
        # Default mesh files
        import glob
        mesh_files = sorted(glob.glob(os.path.join(args.dir, "square_level_*.msh")))
        if not mesh_files:
            mesh_files = sorted(glob.glob(os.path.join(args.dir, "mesh-square-*.msh")))
    
    if not mesh_files:
        print("No mesh files found. Provide mesh files as arguments or use --dir to specify directory.")
    else:
        visualize_multiple_meshes(mesh_files)