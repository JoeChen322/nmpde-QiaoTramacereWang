import gmsh

def generate_and_analyze_square_mesh(refinement_levels):
    """Generate multiple refinement levels and analyze mesh quality"""
    
    results = {}
    
    for level in refinement_levels:
        gmsh.initialize()
        gmsh.model.add(f"square_level_{level}")
        
        # Create square
        gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2)
        gmsh.model.occ.synchronize()
        
        # Set mesh size
        mesh_size = 0.5 / (2 ** level)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        
        # Generate mesh
        gmsh.model.mesh.generate(2)
        
        # Get mesh information
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(2)
        nodes = gmsh.model.mesh.getNodes()
        
        # Calculate mesh statistics
        num_elements = len(element_tags[0]) if element_tags else 0
        num_nodes = len(nodes[1]) // 3
        
        results[level] = {
            'mesh_size': mesh_size,
            'elements': num_elements,
            'nodes': num_nodes,
            'filename': f"square_level_{level}.msh"
        }
        
        print(f"Level {level}: {num_elements} elements, {num_nodes} nodes, size={mesh_size:.4f}")
        
        gmsh.write(results[level]['filename'])
        gmsh.finalize()
    
    return results

# Generate meshes and analyze
refinement_levels = [0, 1, 2, 3, 4, 5, 6, 7]
results = generate_and_analyze_square_mesh(refinement_levels)

# Print summary
print("\n=== Mesh Refinement Summary ===")
for level, data in results.items():
    print(f"Level {level}: {data['elements']:6d} elements, "
          f"size={data['mesh_size']:.4f}")