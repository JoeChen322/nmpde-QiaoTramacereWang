#!/usr/bin/env python3
"""
Visualize 2D Wave Equation Results from Text Files
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.tri import Triangulation
import sys
import os
import glob

# Determine the base path for output files
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)

# Check possible locations for build directory
possible_paths = [
    os.path.join(workspace_root, 'build', '2d', 'txt'),  # New organized structure
    os.path.join(workspace_root, 'build'),  # Old location (fallback)
    os.path.join(script_dir, '..', 'build', '2d', 'txt'),  # Relative to script
    os.path.join(script_dir, '..', 'build'),  # Old relative (fallback)
    'build/2d/txt',  # From current directory
    'build',  # Old from current (fallback)
]

BASE_PATH = None
for path in possible_paths:
    if os.path.isdir(path):
        # Check if it contains 2D output files
        test_files = glob.glob(os.path.join(path, 'output_2d_step_*.txt'))
        if test_files:
            BASE_PATH = path
            break

if BASE_PATH is None:
    print("Error: Could not find build directory with 2D output text files!")
    print("Searched in:")
    for path in possible_paths:
        print(f"  - {os.path.abspath(path)}")
    sys.exit(1)

print(f"Found output files in: {os.path.abspath(BASE_PATH)}")


def read_2d_text_file(filename):
    """Read 2D wave data from text file"""
    try:
        data = np.loadtxt(filename, comments='#')
        x = data[:, 0]
        y = data[:, 1]
        displacement = data[:, 2]
        velocity = data[:, 3]
        return x, y, displacement, velocity
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None, None, None


# Find all text files
txt_files = sorted(glob.glob(os.path.join(BASE_PATH, 'output_2d_step_*.txt')))

if not txt_files:
    print(f"No text files found in {BASE_PATH}!")
    print("Make sure you have run the 2D wave solver to generate output files.")
    sys.exit(1)

print(f"Found {len(txt_files)} text files")

# Load all data
print("Loading data files...")
data = []
for i, txt_file in enumerate(txt_files):
    if i % 20 == 0:
        print(f"  Loading {i+1}/{len(txt_files)}...")
    
    x, y, displacement, velocity = read_2d_text_file(txt_file)
    if x is not None:
        # Extract step number from filename
        filename = os.path.basename(txt_file)
        step = int(filename.replace('output_2d_step_', '').replace('.txt', ''))
        data.append({
            'step': step,
            'x': x,
            'y': y,
            'displacement': displacement,
            'velocity': velocity
        })

if not data:
    print("No data could be loaded!")
    sys.exit(1)

# Sort by step number
data.sort(key=lambda d: d['step'])
print(f"Loaded {len(data)} time steps")

# Create triangulation for the mesh (assumes fixed mesh topology)
# Remove duplicate points for triangulation
first_data = data[0]
points = np.column_stack([first_data['x'], first_data['y']])
unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)

# Create triangulation
tri = Triangulation(unique_points[:, 0], unique_points[:, 1])

print(f"Mesh: {len(unique_points)} unique points, {len(tri.triangles)} triangles")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Determine color scale limits from all data
all_displacements = np.concatenate([d['displacement'] for d in data])
all_velocities = np.concatenate([d['velocity'] for d in data])

# Use unique values for color scaling
unique_displacements = np.unique(all_displacements)
unique_velocities = np.unique(all_velocities)

vmin_u = np.min(unique_displacements)
vmax_u = np.max(unique_displacements)
vmin_v = np.min(unique_velocities)
vmax_v = np.max(unique_velocities)

print(f"Displacement range: [{vmin_u:.6f}, {vmax_u:.6f}]")
print(f"Velocity range: [{vmin_v:.6f}, {vmax_v:.6f}]")

# Map data to unique points for the first frame
unique_displacement = np.zeros(len(unique_points))
unique_velocity = np.zeros(len(unique_points))

for i, idx in enumerate(inverse_indices):
    unique_displacement[idx] = data[0]['displacement'][i]
    unique_velocity[idx] = data[0]['velocity'][i]

# Initial plots
tcf1 = ax1.tripcolor(tri, unique_displacement, cmap='seismic', 
                     vmin=vmin_u, vmax=vmax_u, shading='flat')
tcf2 = ax2.tripcolor(tri, unique_velocity, cmap='seismic', 
                     vmin=vmin_v, vmax=vmax_v, shading='flat')

# Add colorbars
cbar1 = plt.colorbar(tcf1, ax=ax1)
cbar1.set_label('Displacement u(x,y,t)', fontsize=11)

cbar2 = plt.colorbar(tcf2, ax=ax2)
cbar2.set_label('Velocity v(x,y,t)', fontsize=11)

# Set labels
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Displacement', fontsize=14, fontweight='bold')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Velocity', fontsize=14, fontweight='bold')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Time display
time_text = fig.suptitle('', fontsize=16, fontweight='bold')

plt.tight_layout()


def init():
    """Initialize animation"""
    return tcf1, tcf2, time_text


def animate(frame):
    """Update animation frame"""
    d = data[frame]
    
    # Map data to unique points
    unique_displacement = np.zeros(len(unique_points))
    unique_velocity = np.zeros(len(unique_points))
    
    for i, idx in enumerate(inverse_indices):
        unique_displacement[idx] = d['displacement'][i]
        unique_velocity[idx] = d['velocity'][i]
    
    # Update plots by recreating them
    ax1.clear()
    ax2.clear()
    
    tcf1_new = ax1.tripcolor(tri, unique_displacement, cmap='seismic', 
                              vmin=vmin_u, vmax=vmax_u, shading='flat')
    tcf2_new = ax2.tripcolor(tri, unique_velocity, cmap='seismic', 
                              vmin=vmin_v, vmax=vmax_v, shading='flat')
    
    # Restore labels
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Displacement', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Velocity', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Update time display
    time = d['step'] * 0.01
    time_text.set_text(f'2D Wave Equation - Step {d["step"]}, Time = {time:.2f} s')
    
    return tcf1_new, tcf2_new, time_text


# Create animation
print("\nCreating animation...")
anim = FuncAnimation(fig, animate, init_func=init,
                     frames=len(data), interval=50, blit=False, repeat=True)

print("\nDisplaying animation. Close the window when done.")
print("(The animation will loop continuously)")

plt.show()
