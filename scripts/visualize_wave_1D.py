#!/usr/bin/env python3
"""
Visualize 1D Wave Equation Results
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# Determine the base path for output files
# Try build directory relative to script location and workspace root
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(script_dir)

# Check possible locations for build directory
possible_paths = [
    os.path.join(workspace_root, 'build', '1d', 'txt'),  # New organized structure
    os.path.join(workspace_root, 'build'),  # Old location (fallback)
    os.path.join(script_dir, '..', 'build', '1d', 'txt'),  # Relative to script
    os.path.join(script_dir, '..', 'build'),  # Old relative (fallback)
    'build/1d/txt',  # From current directory
    'build',  # Old from current (fallback)
]

BASE_PATH = None
for path in possible_paths:
    if os.path.isdir(path):
        # Check if it contains output files
        test_file = os.path.join(path, 'output_1d_step_0.txt')
        if os.path.exists(test_file):
            BASE_PATH = path
            break

if BASE_PATH is None:
    print("Error: Could not find build directory with output files!")
    print("Searched in:")
    for path in possible_paths:
        print(f"  - {os.path.abspath(path)}")
    sys.exit(1)

print(f"Found output files in: {os.path.abspath(BASE_PATH)}")

# Read all time steps
steps = list(range(0, 601, 1))
data = {}

print("Loading data files...")
for step in steps:
    filename = os.path.join(BASE_PATH, f'output_1d_step_{step}.txt')
    if not os.path.exists(filename):
        print(f"Warning: Could not find {filename}")
        continue
    try:
        d = np.loadtxt(filename, comments='#')
        data[step] = d
    except Exception as e:
        print(f"  Error loading {filename}: {e}")

if not data:
    print(f"No data files found in {BASE_PATH}!")
    print("Make sure you have run the wave solver to generate output files.")
    sys.exit(1)

print(f"Loaded {len(data)} time steps")

# Create animation
print("\nCreating animation...")
fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

# Top plot: displacement
line1, = ax1.plot([], [], 'b-', linewidth=2, label='Displacement')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-0.2, 4.0)
ax1.set_xlabel('Position x', fontsize=12)
ax1.set_ylabel('Displacement u(x,t)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()
title = ax1.set_title('', fontsize=14, fontweight='bold')


plt.tight_layout()

# Prepare frames (sorted) and cache arrays for fast access
steps_sorted = sorted(data.keys())
frames_data = [data[s] for s in steps_sorted]
n_frames = len(frames_data)

def init():
    line1.set_data([], [])
    return (line1,)

def animate(i):
    d = frames_data[i]
    line1.set_data(d[:, 0], d[:, 1])
    time = steps_sorted[i] * 0.01
    title.set_text(f'1D Wave Equation: Step {steps_sorted[i]}, Time = {time:.2f} s')
    return (line1,)

# Faster playback: shorter interval and use blitting
anim = FuncAnimation(fig, animate, init_func=init,
                     frames=n_frames, interval=50, blit=True, repeat=True)

print("\nDisplaying animation. Close the window when done.")
print("(The animation will loop continuously)")

plt.show()

# Optionally save the animation
# save = input("\nSave animation as GIF? (y/n): ").lower().strip()
# if save == 'y':
#     print("Saving animation (this may take a moment)...")
#     anim.save('wave_animation.gif', writer='pillow', fps=5)
#     print("Saved as wave_animation.gif")
