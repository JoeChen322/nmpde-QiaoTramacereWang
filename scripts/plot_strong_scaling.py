#!/usr/bin/env python3
"""
Plot strong scaling results for the Wave Equation Solver.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Strong scaling data (AMG preconditioner, mesh-square-512, 304k DoFs)
procs = np.array([1, 2, 4, 8, 12])
solve_times = np.array([466.2, 254.6, 147.6, 148.1, 140.5])  # seconds

# Compute metrics
T1 = solve_times[0]  # baseline (1 proc)
speedup = T1 / solve_times
efficiency = speedup / procs * 100  # percentage

# Ideal scaling
ideal_speedup = procs.copy().astype(float)

# Create output directory
out_dir = Path(__file__).parent.parent / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Plot 1: Solve Time vs Processors (log-log) ---
fig1, ax1 = plt.subplots(figsize=(5, 4))
ax1.loglog(procs, solve_times, 'bo-', markersize=8, linewidth=2, label='Measured')
ax1.loglog(procs, T1 / procs, 'k--', linewidth=1.5, alpha=0.7, label='Ideal')
ax1.set_xlabel('Number of MPI Processes', fontsize=11)
ax1.set_ylabel('Solve Time (s)', fontsize=11)
ax1.set_title('Strong Scaling: Solve Time', fontsize=12, fontweight='bold')
ax1.set_xticks(procs)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(out_dir / "strong_scaling_time.png", dpi=150, bbox_inches='tight')
print(f"Saved: {out_dir / 'strong_scaling_time.png'}")
plt.close()

# --- Plot 2: Speedup vs Processors (log-log) ---
fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.loglog(procs, speedup, 'go-', markersize=8, linewidth=2, label='Measured')
ax2.loglog(procs, ideal_speedup, 'k--', linewidth=1.5, alpha=0.7, label='Ideal (linear)')
ax2.set_xlabel('Number of MPI Processes', fontsize=11)
ax2.set_ylabel('Speedup $S_p = T_1 / T_p$', fontsize=11)
ax2.set_title('Strong Scaling: Speedup', fontsize=12, fontweight='bold')
ax2.set_xticks(procs)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(out_dir / "strong_scaling_speedup.png", dpi=150, bbox_inches='tight')
print(f"Saved: {out_dir / 'strong_scaling_speedup.png'}")
plt.close()

# --- Plot 3: Efficiency vs Processors ---
fig3, ax3 = plt.subplots(figsize=(5, 4))
bars = ax3.bar(procs, efficiency, color='steelblue', edgecolor='black', width=0.8)
ax3.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (100%)')
ax3.set_xlabel('Number of MPI Processes', fontsize=11)
ax3.set_ylabel('Parallel Efficiency (%)', fontsize=11)
ax3.set_title('Strong Scaling: Efficiency', fontsize=12, fontweight='bold')
ax3.set_xticks(procs)
ax3.set_ylim([0, 110])
ax3.grid(True, alpha=0.3, axis='y')

# Add efficiency values on bars
for bar, eff in zip(bars, efficiency):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{eff:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(out_dir / "strong_scaling_efficiency.png", dpi=150, bbox_inches='tight')
print(f"Saved: {out_dir / 'strong_scaling_efficiency.png'}")
plt.close()

# Print summary table
print("\n" + "="*60)
print("STRONG SCALING SUMMARY (mesh-square-512, 304k DoFs, AMG)")
print("="*60)
print(f"{'Procs':<8} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<10}")
print("-"*60)
for p, t, s, e in zip(procs, solve_times, speedup, efficiency):
    print(f"{p:<8} {t:<12.1f} {s:<10.2f} {e:<10.1f}%")
print("="*60)
