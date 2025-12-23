#!/usr/bin/env python3
"""
Plot weak scaling results for the Wave Equation Solver.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Weak scaling data (AMG preconditioner, ~4-5k DoFs per processor)
procs = np.array([1, 2, 4, 8, 12])
dofs = np.array([4887, 9571, 19237, 37854, 56389])
solve_times = np.array([4.68, 5.76, 7.38, 15.61, 22.50])  # seconds
cg_iters = np.array([15456, 14926, 14894, 14771, 14298])
avg_cg = cg_iters / 1000  # per step (1000 steps)

# Compute metrics
dofs_per_proc = dofs / procs
T1 = solve_times[0]  # baseline (1 proc)
efficiency = T1 / solve_times * 100  # percentage (ideal = 100%)

# Create output directory
out_dir = Path(__file__).parent.parent / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Plot 1: Solve Time vs Processors (log-log) ---
fig1, ax1 = plt.subplots(figsize=(5, 4))
ax1.loglog(procs, solve_times, 'bo-', markersize=8, linewidth=2, label='Measured')
ax1.axhline(y=T1, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (constant)')
ax1.set_xlabel('Number of MPI Processes', fontsize=11)
ax1.set_ylabel('Solve Time (s)', fontsize=11)
ax1.set_title('Weak Scaling: Solve Time', fontsize=12, fontweight='bold')
ax1.set_xticks(procs)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(out_dir / "weak_scaling_time.png", dpi=150, bbox_inches='tight')
print(f"Saved: {out_dir / 'weak_scaling_time.png'}")
plt.close()

# --- Plot 2: Efficiency vs Processors ---
fig2, ax2 = plt.subplots(figsize=(5, 4))
bars = ax2.bar(procs, efficiency, color='coral', edgecolor='black', width=0.8)
ax2.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (100%)')
ax2.set_xlabel('Number of MPI Processes', fontsize=11)
ax2.set_ylabel('Weak Scaling Efficiency (%)', fontsize=11)
ax2.set_title('Weak Scaling: Efficiency $E_p = T_1 / T_p$', fontsize=12, fontweight='bold')
ax2.set_xticks(procs)
ax2.set_ylim([0, 120])
ax2.grid(True, alpha=0.3, axis='y')

# Add efficiency values on bars
for bar, eff in zip(bars, efficiency):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{eff:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(out_dir / "weak_scaling_efficiency.png", dpi=150, bbox_inches='tight')
print(f"Saved: {out_dir / 'weak_scaling_efficiency.png'}")
plt.close()

# --- Plot 3: CG Iterations (shows preconditioner quality) ---
fig3, ax3 = plt.subplots(figsize=(5, 4))
ax3.plot(procs, avg_cg, 'gs-', markersize=8, linewidth=2)
ax3.set_xlabel('Number of MPI Processes', fontsize=11)
ax3.set_ylabel('Avg CG Iterations per Step', fontsize=11)
ax3.set_title('Preconditioner Quality (AMG)', fontsize=12, fontweight='bold')
ax3.set_xticks(procs)
ax3.set_ylim([0, 20])
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 13])


plt.tight_layout()
plt.savefig(out_dir / "weak_scaling_cg_iters.png", dpi=150, bbox_inches='tight')
print(f"Saved: {out_dir / 'weak_scaling_cg_iters.png'}")
plt.close()

# Print summary table
print("\n" + "="*70)
print("WEAK SCALING SUMMARY (~4-5k DoFs/proc, AMG preconditioner)")
print("="*70)
print(f"{'Procs':<8} {'DoFs':<10} {'DoFs/proc':<12} {'Time (s)':<12} {'Efficiency':<12} {'Avg CG':<8}")
print("-"*70)
for p, d, dpp, t, e, cg in zip(procs, dofs, dofs_per_proc, solve_times, efficiency, avg_cg):
    print(f"{p:<8} {d:<10} {dpp:<12.0f} {t:<12.2f} {e:<12.0f}% {cg:<8.1f}")
print("="*70)
print("\nNote: Weak scaling efficiency is limited on shared-memory systems")
print("      due to memory bandwidth saturation. On a distributed cluster,")
print("      each node would have its own memory bus, improving scaling.")
