#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    csv_path = "../results/dispersion/dispersion_spatial.csv"
    out_dir  = "../results/plots"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.sort_values("h_nom")

    h = df["h_nom"].to_numpy()
    ratio_semi = df["ratio_semi_exact"].to_numpy()
    ratio_num_semi = df["ratio_num_semi"].to_numpy()

    # 1) Spatial dispersion: omega_semi/omega_exact vs h_nom
    plt.figure()
    plt.plot(h, ratio_semi, marker="o")
    plt.xlabel(r"$h_{\mathrm{nom}}=1/N$")
    plt.ylabel(r"$\omega_{\mathrm{semi}}/\omega_{\mathrm{exact}}$")
    plt.title("Spatial dispersion (θ=0.5, tiny dt)")
    plt.grid(True)
    out1 = os.path.join(out_dir, "spatial_dispersion_ratio_semi.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    print("Wrote:", out1)

    # 2) Time-negligible check: omega_num/omega_semi vs h_nom (should be ~1)
    plt.figure()
    plt.plot(h, ratio_num_semi, marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel(r"$h_{\mathrm{nom}}=1/N$")
    plt.ylabel(r"$\omega_{\mathrm{num}}/\omega_{\mathrm{semi}}$")
    plt.title("Check: time dispersion negligible (θ=0.5, tiny dt)")
    plt.grid(True)
    out2 = os.path.join(out_dir, "spatial_time_negligible_check.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    print("Wrote:", out2)

if __name__ == "__main__":
    main()
