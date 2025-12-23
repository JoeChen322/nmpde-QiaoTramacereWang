#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_phase_drift_semi_T(df: pd.DataFrame) -> np.ndarray:
    """
    Convert the stored phase drift (relative to omega_exact) into drift relative
    to omega_semi at final time T.

    If Wave stores:
        phase_drift_T = phi(T) - (phi(0) + omega_exact*T)

    Then drift vs omega_semi is:
        drift_semi_T = phi(T) - (phi(0) + omega_semi*T)
                     = phase_drift_T + (omega_exact - omega_semi)*T
    """
    required = {"phase_drift_T", "omega_exact", "omega_semi", "T"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns needed for drift conversion: {sorted(missing)}")

    return (
        df["phase_drift_T"].to_numpy()
        + (df["omega_exact"].to_numpy() - df["omega_semi"].to_numpy()) * df["T"].to_numpy()
    )


def main():
    csv_path = "../results/dispersion/dispersion_summary.csv"
    out_dir  = "../results/plots"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Expect columns from the updated study:
    # dt, T, omega_semi_dt, ratio_num_semi, ratio_pred_theta05, m, n,
    # omega_exact, omega_semi, phase_drift_T
    df = df.sort_values(["m", "n", "omega_semi_dt"])

    # -------------------------
    # Plot 1: Standard CN dispersion curve collapse
    # -------------------------
    plt.figure()
    for (m, n), g in df.groupby(["m", "n"]):
        x = g["omega_semi_dt"].to_numpy()
        y = g["ratio_num_semi"].to_numpy()
        plt.plot(x, y, marker="o", label=rf"$(m,n)=({m},{n})$")

    # Theory curve for theta=0.5: (2/x) atan(x/2)
    xmax = float(df["omega_semi_dt"].max()) if len(df) else 1.0
    xgrid = np.linspace(0.0, 1.05 * xmax, 400)
    ygrid = np.ones_like(xgrid)
    mask = xgrid > 1e-14
    ygrid[mask] = (2.0 / xgrid[mask]) * np.arctan(xgrid[mask] / 2.0)

    plt.plot(xgrid, ygrid, label=r"theory $\theta=0.5$")
    plt.xlabel(r"$x=\omega_{\mathrm{semi}}\Delta t$")
    plt.ylabel(r"$\omega_{\mathrm{num}}/\omega_{\mathrm{semi}}$")
    plt.title(r"Time dispersion ($\theta=0.5$)")
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(out_dir, "time_dispersion.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Wrote:", out_path)

    # -------------------------
    # Plot 2: Accumulated phase lag at final time T vs dt (mode-separated)
    # -------------------------
    # Drift vs semi at T (radians)
    drift_semi_T = compute_phase_drift_semi_T(df)

    # Convert to "lag in cycles"
    # lag_cycles_T > 0 means the numerical solution lags behind the semi-discrete oscillation
    lag_cycles_T = -drift_semi_T / (2.0 * np.pi)

    df2 = df.copy()
    df2["phase_drift_semi_T"] = drift_semi_T
    df2["lag_cycles_T"] = lag_cycles_T

    df2 = df2.sort_values(["m", "n", "dt"])

    plt.figure()
    for (m, n), g in df2.groupby(["m", "n"]):
        x = g["dt"].to_numpy()
        y = g["lag_cycles_T"].to_numpy()
        plt.plot(x, y, marker="o", label=rf"$(m,n)=({m},{n})$")

    plt.xlabel(r"$\Delta t$")
    # simpler wording:
    plt.ylabel(r"phase lag at $T$ (cycles; positive = lag)")

    T_unique = np.unique(df2["T"].to_numpy())
    if len(T_unique) == 1:
        plt.title(rf"Phase lag vs $\Delta t$ ($\theta=0.5$, $T={T_unique[0]:g}$)")
    else:
        plt.title(r"Phase lag vs $\Delta t$ ($\theta=0.5$)")

    plt.grid(True)
    plt.legend()

    out_path = os.path.join(out_dir, "phase_lag_vs_dt.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
