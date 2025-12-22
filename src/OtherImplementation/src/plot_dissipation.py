from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# IO helpers
# -----------------------------
def read_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)

    # Normalize column names (just in case)
    df.columns = [c.strip() for c in df.columns]

    required = {"m", "n", "theta", "dt", "T", "decay_rate", "energy_csv"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Summary CSV missing columns: {sorted(missing)}")

    # numeric conversions
    for c in ["m", "n"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    for c in ["theta", "dt", "T", "decay_rate", "omega", "omega_dt", "ET_over_E0"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # path column
    df["energy_csv"] = df["energy_csv"].astype(str)

    return df


def read_energy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["time", "E_over_E0", "energy"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "time" not in df.columns:
        raise RuntimeError(f"{path} has no 'time' column.")

    if "E_over_E0" not in df.columns or df["E_over_E0"].isna().all():
        if "energy" not in df.columns:
            raise RuntimeError(f"{path} has neither 'E_over_E0' nor 'energy'.")
        E0 = float(df["energy"].iloc[0])
        df["E_over_E0"] = df["energy"] / (E0 if E0 != 0.0 else 1.0)

    df = df.dropna(subset=["time", "E_over_E0"]).sort_values("time")
    return df


# -----------------------------
# Selection utilities
# -----------------------------
def _isclose_series(s: pd.Series, val: float) -> pd.Series:
    return np.isclose(s.to_numpy(dtype=float), float(val))


def filter_cases(df: pd.DataFrame,
                 m: int | None = None,
                 n: int | None = None,
                 theta: float | None = None,
                 dt: float | None = None,
                 T: float | None = None,
                 theta_list: list[float] | None = None,
                 dt_list: list[float] | None = None,
                 mode_list: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    out = df.copy()

    if m is not None:
        out = out[out["m"] == m]
    if n is not None:
        out = out[out["n"] == n]
    if theta is not None:
        out = out[_isclose_series(out["theta"], theta)]
    if dt is not None:
        out = out[_isclose_series(out["dt"], dt)]
    if T is not None:
        out = out[_isclose_series(out["T"], T)]

    if theta_list is not None:
        mask = np.zeros(len(out), dtype=bool)
        for th in theta_list:
            mask |= np.isclose(out["theta"].to_numpy(dtype=float), float(th))
        out = out[mask]

    if dt_list is not None:
        mask = np.zeros(len(out), dtype=bool)
        for d in dt_list:
            mask |= np.isclose(out["dt"].to_numpy(dtype=float), float(d))
        out = out[mask]

    if mode_list is not None:
        keep = set((int(mm), int(nn)) for mm, nn in mode_list)
        out = out[out.apply(lambda r: (int(r["m"]), int(r["n"])) in keep, axis=1)]

    return out


# -----------------------------
# Plotting: overlay energy curves
# -----------------------------
def overlay_energy(summary_sel: pd.DataFrame,
                   out_png: Path,
                   title: str,
                   legend_key: str,
                   logy: bool = True) -> None:
    """
    legend_key: "theta" or "dt" or "mode"
    """
    if summary_sel.empty:
        raise RuntimeError("No rows selected for overlay plot.")

    out_png.parent.mkdir(parents=True, exist_ok=True)

    # stable ordering
    if legend_key == "theta":
        summary_sel = summary_sel.sort_values(["theta", "dt", "m", "n"])
    elif legend_key == "dt":
        summary_sel = summary_sel.sort_values(["dt", "theta", "m", "n"])
    elif legend_key == "mode":
        summary_sel = summary_sel.sort_values(["m", "n", "theta", "dt"])
    else:
        summary_sel = summary_sel.sort_values(["theta", "dt", "m", "n"])

    plt.figure()

    for _, row in summary_sel.iterrows():
        p = Path(row["energy_csv"])
        dfE = read_energy(p)

        if legend_key == "theta":
            label = f"theta={row['theta']:.3g}"
        elif legend_key == "dt":
            label = f"dt={row['dt']:.3g}"
        elif legend_key == "mode":
            label = f"(m,n)=({int(row['m'])},{int(row['n'])})"
        else:
            label = f"theta={row['theta']:.3g}, dt={row['dt']:.3g}, (m,n)=({int(row['m'])},{int(row['n'])})"

        plt.plot(dfE["time"].to_numpy(), dfE["E_over_E0"].to_numpy(), label=label)

    plt.xlabel("time t")
    plt.ylabel("E(t)/E(0)")
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.legend()

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Plotting: decay-rate curves
# -----------------------------
def plot_decay_vs_dt(df: pd.DataFrame, out_png: Path, title: str) -> None:
    if df.empty:
        raise RuntimeError("No rows selected for decay-vs-dt plot.")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("dt")
    plt.figure()
    plt.plot(df["dt"].to_numpy(), df["decay_rate"].to_numpy(), marker="o")
    plt.xlabel("dt")
    plt.ylabel(r"decay rate $\gamma$")
    plt.title(title)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_decay_vs_theta(df: pd.DataFrame, out_png: Path, title: str) -> None:
    if df.empty:
        raise RuntimeError("No rows selected for decay-vs-theta plot.")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("theta")
    plt.figure()
    plt.plot(df["theta"].to_numpy(), df["decay_rate"].to_numpy(), marker="o")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"decay rate $\gamma$")
    plt.title(title)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_decay_vs_frequency(df: pd.DataFrame, out_png: Path, title: str,
                            xaxis: str = "omega_dt") -> None:
    if df.empty:
        raise RuntimeError("No rows selected for decay-vs-frequency plot.")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    if xaxis == "omega":
        x = df["omega"].to_numpy()
        xlabel = r"$\omega$"
    elif xaxis == "omega2":
        x = (df["omega"].to_numpy()) ** 2
        xlabel = r"$\omega^2$"
    else:
        x = df["omega_dt"].to_numpy()
        xlabel = r"$\omega\,dt$"

    order = np.argsort(x)
    plt.figure()
    plt.plot(x[order], df["decay_rate"].to_numpy()[order], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(r"decay rate $\gamma$")
    plt.title(title)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main: configure once
# -----------------------------
if __name__ == "__main__":
    base = Path("../build/results/dissipation")
    summary_csv = base / "dissipation_summary.csv"
    plots_dir = Path("./Plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    S = read_summary(summary_csv)

    # ---- User config (edit these) ----
    T_fixed = 10.0
    mode_fixed = (1, 1)
    dt_fixed = 0.025
    theta_fixed = 0.75

    theta_sweep = [0.5, 0.75, 1.0]
    dt_sweep = [0.1, 0.05, 0.025]
    mode_sweep = [(1, 1), (2, 2), (4, 4)]
    # ---------------------------------

    m0, n0 = mode_fixed

    # 1) Overlay theta: fixed (m,n), fixed dt, fixed T
    df1 = filter_cases(S, m=m0, n=n0, dt=dt_fixed, T=T_fixed, theta_list=theta_sweep)
    overlay_energy(
        df1,
        plots_dir / "energy_overlay_theta.png",
        title=f"Energy curves (vary theta), (m,n)=({m0},{n0}), dt={dt_fixed}, T={T_fixed}",
        legend_key="theta",
        logy=True
    )

    # 2) Overlay dt: fixed (m,n), fixed theta, fixed T
    df2 = filter_cases(S, m=m0, n=n0, theta=theta_fixed, T=T_fixed, dt_list=dt_sweep)
    overlay_energy(
        df2,
        plots_dir / "energy_overlay_dt.png",
        title=f"Energy curves (vary dt), (m,n)=({m0},{n0}), theta={theta_fixed}, T={T_fixed}",
        legend_key="dt",
        logy=True
    )

    # 3) Overlay modes: fixed dt, fixed theta, fixed T
    df3 = filter_cases(S, theta=theta_fixed, dt=dt_fixed, T=T_fixed, mode_list=mode_sweep)
    overlay_energy(
        df3,
        plots_dir / "energy_overlay_modes.png",
        title=f"Energy curves (vary modes), theta={theta_fixed}, dt={dt_fixed}, T={T_fixed}",
        legend_key="mode",
        logy=True
    )

    # Optional summary plots (rates)
    df_dt = filter_cases(S, m=m0, n=n0, theta=theta_fixed, T=T_fixed)
    plot_decay_vs_dt(
        df_dt,
        plots_dir / "decay_vs_dt.png",
        title=f"Decay rate vs dt, (m,n)=({m0},{n0}), theta={theta_fixed}, T={T_fixed}"
    )

    df_th = filter_cases(S, m=m0, n=n0, dt=dt_fixed, T=T_fixed)
    plot_decay_vs_theta(
        df_th,
        plots_dir / "decay_vs_theta.png",
        title=f"Decay rate vs theta, (m,n)=({m0},{n0}), dt={dt_fixed}, T={T_fixed}"
    )

    df_fr = filter_cases(S, theta=theta_fixed, dt=dt_fixed, T=T_fixed)
    plot_decay_vs_frequency(
        df_fr,
        plots_dir / "decay_vs_frequency.png",
        title=f"Decay rate vs frequency, theta={theta_fixed}, dt={dt_fixed}, T={T_fixed}",
        xaxis="omega_dt"
    )

    print(f"Plots written to: {plots_dir.resolve()}")
