#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fit_order(dt, err):
    """Least-squares fit of log(err)=a + p*log(dt) -> p is observed order."""
    x = np.log(dt)
    y = np.log(err)
    A = np.vstack([np.ones_like(x), x]).T
    a, p = np.linalg.lstsq(A, y, rcond=None)[0]
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="time_convergence.csv", help="Input CSV file.")
    parser.add_argument("--out", default="", help="Output image file (e.g., time_conv.png). If empty, just show.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Basic sanity: sort by dt descending (coarse->fine) for clean plotting
    df = df.sort_values("dt", ascending=False).reset_index(drop=True)

    dt = df["dt"].to_numpy()
    eu = df["err_u"].to_numpy()
    ev = df["err_v"].to_numpy()

    # Fit observed order using all points except potential outliers if you want;
    # here we fit all levels where errors are positive.
    mask_u = eu > 0
    mask_v = ev > 0
    p_u = fit_order(dt[mask_u], eu[mask_u])
    p_v = fit_order(dt[mask_v], ev[mask_v])

    # Reference slope-2 line anchored at the finest dt point for u
    dt_ref = dt[-1]
    eu_ref = eu[-1]
    ref2 = eu_ref * (dt / dt_ref) ** 2

    plt.figure()
    plt.loglog(dt, eu, marker="o", linestyle="-", label=f"||u-ue||_L2 (fit p≈{p_u:.2f})")
    plt.loglog(dt, ev, marker="s", linestyle="-", label=f"||v-ve||_L2 (fit p≈{p_v:.2f})")
    plt.loglog(dt, ref2, linestyle="--", label="reference slope 2")

    plt.gca().invert_xaxis()  # optional: show finer dt to the right? invert_xaxis makes finer dt to the right if dt decreases
    plt.xlabel("dt")
    plt.ylabel("L2 error at T")
    plt.title("Time convergence (log-log)")
    plt.grid(True, which="both")
    plt.legend()

    if args.out:
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Wrote: {args.out}")
    else:
        plt.show()

    # Also print the table for convenience
    print("\nData:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
