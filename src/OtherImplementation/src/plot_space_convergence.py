#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fit_order(h, err):
    """Least-squares fit of log(err)=a + p*log(h) -> p is observed spatial order."""
    x = np.log(h)
    y = np.log(err)
    A = np.vstack([np.ones_like(x), x]).T
    a, p = np.linalg.lstsq(A, y, rcond=None)[0]
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="space_convergence.csv", help="Input CSV file.")
    parser.add_argument("--out", default="", help="Output image file (e.g., space_conv.png). If empty, just show.")
    parser.add_argument("--hcol", default="h_nom",
                        help="Which mesh-size column to use on x-axis: h_nom (recommended) or h_min.")
    parser.add_argument("--fit-last", type=int, default=4,
                        help="Fit order using the finest K points (default: 4). Use 0 to fit all points.")
    parser.add_argument("--ref-order", type=float, default=2.0,
                        help="Reference slope to plot (default: 2).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.hcol not in df.columns:
        raise ValueError(f"--hcol={args.hcol} not found. Available columns: {list(df.columns)}")

    # Sort by h descending (coarse -> fine) for clean plotting
    df = df.sort_values(args.hcol, ascending=False).reset_index(drop=True)

    h = df[args.hcol].to_numpy(dtype=float)
    eu = df["err_u"].to_numpy(dtype=float)
    ev = df["err_v"].to_numpy(dtype=float)

    # Valid entries only
    mask_u = (h > 0) & (eu > 0)
    mask_v = (h > 0) & (ev > 0)

    hu, euu = h[mask_u], eu[mask_u]
    hv, evv = h[mask_v], ev[mask_v]

    def select_last(x, y, k):
        if k is None or k <= 0 or k >= len(x):
            return x, y
        return x[-k:], y[-k:]

    # Fit observed order (by default on finest points)
    hu_fit, eu_fit = select_last(hu, euu, args.fit_last)
    hv_fit, ev_fit = select_last(hv, evv, args.fit_last)

    p_u = fit_order(hu_fit, eu_fit) if len(hu_fit) >= 2 else float("nan")
    p_v = fit_order(hv_fit, ev_fit) if len(hv_fit) >= 2 else float("nan")

    # Reference slope line anchored at the finest h point (smallest h)
    # Use u's finest point if available, else fall back to v.
    if len(hu) > 0:
        h_ref = hu[-1]
        e_ref = euu[-1]
    elif len(hv) > 0:
        h_ref = hv[-1]
        e_ref = evv[-1]
    else:
        raise ValueError("No positive h and error values to plot.")

    ref = e_ref * (h / h_ref) ** args.ref_order

    plt.figure()
    plt.loglog(h, eu, marker="o", linestyle="-", label=f"||u-ue||_L2 (fit p≈{p_u:.2f})")
    plt.loglog(h, ev, marker="s", linestyle="-", label=f"||v-ve||_L2 (fit p≈{p_v:.2f})")
    plt.loglog(h, ref, linestyle="--", label=f"reference slope {args.ref_order:g}")

    # Many people prefer finer meshes to the right (smaller h to the right)
    plt.gca().invert_xaxis()

    plt.xlabel(args.hcol)
    plt.ylabel("L2 error at T")
    plt.title("Spatial convergence (log-log)")
    plt.grid(True, which="both")
    plt.legend()

    if args.out:
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Wrote: {args.out}")
    else:
        plt.show()

    print("\nData:")
    print(df.to_string(index=False))

    # Also print what was used for fitting (helpful for reports)
    if args.fit_last and args.fit_last > 0:
        print(f"\nFit used the finest {args.fit_last} levels (when available).")
    else:
        print("\nFit used all levels.")


if __name__ == "__main__":
    main()
