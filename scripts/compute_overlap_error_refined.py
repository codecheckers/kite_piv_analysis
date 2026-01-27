"""
Refined overlap error analysis focusing on u-velocity and per-plane breakdown.

Addresses reviewer comment Line 160 with focus on:
1. u-velocity component (primary flow direction)
2. Per-plane breakdown (Y1-Y7) to show increasing uncertainty toward tip
3. Separating systematic bias from random scatter
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from utils import project_dir

# Import functions from main script
from compute_overlap_error import (
    load_dat_file_with_std,
    load_dat_file,
    interpolate_to_grid,
)


def analyze_refined():
    """
    Refined analysis focusing on u-velocity per Y-plane.
    """
    input_dir = Path(project_dir) / "ALL_ERIK_FILES" / "JelleStitching" / "Input"
    output_dir = Path(project_dir) / "results" / "overlap_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    U_inf = 15.0  # Freestream velocity
    std_threshold = 0.1  # Filter threshold
    shift_x2 = 0.300  # m
    shift_x3 = 0.600  # m

    all_results = []

    for aoa in [13, 23]:
        aoa_dir = input_dir / f"aoa_{aoa}"
        if not aoa_dir.exists():
            continue

        y_planes = sorted([int(d.name[1:]) for d in aoa_dir.iterdir() if d.is_dir()])

        for y_plane in y_planes:
            base_path = aoa_dir / f"Y{y_plane}"

            # Find plane directories
            plane_dirs = {}
            for item in base_path.iterdir():
                if item.is_dir() and "normal" in item.name:
                    if "_X1" in item.name:
                        plane_dirs["X1"] = item
                    elif "_X2" in item.name:
                        plane_dirs["X2"] = item
                    elif "_X3" in item.name:
                        plane_dirs["X3"] = item

            if len(plane_dirs) < 2:
                continue

            # Load data from each plane with std filtering
            planes_data = {}
            for plane_name, plane_dir in plane_dirs.items():
                mean_file = plane_dir / "B0001.dat"
                std_file = plane_dir / "B0002.dat"
                if mean_file.exists() and std_file.exists():
                    planes_data[plane_name] = load_dat_file_with_std(
                        mean_file, std_file, U_inf=U_inf, std_threshold=std_threshold
                    )

            if len(planes_data) < 2:
                continue

            # Apply shifts
            if "X2" in planes_data:
                planes_data["X2"]["x"] = planes_data["X2"]["x"] + shift_x2
            if "X3" in planes_data:
                planes_data["X3"]["x"] = planes_data["X3"]["x"] + shift_x3

            # Create common grid
            all_x = np.concatenate([p["x"] for p in planes_data.values()])
            all_y = np.concatenate([p["y"] for p in planes_data.values()])

            dx, dy = 0.005, 0.005
            x_min, x_max = np.min(all_x), np.max(all_x)
            y_min, y_max = np.min(all_y), np.max(all_y)

            grid_x_1d = np.arange(x_min, x_max + dx, dx)
            grid_y_1d = np.arange(y_min, y_max + dy, dy)
            grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

            # Interpolate each plane
            interpolated = {}
            for plane_name, data in planes_data.items():
                interpolated[plane_name] = interpolate_to_grid(data, grid_x, grid_y)

            # Compute statistics for each plane pair (focus on u)
            plane_pairs = [("X1", "X2"), ("X2", "X3")]

            for p1, p2 in plane_pairs:
                if p1 not in interpolated or p2 not in interpolated:
                    continue

                u1 = interpolated[p1]["u"]
                u2 = interpolated[p2]["u"]

                # Find overlap
                overlap_mask = ~np.isnan(u1) & ~np.isnan(u2)
                n_overlap = np.sum(overlap_mask)

                if n_overlap < 10:
                    continue

                diff = u1[overlap_mask] - u2[overlap_mask]
                abs_diff = np.abs(diff)

                # Systematic bias (mean difference)
                bias = np.mean(diff)
                # Random scatter (std around the bias)
                scatter = np.std(diff)
                # RMS (total error)
                rms = np.sqrt(np.mean(diff**2))

                all_results.append(
                    {
                        "aoa": aoa,
                        "y_plane": y_plane,
                        "plane_pair": f"{p1}-{p2}",
                        "n_overlap": n_overlap,
                        "bias_m_s": bias,
                        "bias_pct_Uinf": bias / U_inf * 100,
                        "scatter_m_s": scatter,
                        "scatter_pct_Uinf": scatter / U_inf * 100,
                        "rms_m_s": rms,
                        "rms_pct_Uinf": rms / U_inf * 100,
                        "mean_abs_diff_m_s": np.mean(abs_diff),
                        "mean_abs_diff_pct_Uinf": np.mean(abs_diff) / U_inf * 100,
                        "max_abs_diff_m_s": np.max(abs_diff),
                    }
                )

    df = pd.DataFrame(all_results)

    # Save detailed results
    csv_path = output_dir / "overlap_u_refined.csv"
    df.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"Detailed results saved to: {csv_path}")

    # Create summary by Y-plane (averaging over plane pairs and aoa)
    print("\n" + "=" * 80)
    print("REFINED ANALYSIS: u-velocity overlap differences by Y-plane")
    print("=" * 80)
    print(f"Freestream velocity: U_inf = {U_inf} m/s")
    print(f"Quality filter: std/U_inf < {std_threshold}")
    print("=" * 80)

    # Summary by Y-plane
    summary_by_plane = (
        df.groupby("y_plane")
        .agg(
            {
                "bias_m_s": "mean",
                "bias_pct_Uinf": "mean",
                "scatter_m_s": "mean",
                "scatter_pct_Uinf": "mean",
                "mean_abs_diff_m_s": "mean",
                "mean_abs_diff_pct_Uinf": "mean",
                "n_overlap": "sum",
            }
        )
        .round(2)
    )

    print(
        "\n--- u-velocity: Mean values per Y-plane (averaged over X1-X2 and X2-X3) ---"
    )
    print(
        f"{'Y-plane':<10} {'Bias (m/s)':<12} {'Bias (%U∞)':<12} {'Scatter (m/s)':<14} {'Scatter (%U∞)':<14} {'|Δu| (m/s)':<12} {'|Δu| (%U∞)':<12} {'N points':<10}"
    )
    print("-" * 100)

    for y_plane in sorted(df["y_plane"].unique()):
        row = summary_by_plane.loc[y_plane]
        print(
            f"Y{y_plane:<9} {row['bias_m_s']:<12.2f} {row['bias_pct_Uinf']:<12.1f} {row['scatter_m_s']:<14.2f} {row['scatter_pct_Uinf']:<14.1f} {row['mean_abs_diff_m_s']:<12.2f} {row['mean_abs_diff_pct_Uinf']:<12.1f} {int(row['n_overlap']):<10}"
        )

    # Create table for paper (LaTeX format)
    print("\n\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)

    print(
        r"""
\begin{table}[htp]
    \caption{Overlap velocity differences for streamwise velocity component $u$ between adjacent measurement regions, reported as indication of measurement uncertainty. Bias represents systematic offset, scatter represents random variation.}
    \label{tab:overlap_error}
    \centering
    \begin{tabular}{ccccc}
    \hline
    Plane & Bias (\% $U_\infty$) & Scatter (\% $U_\infty$) & $|\Delta u|$ (\% $U_\infty$) & $N$ \\
    \hline"""
    )

    for y_plane in sorted(df["y_plane"].unique()):
        row = summary_by_plane.loc[y_plane]
        print(
            f"    $Y{y_plane}$ & {row['bias_pct_Uinf']:.1f} & {row['scatter_pct_Uinf']:.1f} & {row['mean_abs_diff_pct_Uinf']:.1f} & {int(row['n_overlap'])} \\\\"
        )

    print(
        r"""    \hline
    \end{tabular}
\end{table}
"""
    )

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY (for reviewer response)")
    print("=" * 80)

    overall_bias = df["bias_pct_Uinf"].mean()
    overall_scatter = df["scatter_pct_Uinf"].mean()
    overall_mean_abs = df["mean_abs_diff_pct_Uinf"].mean()

    y1_data = df[df["y_plane"] == 1]
    outboard_data = df[df["y_plane"] >= 5]

    print(f"\nOverall (all planes):")
    print(f"  Mean systematic bias: {overall_bias:.1f}% of U∞")
    print(f"  Mean random scatter: {overall_scatter:.1f}% of U∞")
    print(f"  Mean |Δu|: {overall_mean_abs:.1f}% of U∞")

    if len(y1_data) > 0:
        print(f"\nMid-span (Y1):")
        print(f"  Systematic bias: {y1_data['bias_pct_Uinf'].mean():.1f}% of U∞")
        print(f"  Random scatter: {y1_data['scatter_pct_Uinf'].mean():.1f}% of U∞")
        print(f"  Mean |Δu|: {y1_data['mean_abs_diff_pct_Uinf'].mean():.1f}% of U∞")

    if len(outboard_data) > 0:
        print(f"\nOutboard (Y5-Y7):")
        print(f"  Systematic bias: {outboard_data['bias_pct_Uinf'].mean():.1f}% of U∞")
        print(
            f"  Random scatter: {outboard_data['scatter_pct_Uinf'].mean():.1f}% of U∞"
        )
        print(
            f"  Mean |Δu|: {outboard_data['mean_abs_diff_pct_Uinf'].mean():.1f}% of U∞"
        )

    # Save summary to file
    summary_path = output_dir / "overlap_u_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Refined Overlap Analysis: u-velocity component\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Freestream velocity: U_inf = {U_inf} m/s\n")
        f.write(f"Quality filter: std/U_inf < {std_threshold}\n\n")
        f.write("Per Y-plane summary:\n")
        f.write(summary_by_plane.to_string())
        f.write(f"\n\nOverall mean |Δu|: {overall_mean_abs:.1f}% of U∞\n")
        if len(y1_data) > 0:
            f.write(
                f"Mid-span (Y1) mean |Δu|: {y1_data['mean_abs_diff_pct_Uinf'].mean():.1f}% of U∞\n"
            )
        if len(outboard_data) > 0:
            f.write(
                f"Outboard (Y5-Y7) mean |Δu|: {outboard_data['mean_abs_diff_pct_Uinf'].mean():.1f}% of U∞\n"
            )

    print(f"\nSummary saved to: {summary_path}")

    return df


if __name__ == "__main__":
    df = analyze_refined()
