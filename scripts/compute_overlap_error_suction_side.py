"""
Overlap error analysis for FLIPPED (suction side) data with upper/lower separation.

Analyzes the horizontal (X) overlap between X1-X2 and X2-X3 planes.
Separates analysis into upper q% and lower (1-q)% based on y-coordinate.

NOTE: The y-coordinate is FLIPPED (negated) to match the MATLAB stitching code,
so that "upper" corresponds to regions far from the airfoil surface (freestream)
and "lower" corresponds to regions close to the airfoil surface.

Addresses reviewer comment Line 160.
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
from plot_styling import set_plot_style


def analyze_suction_side_overlap(q=0.5):
    """
    Analyze overlap differences for flipped (suction side) data.

    Parameters:
        q: Fraction for y-coordinate split (default 0.5 = 50%)
           Upper q% of y-values (far from surface) vs lower (1-q)% (close to surface)

    Note: y-coordinates are flipped (negated) like in MATLAB stitching code,
    so "upper" = higher y after flip = further from airfoil surface.
    """
    input_dir = Path(project_dir) / "ALL_ERIK_FILES" / "JelleStitching" / "Input"
    output_dir = Path(project_dir) / "results" / "overlap_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    U_inf = 15.0  # Freestream velocity
    std_threshold = 0.1  # Filter threshold
    shift_x2 = 0.300  # m
    shift_x3 = 0.600  # m

    all_results = []
    y1_data_for_plot = None  # Store Y1 data for plotting

    for aoa in [13]:  # Focus on aoa=13 as primary case
        aoa_dir = input_dir / f"aoa_{aoa}"
        if not aoa_dir.exists():
            continue

        y_planes = sorted([int(d.name[1:]) for d in aoa_dir.iterdir() if d.is_dir()])

        for y_plane in y_planes:
            base_path = aoa_dir / f"Y{y_plane}"

            # Find FLIPPED plane directories (suction side)
            plane_dirs = {}
            for item in base_path.iterdir():
                if item.is_dir() and "flipped" in item.name:
                    if "_X1" in item.name:
                        plane_dirs["X1"] = item
                    elif "_X2" in item.name:
                        plane_dirs["X2"] = item
                    elif "_X3" in item.name:
                        plane_dirs["X3"] = item

            if len(plane_dirs) < 2:
                print(f"  Y{y_plane}: Not enough flipped planes found, skipping")
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
                    n_valid = np.sum(planes_data[plane_name]["is_valid"])
                    n_total = len(planes_data[plane_name]["x"])
                    print(
                        f"  Y{y_plane} {plane_name}: {n_valid}/{n_total} valid points"
                    )

            if len(planes_data) < 2:
                continue

            # Apply x-shifts to X2 and X3
            if "X2" in planes_data:
                planes_data["X2"]["x"] = planes_data["X2"]["x"] + shift_x2
            if "X3" in planes_data:
                planes_data["X3"]["x"] = planes_data["X3"]["x"] + shift_x3

            # FLIP y-coordinates (like MATLAB: stdata.flipped.stitched.d2y = -stdata.flipped.stitched.d2y)
            # This makes "upper" = far from surface, "lower" = close to surface
            for plane_name in planes_data:
                planes_data[plane_name]["y"] = -planes_data[plane_name]["y"]

            # Create common grid (after y-flip)
            all_x = np.concatenate([p["x"] for p in planes_data.values()])
            all_y = np.concatenate([p["y"] for p in planes_data.values()])

            dx, dy = 0.005, 0.005
            x_min, x_max = np.min(all_x), np.max(all_x)
            y_min, y_max = np.min(all_y), np.max(all_y)

            grid_x_1d = np.arange(x_min, x_max + dx, dx)
            grid_y_1d = np.arange(y_min, y_max + dy, dy)
            grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

            # Compute y-threshold for upper/lower split
            # After flip: higher y = further from surface (upper/freestream region)
            #             lower y = closer to surface (lower/boundary layer region)
            y_threshold = y_min + q * (y_max - y_min)

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

                # Create upper/lower masks based on y-coordinate (after flip)
                # upper = higher y = far from surface (freestream region)
                # lower = lower y = close to surface (near-wall/boundary layer region)
                upper_mask = overlap_mask & (grid_y >= y_threshold)
                lower_mask = overlap_mask & (grid_y < y_threshold)

                # Analyze each region
                for region_name, region_mask in [
                    ("upper", upper_mask),
                    ("lower", lower_mask),
                ]:
                    n_overlap = np.sum(region_mask)

                    if n_overlap < 10:
                        continue

                    diff = u1[region_mask] - u2[region_mask]
                    abs_diff = np.abs(diff)

                    # Systematic bias (mean difference)
                    bias = np.mean(diff)
                    # Random scatter (std around the bias)
                    scatter = np.std(diff)
                    # Mean absolute difference
                    mean_abs = np.mean(abs_diff)

                    all_results.append(
                        {
                            "aoa": aoa,
                            "y_plane": y_plane,
                            "plane_pair": f"{p1}-{p2}",
                            "region": region_name,
                            "n_overlap": n_overlap,
                            "bias_m_s": bias,
                            "bias_pct_Uinf": bias / U_inf * 100,
                            "scatter_m_s": scatter,
                            "scatter_pct_Uinf": scatter / U_inf * 100,
                            "mean_abs_diff_m_s": mean_abs,
                            "mean_abs_diff_pct_Uinf": mean_abs / U_inf * 100,
                            "max_abs_diff_m_s": np.max(abs_diff),
                        }
                    )

            # Store Y1 data for plotting
            if y_plane == 1:
                y1_data_for_plot = {
                    "interpolated": interpolated,
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "y_threshold": y_threshold,
                    "y_min": y_min,
                    "y_max": y_max,
                }

    df = pd.DataFrame(all_results)

    if df.empty:
        print("No results found!")
        return None, None

    # Save detailed results
    csv_path = output_dir / "overlap_u_suction_side.csv"
    df.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"\nDetailed results saved to: {csv_path}")

    # Print summary
    print("\n" + "=" * 100)
    print(
        f"SUCTION SIDE (flipped) OVERLAP ANALYSIS: u-velocity with upper/lower split (q={q:.0%})"
    )
    print("=" * 100)
    print(f"Freestream velocity: U_inf = {U_inf} m/s")
    print(f"Quality filter: std/U_inf < {std_threshold}")
    print(f"Y-coordinate FLIPPED (negated) to match MATLAB stitching convention")
    print(f"  Upper {q:.0%} = far from airfoil surface (freestream region)")
    print(f"  Lower {(1-q):.0%} = close to airfoil surface (boundary layer region)")
    print("=" * 100)

    # Summary by Y-plane and region
    print(
        f"\n{'Y-plane':<8} {'Region':<8} {'Bias (%U∞)':<12} {'Scatter (%U∞)':<14} {'|Δu| (%U∞)':<12} {'N points':<10}"
    )
    print("-" * 70)

    for y_plane in sorted(df["y_plane"].unique()):
        for region in ["lower", "upper"]:
            mask = (df["y_plane"] == y_plane) & (df["region"] == region)
            if mask.sum() == 0:
                continue
            subset = df[mask]
            bias = subset["bias_pct_Uinf"].mean()
            scatter = subset["scatter_pct_Uinf"].mean()
            mean_abs = subset["mean_abs_diff_pct_Uinf"].mean()
            n_pts = subset["n_overlap"].sum()
            print(
                f"Y{y_plane:<7} {region:<8} {bias:<12.1f} {scatter:<14.1f} {mean_abs:<12.1f} {int(n_pts):<10}"
            )

    # Overall summary by region
    print("\n" + "-" * 70)
    print("OVERALL SUMMARY BY REGION (all Y-planes combined)")
    print("-" * 70)

    for region in ["lower", "upper"]:
        mask = df["region"] == region
        if mask.sum() == 0:
            continue
        subset = df[mask]
        bias = subset["bias_pct_Uinf"].mean()
        scatter = subset["scatter_pct_Uinf"].mean()
        mean_abs = subset["mean_abs_diff_pct_Uinf"].mean()
        n_pts = subset["n_overlap"].sum()
        region_desc = "close to surface" if region == "lower" else "far from surface"
        print(
            f"{region.upper():<8} ({region_desc}): Bias: {bias:+.1f}% U∞  |  Scatter: {scatter:.1f}% U∞  |  |Δu|: {mean_abs:.1f}% U∞  |  N={int(n_pts)}"
        )

    # Create LaTeX table
    print("\n\n" + "=" * 100)
    print("LATEX TABLE FOR PAPER")
    print("=" * 100)

    print(
        r"""
\begin{table}[htp]
    \caption{Overlap velocity differences for streamwise velocity component $u$ between adjacent measurement regions on the suction side. Lower region is close to the airfoil surface, upper region is towards the freestream.}
    \label{tab:overlap_error_suction}
    \centering
    \begin{tabular}{cccccc}
    \hline
    Plane & Region & Bias (\% $U_\infty$) & Scatter (\% $U_\infty$) & $|\Delta u|$ (\% $U_\infty$) & $N$ \\
    \hline"""
    )

    for y_plane in sorted(df["y_plane"].unique()):
        for region in ["lower", "upper"]:
            mask = (df["y_plane"] == y_plane) & (df["region"] == region)
            if mask.sum() == 0:
                continue
            subset = df[mask]
            bias = subset["bias_pct_Uinf"].mean()
            scatter = subset["scatter_pct_Uinf"].mean()
            mean_abs = subset["mean_abs_diff_pct_Uinf"].mean()
            n_pts = int(subset["n_overlap"].sum())
            region_label = "Upper" if region == "upper" else "Lower"
            print(
                f"    $Y{y_plane}$ & {region_label} & {bias:+.1f} & {scatter:.1f} & {mean_abs:.1f} & {n_pts} \\\\"
            )

    print(
        r"""    \hline
    \end{tabular}
\end{table}
"""
    )

    return df, y1_data_for_plot


def plot_y1_overlap_differences(y1_data, output_dir, q=0.5):
    """
    Plot the velocity differences for Y1 plane, showing:
    - Upper region (far from surface): combined X1-X2 and X2-X3 overlap
    - Lower region (close to surface): combined X1-X2 and X2-X3 overlap

    X1-X2 and X2-X3 are combined using their proper x-shifts.
    """
    if y1_data is None:
        print("No Y1 data available for plotting")
        return

    set_plot_style()

    interpolated = y1_data["interpolated"]
    grid_x = y1_data["grid_x"]
    grid_y = y1_data["grid_y"]
    y_threshold = y1_data["y_threshold"]

    U_inf = 15.0

    # Check available planes
    has_x1_x2 = "X1" in interpolated and "X2" in interpolated
    has_x2_x3 = "X2" in interpolated and "X3" in interpolated

    if not has_x1_x2 and not has_x2_x3:
        print("Missing plane data for Y1 plot")
        return

    # X-shift for combining X1-X2 and X2-X3 overlaps into one plot
    # NOTE: X-shifts are already applied during data loading (see lines 94-97)
    # X2 data is shifted by +300mm and X3 by +600mm BEFORE creating the common grid
    # So all data is already in global coordinates - no additional shifts needed for plotting!

    # Compute differences for both overlaps
    diff_x1x2 = None
    diff_x2x3 = None

    if has_x1_x2:
        u1 = interpolated["X1"]["u"]
        u2 = interpolated["X2"]["u"]
        overlap_x1x2 = ~np.isnan(u1) & ~np.isnan(u2)
        diff_x1x2 = np.full_like(u1, np.nan)
        diff_x1x2[overlap_x1x2] = u1[overlap_x1x2] - u2[overlap_x1x2]

    if has_x2_x3:
        u2 = interpolated["X2"]["u"]
        u3 = interpolated["X3"]["u"]
        overlap_x2x3 = ~np.isnan(u2) & ~np.isnan(u3)
        diff_x2x3 = np.full_like(u2, np.nan)
        diff_x2x3[overlap_x2x3] = u2[overlap_x2x3] - u3[overlap_x2x3]

    # Find common color scale
    all_diffs = []
    if diff_x1x2 is not None:
        all_diffs.extend(diff_x1x2[~np.isnan(diff_x1x2)])
    if diff_x2x3 is not None:
        all_diffs.extend(diff_x2x3[~np.isnan(diff_x2x3)])

    if len(all_diffs) > 0:
        vmax = np.percentile(
            np.abs(all_diffs), 95
        )  # Use 95th percentile for better visualization
        vmax = max(vmax, 1.0)  # At least 1 m/s
    else:
        vmax = 5.0

    # Masks for upper/lower regions
    upper_mask = grid_y >= y_threshold
    lower_mask = grid_y < y_threshold

    # Create figure: single plot combining upper and lower
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Set dark grey background
    ax.set_facecolor("darkgrey")

    # First, plot the ENTIRE grid coverage of X1, X2, X3 in purple (as background)
    # This shows where data exists before plotting the overlap differences on top
    # All data is already in global coordinates (shifts applied during loading)

    if "X1" in interpolated:
        u1_valid = ~np.isnan(interpolated["X1"]["u"])
        ax.pcolormesh(
            grid_x,
            grid_y,
            np.where(u1_valid, 1.0, np.nan),
            cmap="Purples",
            vmin=0,
            vmax=2,
            shading="auto",
            alpha=0.5,
        )

    if "X2" in interpolated:
        u2_valid = ~np.isnan(interpolated["X2"]["u"])
        ax.pcolormesh(
            grid_x,
            grid_y,
            np.where(u2_valid, 1.0, np.nan),
            cmap="Purples",
            vmin=0,
            vmax=2,
            shading="auto",
            alpha=0.5,
        )

    if "X3" in interpolated:
        u3_valid = ~np.isnan(interpolated["X3"]["u"])
        ax.pcolormesh(
            grid_x,
            grid_y,
            np.where(u3_valid, 1.0, np.nan),
            cmap="Purples",
            vmin=0,
            vmax=2,
            shading="auto",
            alpha=0.5,
        )

    # Now plot the overlap differences on top (all on the same grid_x, grid_y)
    im = None

    # Plot X1-X2 overlap
    if diff_x1x2 is not None:
        im = ax.pcolormesh(
            grid_x,
            grid_y,
            diff_x1x2,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
        )

    # Plot X2-X3 overlap (same grid - data already in global coordinates)
    if diff_x2x3 is not None:
        im = ax.pcolormesh(
            grid_x,
            grid_y,
            diff_x2x3,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
        )

    # Add separation line with legend
    ax.axhline(
        y=y_threshold,
        color="k",
        linestyle="--",
        linewidth=2,
        label=f"Upper/Lower separation (y = {y_threshold:.3f} m)",
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m) [flipped]")

    # Calculate statistics for upper region
    upper_vals = []
    if diff_x1x2 is not None:
        vals = diff_x1x2[upper_mask & ~np.isnan(diff_x1x2)]
        upper_vals.extend(vals)
    if diff_x2x3 is not None:
        vals = diff_x2x3[upper_mask & ~np.isnan(diff_x2x3)]
        upper_vals.extend(vals)

    n_upper = len(upper_vals)
    bias_upper = np.mean(upper_vals) if n_upper > 0 else 0
    scatter_upper = np.std(upper_vals) if n_upper > 0 else 0
    mean_abs_upper = np.mean(np.abs(upper_vals)) if n_upper > 0 else 0

    # Calculate statistics for lower region
    lower_vals = []
    if diff_x1x2 is not None:
        vals = diff_x1x2[lower_mask & ~np.isnan(diff_x1x2)]
        lower_vals.extend(vals)
    if diff_x2x3 is not None:
        vals = diff_x2x3[lower_mask & ~np.isnan(diff_x2x3)]
        lower_vals.extend(vals)

    n_lower = len(lower_vals)
    bias_lower = np.mean(lower_vals) if n_lower > 0 else 0
    scatter_lower = np.std(lower_vals) if n_lower > 0 else 0
    mean_abs_lower = np.mean(np.abs(lower_vals)) if n_lower > 0 else 0

    # Add text boxes with statistics
    upper_text = (
        f"Upper (far from surface):\n"
        f"Bias = {bias_upper/U_inf*100:+.1f}% U$_\\infty$\n"
        f"Scatter = {scatter_upper/U_inf*100:.1f}% U$_\\infty$\n"
        f"|$\\Delta$u| = {mean_abs_upper/U_inf*100:.1f}% U$_\\infty$\n"
        f"N = {n_upper}"
    )
    lower_text = (
        f"Lower (close to surface):\n"
        f"Bias = {bias_lower/U_inf*100:+.1f}% U$_\\infty$\n"
        f"Scatter = {scatter_lower/U_inf*100:.1f}% U$_\\infty$\n"
        f"|$\\Delta$u| = {mean_abs_lower/U_inf*100:.1f}% U$_\\infty$\n"
        f"N = {n_lower}"
    )

    # Position text boxes
    bbox_props = dict(
        boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="black"
    )
    ax.text(
        0.02,
        0.98,
        upper_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=bbox_props,
    )
    ax.text(
        0.02,
        0.02,
        lower_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=bbox_props,
    )

    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10)

    # Force equal tick spacing on both axes
    from matplotlib.ticker import MultipleLocator

    tick_spacing = 0.05  # 50mm tick spacing
    ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))

    if im is not None:
        plt.colorbar(im, ax=ax, label=r"$\Delta u$ (m/s)")

    plt.title(
        f"Y1 Suction Side: Overlap velocity differences (X1-X2 + X2-X3 combined)\n"
        f"Split at {q:.0%} of y-range (y-coord flipped: upper = far from surface)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save figure
    save_path = output_dir / "overlap_Y1_suction_upper_lower.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")

    return fig


def main():
    """Main function."""
    q = 0.6  # Split at 50% of y-range

    print("=" * 100)
    print("SUCTION SIDE (FLIPPED) OVERLAP ANALYSIS")
    print("=" * 100)
    print(f"Analyzing horizontal (X) overlap with upper/lower split at q={q:.0%}")
    print("Y-coordinates are FLIPPED to match MATLAB convention:")
    print("  - Upper = far from airfoil surface (freestream region)")
    print("  - Lower = close to airfoil surface (boundary layer region)")
    print("=" * 100)

    df, y1_data = analyze_suction_side_overlap(q=q)

    if y1_data is not None:
        output_dir = Path(project_dir) / "results" / "overlap_analysis"
        plot_y1_overlap_differences(y1_data, output_dir, q=q)

    # Save summary
    if df is not None:
        output_dir = Path(project_dir) / "results" / "overlap_analysis"
        summary_path = output_dir / "overlap_u_suction_side_summary.txt"

        with open(summary_path, "w") as f:
            f.write("Suction Side (Flipped) Overlap Analysis\n")
            f.write("=" * 60 + "\n\n")
            f.write("Y-coordinates FLIPPED to match MATLAB stitching convention.\n")
            f.write(f"Upper {q:.0%} = far from airfoil surface (freestream)\n")
            f.write(
                f"Lower {(1-q):.0%} = close to airfoil surface (boundary layer)\n\n"
            )

            for region in ["lower", "upper"]:
                mask = df["region"] == region
                if mask.sum() == 0:
                    continue
                subset = df[mask]
                bias = subset["bias_pct_Uinf"].mean()
                scatter = subset["scatter_pct_Uinf"].mean()
                mean_abs = subset["mean_abs_diff_pct_Uinf"].mean()
                n_pts = subset["n_overlap"].sum()
                region_desc = (
                    "close to surface" if region == "lower" else "far from surface"
                )
                f.write(f"{region.upper()} region ({region_desc}):\n")
                f.write(f"  Bias: {bias:+.1f}% U∞\n")
                f.write(f"  Scatter: {scatter:.1f}% U∞\n")
                f.write(f"  Mean |Δu|: {mean_abs:.1f}% U∞\n")
                f.write(f"  N points: {int(n_pts)}\n\n")

        print(f"\nSummary saved to: {summary_path}")

    return df


if __name__ == "__main__":
    df = main()
    plt.show()
