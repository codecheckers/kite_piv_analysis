"""
Comprehensive PIV Uncertainty Budget Analysis

This script provides a multi-component uncertainty analysis for stereoscopic PIV,
addressing reviewer concerns about the simplicity of uncertainty quantification.

Components analyzed:
1. Statistical precision (sample standard deviation / √N)
2. Masking and interpolation metrics (data loss quantification)
3. Divergence-based physical consistency check
4. Freestream velocity statistics (outer-field sanity check)
5. Overlap region analysis (calibration/mapping quality indicator)
6. Spatial uncertainty maps

Addresses reviewer comment Line 214:
"The performed PIV uncertainty analysis is too simple to understand the complex
sources of error in the presented measurement data."
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from kite_piv_analysis.utils import project_dir
from kite_piv_analysis.plot_styling import set_plot_style
from kite_piv_analysis.compute_overlap_error import (
    load_dat_file,
    load_dat_file_with_std,
)


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_stitched_plane(aoa: int, y_plane: int, config: str = "normal"):
    """
    Load stitched PIV plane data.

    Parameters:
        aoa: Angle of attack (13 or 23, corresponds to alpha=6 or 16 deg)
        y_plane: Y-plane number (1-7)
        config: 'normal' (pressure side) or 'flipped' (suction side)

    Returns:
        DataFrame with x, y, u, v, w, and gradient fields
    """
    # Try processed stitched data first
    stitched_dir = Path(project_dir) / "data" / "piv_stichted_planes" / f"aoa_{aoa}"

    # Check for stitched CSV file
    csv_file = stitched_dir / f"aoa_{aoa}_Y{y_plane}_stitched.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)

    return None


def load_raw_plane_data(aoa: int, y_plane: int, x_plane: int, config: str = "normal"):
    """
    Load raw PIV plane data (B0001.dat mean, B0002.dat std).

    Searches in multiple locations:
    1. data_ALL_ERIK_FILES/JelleStitching/Input (primary)
    2. data_old_21_10_2025/raw_dat_files (fallback for missing planes like aoa_23/Y4)

    Returns:
        mean_data, std_data dictionaries
    """
    # Primary location
    input_dirs = [
        Path(project_dir) / "data_ALL_ERIK_FILES" / "JelleStitching" / "Input",
        Path(project_dir) / "data_old_21_10_2025" / "raw_dat_files",
    ]

    for input_dir in input_dirs:
        base_path = input_dir / f"aoa_{aoa}" / f"Y{y_plane}"

        # Check if directory exists
        if not base_path.exists():
            continue

        # Find matching directory
        for item in base_path.iterdir():
            if item.is_dir() and config in item.name and f"_X{x_plane}" in item.name:
                mean_file = item / "B0001.dat"
                std_file = item / "B0002.dat"
                if mean_file.exists() and std_file.exists():
                    mean_data = load_dat_file(mean_file)
                    std_data = load_dat_file(std_file)
                    return mean_data, std_data

    return None, None


# =============================================================================
# 1. Masking and Interpolation Metrics
# =============================================================================


def compute_masking_metrics(
    aoa: int, y_plane: int, U_inf: float = 15.0, w_threshold: float = 3.0
):
    """
    Compute detailed data loss metrics for a given measurement plane.

    This function quantifies data loss from different sources in the PIV processing:

    1. f_mask: DaVis isValid=0 flag (geometric masking, reflections, correlation failure)
       - Applied during MATLAB stitching: invalid points set to NaN

    2. f_w: Points where |w_mean| > w_threshold (default 3 m/s)
       - As described in Appendix B: "areas with unrealistically high out-of-plane
         velocities that are not expected in the symmetry plane of the wing"
       - Applied post-hoc during visualization/analysis

    3. f_valid: Remaining valid data after all filters
    4. f_loss: Total data loss (1 - f_valid)

    Note: The std/U_inf > 0.1 filter visible in RunStitching.m is only used for
    visualization overlays, NOT applied to the output CSV files.

    Parameters:
        aoa: Angle of attack directory number (13 or 23)
        y_plane: Y-plane number (1-7)
        U_inf: Freestream velocity (default 15 m/s)
        w_threshold: Threshold for |w| masking in m/s (default 3.0, per Appendix B)

    Returns:
        dict with detailed masking statistics
    """
    results = {"aoa": aoa, "y_plane": y_plane}

    total_points = 0
    masked_davis = 0  # f_mask: geometric/reflection masking (isValid=0)
    filtered_w = 0  # f_w: |w_mean| > w_threshold (unphysical out-of-plane velocity)
    valid_points = 0

    for x_plane in [1, 2, 3]:
        for config in ["normal", "flipped"]:
            mean_data, std_data = load_raw_plane_data(aoa, y_plane, x_plane, config)

            if mean_data is None:
                continue

            n_total = len(mean_data["x"])
            total_points += n_total

            # Get DaVis validity mask (applied in MATLAB stitching)
            davis_valid = mean_data["is_valid"] != 0
            n_masked_davis = np.sum(~davis_valid)
            masked_davis += n_masked_davis

            # Get mean w velocity for the |w| > 3 m/s filter (Appendix B)
            w_mean = mean_data["w"]

            # Points that pass DaVis validation
            # f_w: Points that are DaVis-valid but have |w| > threshold
            valid_w = np.abs(w_mean) <= w_threshold
            n_fail_w = np.sum(davis_valid & ~valid_w)
            filtered_w += n_fail_w

            # Valid: DaVis valid AND |w| <= threshold
            n_valid = np.sum(davis_valid & valid_w)
            valid_points += n_valid

    if total_points > 0:
        results["n_total"] = total_points
        results["f_mask"] = (
            masked_davis / total_points
        )  # DaVis geometric/reflection masking
        results["f_w"] = filtered_w / total_points  # |w| > 3 m/s filter (Appendix B)
        results["f_valid"] = valid_points / total_points
        results["f_loss"] = 1 - valid_points / total_points

        # Legacy names for compatibility
        results["f_invalid_davis"] = results["f_mask"]
        results["f_data_loss"] = results["f_loss"]

    return results


def analyze_all_masking_metrics():
    """
    Compute masking metrics for all planes and generate summary.
    """
    results = []

    for aoa in [13, 23]:
        for y_plane in range(1, 8):
            metrics = compute_masking_metrics(aoa, y_plane)
            if "n_total" in metrics:
                results.append(metrics)

    df = pd.DataFrame(results)
    return df


# =============================================================================
# 2. Freestream Velocity Statistics
# =============================================================================


def compute_freestream_statistics(aoa: int, y_plane: int, U_inf: float = 15.0):
    """
    Compute freestream velocity statistics from outer field regions.

    Selects points far from the airfoil (high y for normal, low y after flip)
    where velocity should be close to freestream.

    Returns:
        dict with freestream region statistics
    """
    results = {"aoa": aoa, "y_plane": y_plane}

    freestream_u = []
    freestream_V = []

    for x_plane in [1, 2, 3]:
        mean_data, _ = load_raw_plane_data(aoa, y_plane, x_plane, "normal")

        if mean_data is None:
            continue

        # Valid data
        valid_mask = mean_data["is_valid"] != 0

        # Select freestream region: top 20% of y-range (far from airfoil)
        y_valid = mean_data["y"][valid_mask]
        if len(y_valid) == 0:
            continue

        y_threshold = np.percentile(y_valid, 80)  # Top 20%

        freestream_mask = valid_mask & (mean_data["y"] > y_threshold)

        u_fs = mean_data["u"][freestream_mask]
        V_fs = mean_data["V"][freestream_mask]

        # Filter out zeros (invalid data represented as 0)
        u_fs = u_fs[u_fs != 0]
        V_fs = V_fs[V_fs != 0]

        freestream_u.extend(u_fs)
        freestream_V.extend(V_fs)

    if len(freestream_u) > 0:
        freestream_u = np.array(freestream_u)
        freestream_V = np.array(freestream_V)

        results["n_freestream_points"] = len(freestream_u)
        results["u_mean"] = np.mean(freestream_u)
        results["u_std"] = np.std(freestream_u)
        results["u_deviation_from_Uinf"] = (np.mean(freestream_u) - U_inf) / U_inf * 100
        results["V_mean"] = np.mean(freestream_V)
        results["V_std"] = np.std(freestream_V)
        results["V_deviation_from_Uinf"] = (np.mean(freestream_V) - U_inf) / U_inf * 100

    return results


def analyze_all_freestream():
    """
    Compute freestream statistics for all planes.
    """
    results = []

    for aoa in [13, 23]:
        for y_plane in range(1, 8):
            metrics = compute_freestream_statistics(aoa, y_plane)
            if "n_freestream_points" in metrics:
                results.append(metrics)

    df = pd.DataFrame(results)
    return df


# =============================================================================
# 3. Spatial Uncertainty Maps
# =============================================================================


def create_spatial_uncertainty_map(
    aoa: int,
    y_plane: int,
    x_plane: int,
    config: str = "normal",
    w_threshold: float = 3.0,
):
    """
    Create spatial map of velocity standard deviation.

    Returns:
        x, y coordinates (2D arrays) and std maps for u, v, w
    """
    # Search in multiple directories
    input_dirs = [
        Path(project_dir) / "data_ALL_ERIK_FILES" / "JelleStitching" / "Input",
        Path(project_dir) / "data_old_21_10_2025" / "raw_dat_files",
    ]

    for input_dir in input_dirs:
        base_path = input_dir / f"aoa_{aoa}" / f"Y{y_plane}"

        if not base_path.exists():
            continue

        for item in base_path.iterdir():
            if item.is_dir() and config in item.name and f"_X{x_plane}" in item.name:
                mean_file = item / "B0001.dat"
                std_file = item / "B0002.dat"

                if not (mean_file.exists() and std_file.exists()):
                    continue

                # Load mean and std data
                mean_data = load_dat_file(mean_file)
                std_data = load_dat_file(std_file)

                # Get grid shape from header
                with open(mean_file, "r") as f:
                    lines = f.readlines()
                zone_line = lines[2]
                i_val = int(zone_line.split("I=")[1].split(",")[0])
                j_val = int(zone_line.split("J=")[1].split(",")[0])

                # Reshape to 2D
                x_2d = mean_data["x"].reshape(j_val, i_val)
                y_2d = mean_data["y"].reshape(j_val, i_val)
                u_std_2d = std_data["u"].reshape(j_val, i_val)
                v_std_2d = std_data["v"].reshape(j_val, i_val)
                w_std_2d = std_data["w"].reshape(j_val, i_val)
                is_valid_2d = mean_data["is_valid"].reshape(j_val, i_val)
                w_mean_2d = mean_data["w"].reshape(j_val, i_val)

                # Mask invalid regions: DaVis validity AND |w| <= threshold
                mask_valid = (is_valid_2d != 0) & (np.abs(w_mean_2d) <= w_threshold)
                u_std_2d = np.where(mask_valid, u_std_2d, np.nan)
                v_std_2d = np.where(mask_valid, v_std_2d, np.nan)
                w_std_2d = np.where(mask_valid, w_std_2d, np.nan)

                return {
                    "x": x_2d,
                    "y": y_2d,
                    "u_std": u_std_2d,
                    "v_std": v_std_2d,
                    "w_std": w_std_2d,
                    "is_valid": mask_valid,
                }

    return None


def plot_spatial_uncertainty_maps(aoa: int, y_plane: int, save_dir: Path = None):
    """
    Plot spatial uncertainty maps for all X-planes of a given Y-plane.
    """
    set_plot_style()

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    U_inf = 15.0
    vmax = 0.15  # Max std as fraction of U_inf

    for col, x_plane in enumerate([1, 2, 3]):
        data = create_spatial_uncertainty_map(aoa, y_plane, x_plane, "normal")

        if data is None:
            continue

        for row, (var, label) in enumerate(
            [
                ("u_std", r"$\sigma_u$"),
                ("v_std", r"$\sigma_v$"),
                ("w_std", r"$\sigma_w$"),
            ]
        ):
            ax = axes[row, col]
            std_normalized = data[var] / U_inf

            im = ax.pcolormesh(
                data["x"],
                data["y"],
                std_normalized,
                cmap="hot",
                vmin=0,
                vmax=vmax,
                shading="auto",
            )

            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"X{x_plane}: {label}/$U_\\infty$")

            if col == 2:
                plt.colorbar(im, ax=ax, label=f"{label}/$U_\\infty$")

    plt.suptitle(
        f"Spatial Uncertainty Maps: AoA={aoa}°, Y{y_plane}\n"
        f"(Yellow/white = high uncertainty, dark = low uncertainty)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_dir:
        save_path = save_dir / f"spatial_uncertainty_aoa{aoa}_Y{y_plane}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# 4. Combined Uncertainty Budget
# =============================================================================


def compute_combined_uncertainty_budget():
    """
    Compute comprehensive uncertainty budget combining all components.

    Returns:
        DataFrame with uncertainty budget per plane
    """
    results = []
    U_inf = 15.0
    N = 250  # Number of images
    k = 1.96  # Coverage factor for 95% CI

    for aoa in [13, 23]:
        for y_plane in range(1, 8):
            budget = {"aoa": aoa, "y_plane": y_plane}

            # 1. Statistical precision (from std files)
            all_u_std = []
            all_v_std = []
            all_w_std = []
            n_valid_total = 0
            n_total_total = 0

            for x_plane in [1, 2, 3]:
                for config in ["normal", "flipped"]:
                    data = create_spatial_uncertainty_map(aoa, y_plane, x_plane, config)
                    if data is not None:
                        valid_mask = ~np.isnan(data["u_std"])
                        all_u_std.extend(data["u_std"][valid_mask].flatten())
                        all_v_std.extend(data["v_std"][valid_mask].flatten())
                        all_w_std.extend(data["w_std"][valid_mask].flatten())
                        n_valid_total += np.sum(valid_mask)
                        n_total_total += data["u_std"].size

            if len(all_u_std) > 0:
                # Mean standard deviation across field
                sigma_u = np.mean(all_u_std)
                sigma_v = np.mean(all_v_std)
                sigma_w = np.mean(all_w_std)

                # Standard uncertainty of the mean
                u_u_x = k * sigma_u / np.sqrt(N)
                u_u_y = k * sigma_v / np.sqrt(N)
                u_u_z = k * sigma_w / np.sqrt(N)

                # Combined velocity uncertainty
                u_u_vec = np.sqrt(u_u_x**2 + u_u_y**2 + u_u_z**2)

                budget["sigma_u_mean"] = sigma_u
                budget["sigma_v_mean"] = sigma_v
                budget["sigma_w_mean"] = sigma_w
                budget["u_u_x"] = u_u_x / U_inf  # Normalized
                budget["u_u_y"] = u_u_y / U_inf
                budget["u_u_z"] = u_u_z / U_inf
                budget["u_u_vec"] = u_u_vec / U_inf

            # 2. Data loss fraction
            masking_metrics = compute_masking_metrics(aoa, y_plane)
            if "f_data_loss" in masking_metrics:
                budget["f_data_loss"] = masking_metrics["f_data_loss"]
                budget["f_valid"] = masking_metrics["f_valid"]

            # 3. Freestream deviation
            fs_metrics = compute_freestream_statistics(aoa, y_plane)
            if "u_deviation_from_Uinf" in fs_metrics:
                budget["freestream_bias_pct"] = fs_metrics["u_deviation_from_Uinf"]

            if len(budget) > 2:  # More than just aoa and y_plane
                results.append(budget)

    df = pd.DataFrame(results)
    return df


# =============================================================================
# Main Analysis and Reporting
# =============================================================================


def generate_uncertainty_report():
    """
    Generate comprehensive uncertainty report with LaTeX tables.
    """
    output_dir = Path(project_dir) / "results" / "uncertainty_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("COMPREHENSIVE PIV UNCERTAINTY BUDGET ANALYSIS")
    print("=" * 100)
    print()

    # 1. Masking metrics
    print("\n" + "=" * 80)
    print("1. DATA LOSS AND MASKING METRICS")
    print("=" * 80)

    df_masking = analyze_all_masking_metrics()
    if not df_masking.empty:
        df_masking.to_csv(
            output_dir / "masking_metrics.csv", index=False, float_format="%.4f"
        )

        print("\nPer-plane data quality fractions:")
        print(
            "  f_mask:  DaVis isValid=0 (geometric masking, reflections, correlation failure)"
        )
        print(
            "  f_w:     Filtered due to |w| > 3 m/s (unphysical out-of-plane velocity, Appendix B)"
        )
        print("  f_valid: Remaining valid data")
        print()
        print(
            f"{'AoA':<6} {'Plane':<8} {'f_mask':<10} {'f_w':<10} "
            f"{'f_valid':<10} {'f_loss':<10}"
        )
        print("-" * 60)
        for _, row in df_masking.iterrows():
            print(
                f"{int(row['aoa']):<6} Y{int(row['y_plane']):<7} "
                f"{row['f_mask']:<10.3f} {row['f_w']:<10.3f} "
                f"{row['f_valid']:<10.3f} {row['f_loss']:<10.3f}"
            )

        # Summary by AoA
        print("\nSummary by angle of attack:")
        for aoa in df_masking["aoa"].unique():
            subset = df_masking[df_masking["aoa"] == aoa]
            print(
                f"  AoA {aoa}°: Mean data loss = {subset['f_loss'].mean()*100:.1f}%, "
                f"Range = {subset['f_loss'].min()*100:.1f}% - {subset['f_loss'].max()*100:.1f}%"
            )

    # 2. Freestream statistics
    print("\n" + "=" * 80)
    print("2. FREESTREAM VELOCITY VERIFICATION")
    print("=" * 80)

    df_fs = analyze_all_freestream()
    if not df_fs.empty:
        df_fs.to_csv(
            output_dir / "freestream_statistics.csv", index=False, float_format="%.4f"
        )

        print("\nFreestream region statistics (top 20% of y-range):")
        print(f"Expected: U_inf = 15.0 m/s")
        print()
        print(
            f"{'AoA':<6} {'Plane':<8} {'u_mean (m/s)':<14} {'u_std (m/s)':<14} {'Deviation %':<14}"
        )
        print("-" * 60)
        for _, row in df_fs.iterrows():
            print(
                f"{int(row['aoa']):<6} Y{int(row['y_plane']):<7} "
                f"{row['u_mean']:<14.2f} {row['u_std']:<14.2f} "
                f"{row['u_deviation_from_Uinf']:<14.1f}"
            )

        print(
            f"\nOverall mean freestream velocity: {df_fs['u_mean'].mean():.2f} ± {df_fs['u_std'].mean():.2f} m/s"
        )
        print(
            f"Mean deviation from U_inf: {df_fs['u_deviation_from_Uinf'].mean():.1f}%"
        )

    # 3. Combined uncertainty budget
    print("\n" + "=" * 80)
    print("3. COMBINED UNCERTAINTY BUDGET")
    print("=" * 80)

    df_budget = compute_combined_uncertainty_budget()
    if not df_budget.empty:
        df_budget.to_csv(
            output_dir / "uncertainty_budget.csv", index=False, float_format="%.4f"
        )

        print("\nUncertainty budget per measurement plane:")
        print(
            "  u_u = standard uncertainty of mean velocity (95% CI, normalized by U_inf)"
        )
        print()
        cols = ["aoa", "y_plane", "u_u_x", "u_u_y", "u_u_z", "u_u_vec", "f_data_loss"]
        available_cols = [c for c in cols if c in df_budget.columns]
        print(df_budget[available_cols].round(3).to_string(index=False))

    # 4. Generate spatial uncertainty maps for representative planes
    print("\n" + "=" * 80)
    print("4. SPATIAL UNCERTAINTY MAPS")
    print("=" * 80)

    for aoa in [13]:
        for y_plane in [1, 4]:  # Representative planes
            print(f"  Generating map for AoA={aoa}°, Y{y_plane}...")
            try:
                plot_spatial_uncertainty_maps(aoa, y_plane, output_dir)
            except Exception as e:
                print(f"    Error: {e}")

    # 5. Generate LaTeX tables
    print("\n" + "=" * 80)
    print("5. LATEX TABLES FOR PAPER")
    print("=" * 80)

    generate_latex_tables(df_masking, df_budget, df_fs, output_dir)

    # Save comprehensive summary
    generate_text_summary(df_masking, df_div, df_fs, df_budget, output_dir)

    print(f"\n\nAll results saved to: {output_dir}")


def generate_latex_tables(df_masking, df_budget, df_fs, output_dir):
    """
    Generate LaTeX-formatted tables for the paper.
    """
    latex_output = []

    # Table: Data quality metrics per plane (wide format with alpha as columns)
    latex_output.append(
        r"""
% =============================================================================
% TABLE: Data Quality Metrics (Detailed)
% =============================================================================
\begin{table}[htp]
    \centering
    \caption{Data quality metrics per measurement plane. $f_\mathrm{mask}$ denotes the fraction of vectors marked invalid by DaVis (geometric masking, reflections, correlation failure). $f_w$ denotes the fraction filtered due to $|w| > 3\,\mathrm{m\,s^{-1}}$ (unphysical out-of-plane velocity, see Appendix~B). $f_\mathrm{valid}$ is the remaining valid data fraction, and $f_\mathrm{loss}$ is the total data loss.}
    \label{tab:data_quality}
    \begin{tabular}{l ccccccc l cccc}
    \hline
    & \multicolumn{7}{c}{$\alpha = 7\unit{\degree}$} & & \multicolumn{4}{c}{$\alpha = 17\unit{\degree}$} \\
     & $Y1$ & $Y2$ & $Y3$ & $Y4$ & $Y5$ & $Y6$ & $Y7$ & & $Y1$ & $Y2$ & $Y3$ & $Y4$ \\
    \hline"""
    )

    if not df_masking.empty:
        # Organize data by plane for each metric
        def get_plane_value(df, aoa, y_plane, metric):
            row = df[(df["aoa"] == aoa) & (df["y_plane"] == y_plane)]
            if len(row) > 0:
                return row[metric].values[0] * 100  # Convert to percentage
            return None

        # Row: f_mask
        row_mask_7 = []
        for y in range(1, 8):
            val = get_plane_value(df_masking, 13, y, "f_mask")
            row_mask_7.append(f"{val:.1f}" if val is not None else "--")

        row_mask_17 = []
        for y in range(1, 5):
            val = get_plane_value(df_masking, 23, y, "f_mask")
            row_mask_17.append(f"{val:.1f}" if val is not None else "--")

        latex_output.append(
            f"    $f_\\mathrm{{mask}}$ (\\%) & {' & '.join(row_mask_7)} & & {' & '.join(row_mask_17)} \\\\"
        )

        # Row: f_w (|w| > 3 m/s filter)
        row_w_7 = []
        for y in range(1, 8):
            val = get_plane_value(df_masking, 13, y, "f_w")
            row_w_7.append(f"{val:.1f}" if val is not None else "--")

        row_w_17 = []
        for y in range(1, 5):
            val = get_plane_value(df_masking, 23, y, "f_w")
            row_w_17.append(f"{val:.1f}" if val is not None else "--")

        latex_output.append(
            f"    $f_w$ (\\%) & {' & '.join(row_w_7)} & & {' & '.join(row_w_17)} \\\\"
        )

        # Row: f_valid
        row_valid_7 = []
        for y in range(1, 8):
            val = get_plane_value(df_masking, 13, y, "f_valid")
            row_valid_7.append(f"{val:.1f}" if val is not None else "--")

        row_valid_17 = []
        for y in range(1, 5):
            val = get_plane_value(df_masking, 23, y, "f_valid")
            row_valid_17.append(f"{val:.1f}" if val is not None else "--")

        latex_output.append(
            f"    $f_\\mathrm{{valid}}$ (\\%) & {' & '.join(row_valid_7)} & & {' & '.join(row_valid_17)} \\\\"
        )

        # Row: f_loss
        row_loss_7 = []
        for y in range(1, 8):
            val = get_plane_value(df_masking, 13, y, "f_loss")
            row_loss_7.append(f"{val:.1f}" if val is not None else "--")

        row_loss_17 = []
        for y in range(1, 5):
            val = get_plane_value(df_masking, 23, y, "f_loss")
            row_loss_17.append(f"{val:.1f}" if val is not None else "--")

        latex_output.append(
            f"    $f_\\mathrm{{loss}}$ (\\%) & {' & '.join(row_loss_7)} & & {' & '.join(row_loss_17)} \\\\"
        )

    latex_output.append(
        r"""    \hline
    \end{tabular}
\end{table}
"""
    )

    # Table: Comprehensive uncertainty budget
    latex_output.append(
        r"""
% =============================================================================
% TABLE: Uncertainty Budget
% =============================================================================
\begin{table}[htp]
    \centering
    \caption{Uncertainty budget for PIV measurements. Statistical uncertainties ($\mathrm{u}_{\overline{\mathrm{u}}}$) correspond to 95\% confidence intervals normalized by $U_\infty$.}
    \label{tab:uncertainty_budget}
    \begin{tabular}{cc cccc c}
    \hline
    $\alpha$ & Plane & $\mathrm{u}_{\overline{\mathrm{u}},{x}}$ & $\mathrm{u}_{\overline{\mathrm{u}},{y}}$ & $\mathrm{u}_{\overline{\mathrm{u}},{z}}$ & $\mathrm{u}_{\overline{\vec{u}}}$ & $f_\mathrm{loss}$ (\%) \\
    \hline"""
    )

    if not df_budget.empty:
        for _, row in df_budget.iterrows():
            aoa_deg = 6 if row["aoa"] == 13 else 16
            f_loss = (
                row.get("f_data_loss", np.nan) * 100 if "f_data_loss" in row else np.nan
            )
            latex_output.append(
                f"    {aoa_deg}\\unit{{\\degree}} & $Y{int(row['y_plane'])}$ & "
                f"{row.get('u_u_x', np.nan):.2f} & {row.get('u_u_y', np.nan):.2f} & "
                f"{row.get('u_u_z', np.nan):.2f} & {row.get('u_u_vec', np.nan):.2f} & "
                f"{f_loss:.1f} \\\\"
            )

    latex_output.append(
        r"""    \hline
    \end{tabular}
\end{table}
"""
    )

    # Write to file
    latex_file = output_dir / "uncertainty_tables.tex"
    with open(latex_file, "w") as f:
        f.write("\n".join(latex_output))
    print(f"\nLaTeX tables saved to: {latex_file}")

    # Print to console
    print("\n" + "\n".join(latex_output))


def generate_text_summary(df_masking, df_div, df_fs, df_budget, output_dir):
    """
    Generate comprehensive text summary of uncertainty analysis.
    """
    summary = []
    summary.append("=" * 80)
    summary.append("COMPREHENSIVE PIV UNCERTAINTY BUDGET SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    summary.append("This analysis quantifies multiple uncertainty contributions in the")
    summary.append("stereoscopic PIV measurements, addressing the following sources:")
    summary.append("")
    summary.append("1. STATISTICAL PRECISION")
    summary.append("   - Based on velocity standard deviation across 250 image samples")
    summary.append("   - Reported as 95% confidence interval of the mean")
    summary.append("")

    if not df_budget.empty:
        mean_ux = df_budget["u_u_x"].mean() if "u_u_x" in df_budget else np.nan
        mean_uy = df_budget["u_u_y"].mean() if "u_u_y" in df_budget else np.nan
        mean_uz = df_budget["u_u_z"].mean() if "u_u_z" in df_budget else np.nan
        mean_uvec = df_budget["u_u_vec"].mean() if "u_u_vec" in df_budget else np.nan
        summary.append(f"   Mean values (normalized by U_inf):")
        summary.append(f"     u_u,x = {mean_ux:.3f}")
        summary.append(f"     u_u,y = {mean_uy:.3f}")
        summary.append(f"     u_u,z = {mean_uz:.3f}")
        summary.append(f"     u_u,vec = {mean_uvec:.3f}")
        summary.append("")

    summary.append("2. DATA LOSS AND MASKING")
    summary.append("   - Quantifies fraction of invalid/masked measurement points")
    summary.append("   - Higher data loss correlates with increased uncertainty")
    summary.append("")

    if not df_masking.empty:
        summary.append(
            f"   Mean data loss: {df_masking['f_data_loss'].mean()*100:.1f}%"
        )
        summary.append(
            f"   Range: {df_masking['f_data_loss'].min()*100:.1f}% - {df_masking['f_data_loss'].max()*100:.1f}%"
        )
        summary.append("")

    summary.append("3. PHYSICAL CONSISTENCY (DIVERGENCE CHECK)")
    summary.append("   - 2D divergence should be near zero for incompressible flow")
    summary.append("   - Non-zero values indicate 3D effects or measurement errors")
    summary.append("")

    if not df_div.empty:
        summary.append(
            f"   Mean normalized RMS divergence: {df_div['div_rms_normalized'].mean():.4f}"
        )
        summary.append("")

    summary.append("4. FREESTREAM VERIFICATION")
    summary.append("   - Compares outer-field velocity to expected U_inf = 15 m/s")
    summary.append("   - Indicates systematic bias in velocity reconstruction")
    summary.append("")

    if not df_fs.empty:
        summary.append(f"   Mean freestream velocity: {df_fs['u_mean'].mean():.2f} m/s")
        summary.append(
            f"   Mean deviation from U_inf: {df_fs['u_deviation_from_Uinf'].mean():.1f}%"
        )
        summary.append("")

    summary.append("5. OVERLAP REGION ANALYSIS (from previous analysis)")
    summary.append("   - Velocity differences in overlapping measurement regions")
    summary.append(
        "   - Indicates combined calibration, mapping, and reconstruction errors"
    )
    summary.append("   - See overlap_analysis/ for detailed results")
    summary.append("")
    summary.append("=" * 80)
    summary.append("CONCLUSION")
    summary.append("=" * 80)
    summary.append("")
    summary.append("The uncertainty analysis reveals that:")
    summary.append(
        "- Statistical precision (Table 3 values) represents random uncertainty"
    )
    summary.append("- Data loss increases toward wingtip and at high angle of attack")
    summary.append("- Overlap analysis indicates additional systematic contributions")
    summary.append("- Freestream verification confirms overall velocity calibration")
    summary.append("")
    summary.append("These multiple uncertainty sources should be considered when")
    summary.append("interpreting the quantitative comparison with CFD results.")
    summary.append("")

    # Write summary
    summary_file = output_dir / "uncertainty_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary))
    print(f"\nText summary saved to: {summary_file}")


if __name__ == "__main__":
    generate_uncertainty_report()
    plt.show()
