"""
Script to compute velocity differences in overlapping measurement regions.

This script addresses reviewer comment Line 160:
"The transitions between measurement regions were smoothed out but they are still
clearly visible in Figure 6. Please report the difference in flow velocity at the
same location for different overlapping measurements as indication of the measurement error."

The script:
1. Loads raw PIV data from individual measurement planes (X1, X2, X3)
2. Interpolates them onto a common grid
3. Identifies overlap regions
4. Computes velocity differences between overlapping measurements
5. Reports statistics (mean, max, std) as measurement error indication
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from kite_piv_analysis.utils import project_dir
from kite_piv_analysis.plot_styling import set_plot_style


def load_dat_file(filepath):
    """
    Load a PIV .dat file and return coordinates and velocities.

    Parameters:
        filepath: Path to the B0001.dat file

    Returns:
        dict with x, y, u, v, w, V, is_valid arrays
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Parse header for dimensions (I=91, J=104)
    zone_line = lines[2]
    i_val = int(zone_line.split("I=")[1].split(",")[0])
    j_val = int(zone_line.split("J=")[1].split(",")[0])

    # Load data starting from line 4 (0-indexed: line 4)
    data = []
    for line in lines[4:]:
        values = line.strip().split()
        if len(values) >= 17:
            data.append([float(v) for v in values])

    data = np.array(data)

    return {
        "x": data[:, 0] / 1000,  # Convert mm to m
        "y": data[:, 1] / 1000,  # Convert mm to m
        "u": data[:, 2],
        "v": data[:, 3],
        "w": data[:, 4],
        "V": data[:, 5],
        "is_valid": data[:, -1],
        "shape": (i_val, j_val),
    }


def load_dat_file_with_std(mean_filepath, std_filepath, U_inf=15.0, std_threshold=0.1):
    """
    Load PIV mean data (B0001.dat) and std data (B0002.dat) and filter based on std/U_inf.

    Parameters:
        mean_filepath: Path to the B0001.dat file (mean values)
        std_filepath: Path to the B0002.dat file (standard deviations)
        U_inf: Freestream velocity for normalization (default 15 m/s)
        std_threshold: Maximum allowed std/U_inf ratio (default 0.1 = 10%)

    Returns:
        dict with x, y, u, v, w, V, is_valid arrays (filtered for quality)
    """
    # Load mean data
    mean_data = load_dat_file(mean_filepath)

    # Load std data
    with open(std_filepath, "r") as f:
        lines = f.readlines()

    std_data_raw = []
    for line in lines[4:]:
        values = line.strip().split()
        if len(values) >= 17:
            std_data_raw.append([float(v) for v in values])

    std_data_raw = np.array(std_data_raw)

    # Extract standard deviations (columns 2,3,4 are u_std, v_std, w_std)
    u_std = std_data_raw[:, 2]
    v_std = std_data_raw[:, 3]
    w_std = std_data_raw[:, 4]

    # Apply quality filter: std/U_inf < threshold (like MATLAB code)
    # Filter keeps data where relative std is small (good quality)
    valid_u = (np.abs(u_std) / U_inf) < std_threshold
    valid_v = (np.abs(v_std) / U_inf) < std_threshold
    valid_w = (np.abs(w_std) / U_inf) < std_threshold

    # Combined validity: original is_valid AND passes std filter for all components
    combined_valid = (mean_data["is_valid"] != 0) & valid_u & valid_v & valid_w

    # Set invalid data to NaN
    mean_data["u"] = np.where(combined_valid, mean_data["u"], np.nan)
    mean_data["v"] = np.where(combined_valid, mean_data["v"], np.nan)
    mean_data["w"] = np.where(combined_valid, mean_data["w"], np.nan)
    mean_data["V"] = np.where(combined_valid, mean_data["V"], np.nan)
    mean_data["is_valid"] = combined_valid.astype(float)

    # Store std values for reference
    mean_data["u_std"] = u_std
    mean_data["v_std"] = v_std
    mean_data["w_std"] = w_std

    return mean_data


def interpolate_to_grid(data, grid_x, grid_y):
    """
    Interpolate data onto a common grid.

    Parameters:
        data: dict with x, y, u, v, w arrays
        grid_x, grid_y: 2D meshgrid arrays for interpolation target

    Returns:
        dict with interpolated u, v, w on the grid
    """
    # Create mask for valid data (check both is_valid flag and non-NaN values)
    valid_mask = (data["is_valid"] != 0) & ~np.isnan(data["u"])

    points = np.column_stack([data["x"][valid_mask], data["y"][valid_mask]])

    result = {}
    for var in ["u", "v", "w", "V"]:
        values = data[var][valid_mask]
        # Remove any remaining NaN values
        non_nan_mask = ~np.isnan(values)
        if np.sum(non_nan_mask) > 3:  # Need at least 3 points for interpolation
            # Use linear interpolation, returns NaN outside convex hull
            result[var] = griddata(
                points[non_nan_mask],
                values[non_nan_mask],
                (grid_x, grid_y),
                method="linear",
            )
        else:
            result[var] = np.full_like(grid_x, np.nan)

    return result


def compute_overlap_statistics(
    plane1_interp, plane2_interp, var_name, plane1_name, plane2_name
):
    """
    Compute statistics for the difference between two planes in overlap region.

    Parameters:
        plane1_interp, plane2_interp: interpolated data dicts
        var_name: variable to compare ('u', 'v', 'w', 'V')
        plane1_name, plane2_name: names for reporting

    Returns:
        dict with statistics
    """
    val1 = plane1_interp[var_name]
    val2 = plane2_interp[var_name]

    # Find overlap: both have valid (non-NaN) data
    overlap_mask = ~np.isnan(val1) & ~np.isnan(val2)
    n_overlap = np.sum(overlap_mask)

    if n_overlap == 0:
        return None

    diff = val1[overlap_mask] - val2[overlap_mask]
    abs_diff = np.abs(diff)

    # Also compute relative difference (w.r.t. mean velocity magnitude)
    mean_val = (np.abs(val1[overlap_mask]) + np.abs(val2[overlap_mask])) / 2
    # Avoid division by zero
    rel_diff = np.where(mean_val > 0.1, abs_diff / mean_val * 100, np.nan)

    return {
        "plane1": plane1_name,
        "plane2": plane2_name,
        "variable": var_name,
        "n_overlap_points": n_overlap,
        "mean_diff": np.mean(diff),
        "mean_abs_diff": np.mean(abs_diff),
        "max_abs_diff": np.max(abs_diff),
        "std_diff": np.std(diff),
        "rms_diff": np.sqrt(np.mean(diff**2)),
        "mean_rel_diff_percent": np.nanmean(rel_diff),
    }


def analyze_overlap_for_plane(input_dir, aoa, y_plane, shift_x2=0.300, shift_x3=0.600):
    """
    Analyze overlap between X1, X2, X3 measurement planes for a given Y-plane.

    Parameters:
        input_dir: Path to JelleStitching/Input directory
        aoa: Angle of attack (13 or 23)
        y_plane: Y-plane number (1-7)
        shift_x2, shift_x3: Streamwise shifts for X2 and X3 planes (m)
                            Default: 0.300m for X2, 0.600m for X3 (from planes_location.csv)

    Returns:
        List of statistics dictionaries
    """
    base_path = Path(input_dir) / f"aoa_{aoa}" / f"Y{y_plane}"

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
        print(f"  Not enough planes found for Y{y_plane}")
        return []

    # Load data from each plane (with std filtering)
    planes_data = {}
    U_inf = 15.0  # Freestream velocity
    std_threshold = 0.1  # Filter threshold: std/U_inf < 0.1 (same as MATLAB)

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
                f"  Loaded {plane_name}: {n_valid}/{n_total} valid points (after std filter)"
            )
        elif mean_file.exists():
            # Fallback to old method if no std file
            planes_data[plane_name] = load_dat_file(mean_file)
            print(
                f"  Loaded {plane_name}: {len(planes_data[plane_name]['x'])} points (no std filter)"
            )

    if len(planes_data) < 2:
        return []

    # Apply shifts to X2 and X3
    if "X2" in planes_data:
        planes_data["X2"]["x"] = planes_data["X2"]["x"] + shift_x2
    if "X3" in planes_data:
        planes_data["X3"]["x"] = planes_data["X3"]["x"] + shift_x3

    # Create common grid spanning all planes
    all_x = np.concatenate([p["x"] for p in planes_data.values()])
    all_y = np.concatenate([p["y"] for p in planes_data.values()])

    # Grid resolution (approximately match original)
    dx = 0.005  # 5mm grid spacing
    dy = 0.005

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    grid_x_1d = np.arange(x_min, x_max + dx, dx)
    grid_y_1d = np.arange(y_min, y_max + dy, dy)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

    # Interpolate each plane to common grid
    interpolated = {}
    for plane_name, data in planes_data.items():
        interpolated[plane_name] = interpolate_to_grid(data, grid_x, grid_y)

    # Compute overlap statistics for each pair
    results = []
    plane_pairs = [("X1", "X2"), ("X2", "X3"), ("X1", "X3")]

    for p1, p2 in plane_pairs:
        if p1 in interpolated and p2 in interpolated:
            for var in ["u", "v", "w", "V"]:
                stats = compute_overlap_statistics(
                    interpolated[p1], interpolated[p2], var, p1, p2
                )
                if stats:
                    stats["aoa"] = aoa
                    stats["y_plane"] = y_plane
                    results.append(stats)

    return results, interpolated, grid_x, grid_y


def plot_overlap_difference(interpolated, grid_x, grid_y, aoa, y_plane, save_dir=None):
    """
    Plot the velocity difference in overlap regions.
    """
    set_plot_style()

    if "X1" not in interpolated or "X2" not in interpolated:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, var in enumerate(["u", "v", "w", "V"]):
        ax = axes.flatten()[idx]

        val1 = interpolated["X1"][var]
        val2 = interpolated["X2"][var]

        # Find overlap
        overlap_mask = ~np.isnan(val1) & ~np.isnan(val2)

        diff = np.full_like(val1, np.nan)
        diff[overlap_mask] = val1[overlap_mask] - val2[overlap_mask]

        # Plot
        vmax = np.nanmax(np.abs(diff))
        if vmax > 0:
            im = ax.pcolormesh(
                grid_x,
                grid_y,
                diff,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
            )
            plt.colorbar(im, ax=ax, label=r"$\Delta$" + f"{var} (m/s)")

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{var}: X1 - X2 overlap difference")
        ax.set_aspect("equal")

    plt.suptitle(
        f"Velocity Differences in Overlap Region\nAoA={aoa}, Y-plane {y_plane}"
    )
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / f"overlap_diff_aoa{aoa}_Y{y_plane}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")

    return fig


def main():
    """
    Main function to compute overlap error statistics for all planes.
    """
    input_dir = Path(project_dir) / "data_ALL_ERIK_FILES" / "JelleStitching" / "Input"
    output_dir = Path(project_dir) / "results" / "overlap_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Analyze for both angles of attack
    for aoa in [13, 23]:
        print(f"\n{'='*60}")
        print(f"Analyzing AoA = {aoa}°")
        print("=" * 60)

        # Check which Y-planes exist
        aoa_dir = input_dir / f"aoa_{aoa}"
        if not aoa_dir.exists():
            print(f"  Directory not found: {aoa_dir}")
            continue

        y_planes = sorted([int(d.name[1:]) for d in aoa_dir.iterdir() if d.is_dir()])

        for y_plane in y_planes:
            print(f"\n--- Y-plane {y_plane} ---")
            try:
                results, interpolated, grid_x, grid_y = analyze_overlap_for_plane(
                    input_dir, aoa, y_plane
                )
                all_results.extend(results)

                # Generate plot for first Y-plane of each AoA
                if y_plane == y_planes[0]:
                    plot_overlap_difference(
                        interpolated, grid_x, grid_y, aoa, y_plane, output_dir
                    )

            except Exception as e:
                print(f"  Error processing Y{y_plane}: {e}")
                import traceback

                traceback.print_exc()

    # Convert to DataFrame and save
    if all_results:
        df = pd.DataFrame(all_results)

        # Freestream velocity for normalization
        U_inf = 15.0  # m/s

        # Save detailed results
        csv_path = output_dir / "overlap_velocity_differences.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n\nDetailed results saved to: {csv_path}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY: Velocity Differences in Overlapping Measurement Regions")
        print("=" * 80)

        # Group by variable and compute overall statistics
        summary = (
            df.groupby("variable")
            .agg(
                {
                    "mean_abs_diff": ["mean", "std", "max"],
                    "rms_diff": ["mean", "max"],
                    "mean_rel_diff_percent": "mean",
                    "n_overlap_points": "sum",
                }
            )
            .round(4)
        )

        print("\nOverall statistics across all planes:")
        print(summary.to_string())

        # Create summary for paper
        print("\n" + "-" * 80)
        print("FOR PAPER - Measurement Error Indication (all Y-planes combined):")
        print(f"Freestream velocity U_inf = {U_inf} m/s")
        print("-" * 80)

        for var in ["u", "v", "w", "V"]:
            var_data = df[df["variable"] == var]
            if len(var_data) > 0:
                mean_rms = var_data["rms_diff"].mean()
                mean_abs = var_data["mean_abs_diff"].mean()
                max_abs = var_data["max_abs_diff"].max()
                mean_rel = var_data["mean_rel_diff_percent"].mean()
                print(f"\n{var}-velocity:")
                print(
                    f"  Mean absolute difference: {mean_abs:.3f} m/s ({mean_abs/U_inf*100:.1f}% of U_inf)"
                )
                print(
                    f"  Mean RMS difference: {mean_rms:.3f} m/s ({mean_rms/U_inf*100:.1f}% of U_inf)"
                )
                print(
                    f"  Maximum absolute difference: {max_abs:.3f} m/s ({max_abs/U_inf*100:.1f}% of U_inf)"
                )

        # Save summary
        summary_path = output_dir / "overlap_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Velocity Differences in Overlapping Measurement Regions\n")
            f.write("=" * 60 + "\n\n")
            f.write("This analysis quantifies the measurement error by comparing\n")
            f.write("velocity values at the same spatial locations measured by\n")
            f.write("different overlapping PIV planes (X1, X2, X3).\n\n")
            f.write(f"Freestream velocity: U_inf = {U_inf} m/s\n\n")
            f.write(summary.to_string())
            f.write("\n\n")
            f.write("-" * 60 + "\n")
            f.write("Summary for paper:\n")
            f.write("-" * 60 + "\n")
            for var in ["u", "v", "w", "V"]:
                var_data = df[df["variable"] == var]
                if len(var_data) > 0:
                    mean_rms = var_data["rms_diff"].mean()
                    mean_abs = var_data["mean_abs_diff"].mean()
                    max_abs = var_data["max_abs_diff"].max()
                    f.write(f"\n{var}-velocity:\n")
                    f.write(
                        f"  Mean absolute difference: {mean_abs:.3f} m/s ({mean_abs/U_inf*100:.1f}% of U_inf)\n"
                    )
                    f.write(
                        f"  Mean RMS difference: {mean_rms:.3f} m/s ({mean_rms/U_inf*100:.1f}% of U_inf)\n"
                    )
                    f.write(
                        f"  Maximum absolute difference: {max_abs:.3f} m/s ({max_abs/U_inf*100:.1f}% of U_inf)\n"
                    )

        print(f"\nSummary saved to: {summary_path}")

    return df if all_results else None


if __name__ == "__main__":
    df = main()
    plt.show()
