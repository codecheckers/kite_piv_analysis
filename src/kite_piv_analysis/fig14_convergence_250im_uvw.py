import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path
import numpy as np
from utils import project_dir
from plot_styling import set_plot_style, plot_on_ax


def read_single_dat_file_into_df(file_path):
    """Read a single .dat file and return a pandas DataFrame."""
    # Read header information
    variables = None
    data_lines = []

    with open(file_path, "r") as f:
        for line in f:
            if "VARIABLES" in line:
                # Extract variable names
                variables = [
                    var.strip(' "') for var in line.split("=")[1].strip().split(",")
                ]
            elif line[0].isdigit() or line[0] == "-":
                # Data line
                data_lines.append([float(val) for val in line.strip().split()])

    # Create DataFrame
    df = pd.DataFrame(data_lines, columns=variables)
    return df


# def calculate_values(point_coords):
#     """Calculate values at a specific point for all files in the folder."""
#     # loop through all dat files in folder
#     values_at_point_list = []
#     folder_path = Path(
#         project_dir,
#         "data",
#         "raw_dat_files",
#         "convergence",
#         "flipped_aoa_23_vw_15_H_918_Z1_Y4_X2",
#     )

#     # Make sure folder exists
#     if not folder_path.exists():
#         raise FileNotFoundError(f"Folder not found: {folder_path}")

#     V_value_list = []
#     # Get all .dat files in the folder
#     for file in folder_path.glob("*.dat"):
#         # Read the file
#         df = read_single_dat_file_into_df(file)

#         # Find the closest point to the specified coordinates
#         # Assuming point_coords is a tuple of (x, y)
#         x_coord, y_coord = point_coords
#         df["distance"] = np.sqrt(
#             (df["x [mm]"] - x_coord) ** 2 + (df["y [mm]"] - y_coord) ** 2
#         )
#         closest_point_idx = df["distance"].idxmin()
#         values_at_point = df.iloc[closest_point_idx].drop("distance")

#         values_at_point_list.append(values_at_point)

#         ## find all V values
#         V_values = df["Velocity |V| [m/s]"].values

#     return values_at_point_list, V_values


# def plot_convergence(variable, values_at_point_list):
#     """Plot convergence of a specific variable."""
#     VARIABLES = [
#         "x [mm]",
#         "y [mm]",
#         "Velocity u [m/s]",
#         "Velocity v [m/s]",
#         "Velocity w [m/s]",
#         "Velocity |V| [m/s]",
#         "du/dx [1/s]",
#         "du/dy [1/s]",
#         "dv/dx [1/s]",
#         "dv/dy [1/s]",
#         "dw/dx [1/s]",
#         "dw/dy [1/s]",
#         "Vorticity w_z (dv/dx - du/dy) [1/s]",
#         "|Vorticity| [1/s]",
#         "Divergence 2D (du/dx + dv/dy) [1/s]",
#         "Swirling strength 2D (L_ci) [1/s^2]",
#         "isValid",
#     ]
#     if variable == "V":
#         variable_to_be_plotted = f"Velocity |{variable}| [m/s]"
#     else:
#         variable_to_be_plotted = f"Velocity {variable} [m/s]"

#     # randomly shuffle the values at point list
#     random.shuffle(values_at_point_list)

#     # extract the variable to be plotted
#     variable_values = [point[variable_to_be_plotted] for point in values_at_point_list]

#     # Create x-axis as number of samples
#     x_axis = range(1, len(variable_values) + 1)

#     # plot convergence
#     set_plot_style()
#     fig, ax = plt.subplots(figsize=(10, 5))
#     plot_on_ax(
#         ax,
#         x_axis,
#         variable_values,
#         label="Local value",
#         linestyle="-",
#         color="b",
#         is_with_grid=True,
#     )
#     # ax.plot(x_axis, variable_values, "b-", linewidth=1)
#     ax.set_xlabel("Number of Samples")
#     ax.set_ylabel(f"{variable} [m/s]")
#     # ax.set_title(f"Convergence of {variable_to_be_plotted}")

#     # Calculate and plot running mean
#     running_mean = np.cumsum(variable_values) / np.arange(1, len(variable_values) + 1)
#     ax.plot(x_axis, running_mean, "r-", linewidth=2, label="Running Mean")

#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(
#         Path(project_dir, "results", "paper_plots", f"convergence_250im_{variable}.pdf")
#     )


def calculate_values(point_coords):
    """
    Calculate values at a specific point for all files in the folder and average V values.

    Returns:
        tuple: (values_at_point_list, averaged_V_values)
            - values_at_point_list: List of Series containing values at specified point
            - averaged_V_values: Array of averaged |V| values for entire field
    """
    # loop through all dat files in folder
    values_at_point_list = []
    folder_path = Path(
        project_dir,
        "data",
        "raw_dat_files_convergence",
        "flipped_aoa_23_vw_15_H_918_Z1_Y4_X2",
    )

    # Make sure folder exists
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Initialize list to store V values from all files
    all_V_values = []
    all_is_valid = []
    all_w = []
    first_file = True

    # Get all .dat files in the folder
    for file in folder_path.glob("*.dat"):
        # Read the file
        df = read_single_dat_file_into_df(file)

        # Store the shape of the velocity field from first file
        if first_file:
            field_shape = df["Velocity |V| [m/s]"].values.shape
            first_file = False

        # Find the closest point to the specified coordinates
        x_coord, y_coord = point_coords
        df["distance"] = np.sqrt(
            (df["x [mm]"] - x_coord) ** 2 + (df["y [mm]"] - y_coord) ** 2
        )
        closest_point_idx = df["distance"].idxmin()
        values_at_point = df.iloc[closest_point_idx].drop("distance")

        values_at_point_list.append(values_at_point)

        # Store V values from this file
        all_V_values.append(df["Velocity |V| [m/s]"].values)

        # Store isValid and w arrays for later filtering (preserve ordering)
        if "isValid" in df.columns:
            all_is_valid.append(df["isValid"].values)
        else:
            # if not present, create an all-ones mask
            all_is_valid.append(np.ones_like(df["Velocity |V| [m/s]"].values))

        if "Velocity w [m/s]" in df.columns:
            all_w.append(df["Velocity w [m/s]"].values)
        else:
            all_w.append(np.zeros_like(df["Velocity |V| [m/s]"].values))

    # Convert list of V values to numpy array and calculate mean
    all_V_values = np.array(all_V_values)
    averaged_V_values = np.mean(all_V_values, axis=0)

    # Ensure the averaged values maintain the same shape as original data
    assert averaged_V_values.shape == field_shape, "Shape mismatch in averaged values"

    return values_at_point_list, averaged_V_values, all_V_values, all_is_valid, all_w


def plot_variable_on_ax(
    ax,
    variable_column,
    variable_label,
    values_at_point_list,
    is_xlabel=False,
    is_legend=False,
):
    # Right plot: convergence plot
    if variable_column == "V":
        variable_to_be_plotted = f"Velocity |{variable_column}| [m/s]"
    else:
        variable_to_be_plotted = f"Velocity {variable_column} [m/s]"

    # Randomly shuffle the values for convergence analysis
    shuffled_values = values_at_point_list.copy()
    random.shuffle(shuffled_values)

    # Extract the variable to be plotted
    variable_values = [point.loc[variable_to_be_plotted] for point in shuffled_values]
    x_axis = range(1, len(variable_values) + 1)

    # Create convergence plot
    plot_on_ax(
        ax,
        x_axis,
        variable_values,
        label="Local value",
        linestyle="-",
        color="b",
        is_with_grid=False,
        is_with_x_label=is_xlabel,
        # is_with_x_ticks=is_xlabel,
        is_with_x_tick_label=is_xlabel,
        x_label="Number of samples",
        y_label=f"{variable_label} " + r"(ms$^{-1}$)",
        is_with_legend=False,
    )

    # Calculate and plot running mean
    running_mean = np.cumsum(variable_values) / np.arange(1, len(variable_values) + 1)
    ax.plot(x_axis, running_mean, "r-", linewidth=2, label="Running Mean")
    if is_legend:
        ax.legend()


def plot_convergence(variable, values_at_point_list, point_coords, V_values):
    """
    Plot convergence of a specific variable with a side-by-side comparison:
    - Left: full velocity field with measurement point location
    - Right: convergence plot of specified variable
    """
    # Set up the figure with two subplots side by side
    set_plot_style()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    # # Left plot: velocity field and measurement point
    # # Load CSV file into a DataFrame
    # csv_file_path = Path(
    #     project_dir,
    #     "data",
    #     "raw_images",
    #     "aoa_13",
    #     "Y1",
    #     "normal_Y1_X2",
    #     "B0001_subsampled.csv",
    # )  # Replace with your actual path
    # df = pd.read_csv(csv_file_path)

    # # Filter valid points based on the 'isValid' column
    # valid_points = df[df["isValid"] == 1.0]

    # # Extract data for the scatter plot
    # x_scatter = valid_points["x [mm]"] / 1e3  # Convert to meters if needed
    # y_scatter = valid_points["y [mm]"] / 1e3  # Convert to meters if needed
    # intensity = valid_points["Intensity [counts]"]

    # # Scatter plot of intensity
    # sc = ax1.scatter(
    #     x_scatter,
    #     y_scatter,
    #     c=intensity,
    #     cmap="binary",
    #     alpha=0.7,
    #     s=5,  # Adjust point size
    #     label="Intensity [counts]",
    #     vmin=0,
    #     vmax=4000,
    # )

    # # Add color bar
    # fig.colorbar(sc, ax=ax1, label="Intensity [counts]")

    # # Add labels and legend
    # ax1.set_xlabel("x [m]")
    # ax1.set_ylabel("y [m]")
    # ax1.grid(False)
    ##TODO: You can add a plot, with the raw image if reviewers ask for it
    # Create 3 rows x 2 columns: left = raw, right = filtered (same ordering)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    # Build a single shuffled ordering that will be reused for all variables
    n_samples = len(values_at_point_list)
    indices = list(range(n_samples))
    random.shuffle(indices)

    variables = [
        ("u", r"$u_{x}$"),
        ("v", r"$u_{y}$"),
        ("w", r"$u_{z}$"),
    ]

    for row, (var_col, var_label) in enumerate(variables):
        # column name in DataFrame
        if var_col == "V":
            colname = f"Velocity |{var_col}| [m/s]"
        else:
            colname = f"Velocity {var_col} [m/s]"

        # Unfiltered values (preserve ordering by indices)
        unfiltered = [values_at_point_list[i].loc[colname] for i in indices]

        # Filtered values: apply DaVis isValid and |w| <= 3 m/s on each sample
        filtered = []
        for i, val in zip(indices, unfiltered):
            series = values_at_point_list[i]
            is_valid = series.get("isValid", 1)
            w_val = series.get("Velocity w [m/s]", 0.0)
            if (is_valid != 0) and (abs(w_val) <= 3.0):
                filtered.append(val)
            else:
                filtered.append(np.nan)

        # X-axis
        x_axis = range(1, n_samples + 1)

        # Left: raw
        ax_left = axes[row, 0]
        plot_on_ax(
            ax_left,
            x_axis,
            unfiltered,
            label="Local value",
            linestyle="-",
            color="b",
            is_with_grid=False,
            is_with_x_label=(row == 2),
            is_with_x_tick_label=(row == 2),
            x_label="Number of samples",
            y_label=f"{var_label} (ms$^{{-1}}$)",
        )
        running_mean_un = np.cumsum(np.nan_to_num(unfiltered, nan=0.0)) / np.arange(
            1, n_samples + 1
        )
        ax_left.plot(x_axis, running_mean_un, "r-", linewidth=2)
        if row == 0:
            ax_left.set_title("Raw data")

        # Right: filtered
        ax_right = axes[row, 1]
        plot_on_ax(
            ax_right,
            x_axis,
            filtered,
            label="Local value (filtered)",
            linestyle="-",
            color="b",
            is_with_grid=False,
            is_with_x_label=(row == 2),
            is_with_x_tick_label=(row == 2),
            x_label="Number of samples",
            y_label=f"{var_label} (ms$^{{-1}}$)",
        )
        # running mean should ignore NaNs: compute cumulative mean of valid entries
        running_mean_f = []
        cumsum = 0.0
        count = 0
        for v in filtered:
            if np.isnan(v):
                running_mean_f.append(np.nan if count == 0 else cumsum / count)
            else:
                cumsum += v
                count += 1
                running_mean_f.append(cumsum / count)
        ax_right.plot(x_axis, running_mean_f, "r-", linewidth=2)
        if row == 0:
            ax_right.set_title("Filtered: isValid and |w|<=3 m/s")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        Path(
            project_dir,
            "results",
            "paper_plots_21_10_2025",
            f"fig14_convergence_250im_uvw.pdf",
        )
    )
    plt.close()


# Example usage
if __name__ == "__main__":
    # Example coordinates for the point of interest
    point_coords = (-0.75368, -118.177)  # in free-stream, looks good

    # Calculate values for Y4
    values, V_values, all_V_values, all_is_valid, all_w = calculate_values(point_coords)

    # Plot convergence for Y4 (left = raw, right = filtered)
    plot_convergence("V", values, point_coords, V_values)
