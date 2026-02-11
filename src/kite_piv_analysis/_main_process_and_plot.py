from __future__ import annotations

import argparse
import importlib
import shutil
from contextlib import contextmanager
from pathlib import Path

from utils import project_dir


PROJECT_DIR = Path(project_dir)

# fig05 script is currently not present in scripts/
PLOT_MODULES = [
    "fig04_plane_location",
    "fig06_qualitative_comparison_CFD_PIV",
    "fig07_bounds_CFD_PIV_single_row",
    "fig08_gamma_distribution",
    "fig09_spanwise_CFD_comparison_v_and_p_x_10_to_50",
    "fig10_PIV_normal_masked_Y1",
    "fig11_spanwise_CFD_alpha_comparison",
    "fig12_bounds_CFD_PIV",
    "fig13_convergence_study",
    "fig14_convergence_250im_uvw",
    "fig15_line_interference_PIV",
]


def processed_cfd_path(alpha: int, y_num: int) -> Path:
    return (
        PROJECT_DIR
        / "processed_data"
        / "CFD"
        / f"alpha_{alpha}"
        / f"Y{y_num}_paraview_corrected.csv"
    )


def piv_stitched_path(aoa: int, y_num: int) -> Path:
    return (
        PROJECT_DIR
        / "data"
        / "piv_stichted_planes"
        / f"aoa_{aoa}"
        / f"aoa_{aoa}_Y{y_num}_stitched.csv"
    )


def spanwise_raw_path(alpha: int, cm_offset: int) -> Path:
    return (
        PROJECT_DIR
        / "data"
        / "CFD_slices"
        / "spanwise_slices"
        / f"alpha_{alpha}_CFD_spanwise_slice_{cm_offset}cm_1.csv"
    )


def spanwise_outline_data_path(alpha: int, cm_offset: int) -> Path:
    return (
        PROJECT_DIR
        / "data"
        / "CFD_slices"
        / "spanwise_slices_outline_wing"
        / f"alpha_{alpha}_CFD_{cm_offset}cm_outline_wing.csv"
    )


def convergence_paths(alpha: int, y_num: int) -> list[Path]:
    files: list[Path] = []
    for data_type in ["CFD", "PIV"]:
        for shape in ["Ellipse", "Rectangle"]:
            for parameter in ["iP", "dLx", "dLy"]:
                files.append(
                    PROJECT_DIR
                    / "processed_data"
                    / "convergence_study"
                    / f"alpha_{alpha}_Y{y_num}_{data_type}_{shape}_{parameter}.csv"
                )
    files.append(
        PROJECT_DIR
        / "processed_data"
        / "convergence_study"
        / "PIV_sweep"
        / f"alpha_{alpha}_Y{y_num}_PIV_Ellipse.csv"
    )
    files.append(
        PROJECT_DIR
        / "processed_data"
        / "convergence_study"
        / "PIV_sweep"
        / f"alpha_{alpha}_Y{y_num}_PIV_Rectangle.csv"
    )
    return files


def required_processed_cfd_files() -> list[Path]:
    files: list[Path] = []
    for y_num in range(1, 8):
        files.append(processed_cfd_path(6, y_num))
    for y_num in range(1, 5):
        files.append(processed_cfd_path(16, y_num))
    return files


def required_source_files() -> list[Path]:
    files: list[Path] = []

    # PIV stitched input used by fig06/07/10/12/13/15
    for y_num in range(1, 8):
        files.append(piv_stitched_path(13, y_num))
    for y_num in range(1, 5):
        files.append(piv_stitched_path(23, y_num))

    # Airfoil and bound placement data used by plotting helpers
    files.append(PROJECT_DIR / "data" / "airfoils" / "airfoil_translation_values.csv")
    for y_num in range(1, 8):
        files.append(PROJECT_DIR / "data" / "airfoils" / f"y{y_num}.dat")
    files.append(PROJECT_DIR / "data" / "optimal_bound_placement.csv")

    # fig08 and fig14 source inputs
    files.append(PROJECT_DIR / "data" / "gamma_distribution" / "y_locations.csv")

    # Spanwise raw source data used to generate missing outlines
    for cm_offset in [10, 15, 20, 25, 30]:
        files.append(spanwise_raw_path(6, cm_offset))
    files.append(spanwise_raw_path(16, 25))

    return files


def required_spanwise_outline_files() -> list[Path]:
    files: list[Path] = []
    for cm_offset in [10, 15, 20, 25, 30]:
        files.append(spanwise_outline_data_path(6, cm_offset))
    files.append(spanwise_outline_data_path(16, 25))
    return files


def missing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def ensure_result_directories() -> None:
    (PROJECT_DIR / "results" / "paper_plots_21_10_2025").mkdir(
        parents=True, exist_ok=True
    )
    (PROJECT_DIR / "results" / "convergence_study" / "alpha_6").mkdir(
        parents=True, exist_ok=True
    )
    (PROJECT_DIR / "results" / "convergence_study" / "alpha_16").mkdir(
        parents=True, exist_ok=True
    )
    (PROJECT_DIR / "processed_data" / "gamma_distribution").mkdir(
        parents=True, exist_ok=True
    )
    (PROJECT_DIR / "processed_data" / "CFD" / "spanwise_slices").mkdir(
        parents=True, exist_ok=True
    )
    (PROJECT_DIR / "data" / "CFD_slices" / "spanwise_slices_outline_wing").mkdir(
        parents=True, exist_ok=True
    )
    for alpha in [6, 16]:
        (PROJECT_DIR / "processed_data" / "CFD" / f"alpha_{alpha}").mkdir(
            parents=True, exist_ok=True
        )


def validate_source_data() -> None:
    missing = missing_paths(required_source_files())

    raw_dat_dir = (
        PROJECT_DIR
        / "data"
        / "raw_dat_files_convergence"
        / "flipped_aoa_23_vw_15_H_918_Z1_Y4_X2"
    )
    if not raw_dat_dir.exists() or not any(raw_dat_dir.glob("*.dat")):
        missing.append(raw_dat_dir)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required source data. Add these first:\n" + formatted
        )


def ensure_processed_cfd() -> None:
    required = required_processed_cfd_files()
    missing_before = missing_paths(required)
    if not missing_before:
        print("Processed CFD files already available.")
        return

    print(
        f"Generating missing processed CFD files ({len(missing_before)} missing before run)..."
    )
    import transforming_paraview_output

    transforming_paraview_output.main()

    missing_after = missing_paths(required)
    if missing_after:
        formatted = "\n".join(f"  - {path}" for path in missing_after)
        raise RuntimeError(
            "Could not generate all processed CFD files. Still missing:\n" + formatted
        )
    print("Processed CFD files generated.")


@contextmanager
def _suppress_matplotlib_show():
    import matplotlib.pyplot as plt

    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = original_show


def ensure_spanwise_outlines() -> None:
    missing = missing_paths(required_spanwise_outline_files())
    if not missing:
        print("Spanwise outline files already available.")
        return

    print(f"Generating missing spanwise outlines ({len(missing)} missing before run)...")
    import extract_spanwise_contour

    for data_outline_path in missing:
        stem = data_outline_path.stem  # alpha_6_CFD_25cm_outline_wing
        parts = stem.split("_")
        alpha = int(parts[1])
        cm_offset = int(parts[3].replace("cm", ""))

        raw_path = spanwise_raw_path(alpha, cm_offset)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Cannot generate outline; missing raw spanwise slice: {raw_path}"
            )

        with _suppress_matplotlib_show():
            points = extract_spanwise_contour.transform_raw_csv_to_processed_df(
                alpha=alpha,
                cm_offset=cm_offset,
            )
            generated_path = extract_spanwise_contour.save_contour_points_to_csv(
                points, alpha, cm_offset
            )

        generated_path = Path(generated_path)
        if not generated_path.exists():
            raise RuntimeError(f"Outline generation returned missing file: {generated_path}")

        shutil.copy2(generated_path, data_outline_path)
        print(f"Generated: {data_outline_path}")

    missing_after = missing_paths(required_spanwise_outline_files())
    if missing_after:
        formatted = "\n".join(f"  - {path}" for path in missing_after)
        raise RuntimeError(
            "Could not generate all spanwise outline files. Still missing:\n" + formatted
        )
    print("Spanwise outline files generated.")


def ensure_quantitative_gamma_file() -> None:
    target = (
        PROJECT_DIR / "processed_data" / "quantitative_chordwise_analysis_alpha_6_with_std.csv"
    )
    if target.exists():
        print("Quantitative chordwise analysis file already available.")
        return

    print("Generating quantitative chordwise analysis (alpha=6, Y1-Y7)...")
    import calculating_noca_and_kutta

    calculating_noca_and_kutta.save_results_single_alpha(
        alpha=6,
        y_num_list=[1, 2, 3, 4, 5, 6, 7],
    )
    if not target.exists():
        raise RuntimeError(f"Failed to generate required file: {target}")
    print(f"Generated: {target}")


def ensure_vsm_gamma_cache() -> None:
    target = PROJECT_DIR / "processed_data" / "gamma_distribution" / "VSM_gamma_distribution.csv"
    if target.exists():
        print("VSM gamma cache already available.")
        return

    print("Generating VSM gamma cache...")
    import fig08_gamma_distribution

    fig08_gamma_distribution.get_VSM_gamma_distribution()
    if not target.exists():
        raise RuntimeError(f"Failed to generate required file: {target}")
    print(f"Generated: {target}")


def ensure_convergence_cache() -> None:
    required_pairs = [(6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (16, 1)]
    missing_pairs: list[tuple[int, int]] = []
    for alpha, y_num in required_pairs:
        pair_missing = missing_paths(convergence_paths(alpha, y_num))
        if pair_missing:
            missing_pairs.append((alpha, y_num))

    if not missing_pairs:
        print("Convergence cache files already available.")
        return

    print(f"Generating convergence cache for {len(missing_pairs)} alpha/Y pairs...")
    import fig13_convergence_study

    parameter_names = ["iP", "dLx", "dLy"]
    fast_factor = 25

    for alpha, y_num in missing_pairs:
        print(f"  - alpha={alpha}, Y={y_num}")
        fig13_convergence_study.storing_and_collecting_results(
            alpha=alpha,
            y_num=y_num,
            parameter_names=parameter_names,
            fast_factor=fast_factor,
        )
        fig13_convergence_study.storing_PIV_percentage_sweep(
            alpha=alpha,
            y_num=y_num,
            n_points=10,
        )

    missing_after: list[Path] = []
    for alpha, y_num in required_pairs:
        missing_after.extend(missing_paths(convergence_paths(alpha, y_num)))
    if missing_after:
        formatted = "\n".join(f"  - {path}" for path in missing_after)
        raise RuntimeError(
            "Could not generate all convergence cache files. Still missing:\n" + formatted
        )
    print("Convergence cache files generated.")


def run_plot_scripts() -> None:
    print("Running plot scripts for fig04 and fig06-fig15 (fig05 not present)...")
    failures: list[tuple[str, Exception]] = []
    for module_name in PLOT_MODULES:
        print(f"  - {module_name}.main()")
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, "main"):
                raise AttributeError(f"Module has no main(): {module_name}")
            module.main()
        except Exception as error:
            failures.append((module_name, error))

    if failures:
        details = "\n".join(
            f"  - {name}: {error.__class__.__name__}: {error}"
            for name, error in failures
        )
        raise RuntimeError("One or more plot scripts failed:\n" + details)
    print("All requested figure scripts completed.")


def main(run_plots: bool = True) -> None:
    print("Starting dependency-aware processing pipeline...")
    ensure_result_directories()
    validate_source_data()

    ensure_processed_cfd()
    ensure_spanwise_outlines()
    ensure_quantitative_gamma_file()
    ensure_convergence_cache()
    ensure_vsm_gamma_cache()

    if run_plots:
        run_plot_scripts()
    else:
        print("Skipping plot execution (--skip-plots).")

    print("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare all files needed by fig04 and fig06-fig15, and optionally run plotting."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only generate/check dependencies; do not run figure scripts.",
    )
    args = parser.parse_args()
    main(run_plots=not args.skip_plots)
