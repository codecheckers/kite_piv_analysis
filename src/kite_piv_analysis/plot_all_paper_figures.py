from __future__ import annotations

import importlib
import sys
from pathlib import Path


if __package__ in (None, ""):
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    BASE_PACKAGE = "kite_piv_analysis"
else:
    BASE_PACKAGE = __package__


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    for fig_num in range(4, 16):
        if fig_num == 5:
            continue

        if fig_num == 8:
            continue

        for script_path in sorted(script_dir.glob(f"fig{fig_num:02d}_*.py")):
            module_name = script_path.stem
            module = importlib.import_module(f"{BASE_PACKAGE}.{module_name}")

            if not hasattr(module, "main"):
                print(f"Skipping {module_name}: no main()")
                continue

            print(f"Running {module_name}.main()")
            module.main()


if __name__ == "__main__":
    main()
