<!-- [![CODECHECK](https://codecheck.org.uk/img/codeworks-badge.svg)](https://doi.org/10.5281/zenodo.15603144) -->

# Flow Field Analysis of a Leading-Edge Inflatable Kite Rigid Scale Model Using Stereoscopic Particle Image Velocimetry
This repository contains code that generates the figures a paper, titled: ["Flow Field Analysis of a Leading-Edge Inflatable Kite Rigid Scale Model Using Stereoscopic Particle Image Velocimetry"](https://wes.copernicus.org/preprints/wes-2025-217/wes-2025-217.pdf) published Open-Source in Wind Energy science.

## Usage instructions
1. Install the repository:
   Linux: 
    ```bash
    git clone git@github.com:jellepoland/WES_aero_sim_for_kite_design.git && \
    cd WES_aero_sim_for_kite_design && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install -e .[dev]
    ```
    Windows:
    ```bash
    git clone git@github.com:jellepoland/WES_aero_sim_for_kite_design.git; `
    cd WES_aero_sim_for_kite_design; `
    python -m venv venv; `
    .\venv\Scripts\Activate.ps1; `
    pip install -e .[dev]
    ```
2. Download the necessary data from [Zenodo](https://doi.org/10.5281/zenodo.17395913) and place inside the `data/` folder.

4. Run 
    ```bash
    python ./src/kite_piv_analysis/_main_process_and_plot.py
    ```

### Dependencies
- `numpy`
- `pandas>=1.5.3`
- `matplotlib>=3.7.1`
- `cycler`
- `xarray>=2024.6.0`
- `netCDF4`
- `scipy`
- `openpyxl`
- `VSM @ git+https://github.com/awegroup/Vortex-Step-Method.git@v2.1.0`

## Citation
If you use this project in your research, please consider citing it. 
Citation details can be found in the [CITATION.cff](CITATION.cff) file included in this repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## :warning: License and Waiver

Specify the license under which your software is distributed and include the copyright notice:

> Technische Universiteit Delft hereby disclaims all copyright interest in the program “NAME PROGRAM” (one line description of the content or function) written by the Author(s).
> 
> Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering
> 
> Copyright (c) [2026] Jelle Poland
