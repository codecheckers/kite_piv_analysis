<!-- [![CODECHECK](https://codecheck.org.uk/img/codeworks-badge.svg)](https://doi.org/10.5281/zenodo.15603144) -->

# Computational aerodynamics for soft-wing kite design
This repository contains code that generates the figures a paper, titled "Computational aerodynamics for 
soft-wing kite design" published Open-Source in Wind Energy science, [INSERT LINK HERE].

Download from [Zenodo](https://doi.org/10.5281/zenodo.16925758) and place in `data/ml_models/`

A machine learning model was trained on more than a hundred thousands Reynolds-average Navier Stokes (RANS) Computational Fluid Dynamics (CFD) simulations made for leading-edge inflatable airfoils, documented in the paper and in the MSc. thesis of [K.R.G. Masure](https://resolver.tudelft.nl/uuid:865d59fc-ccff-462e-9bac-e81725f1c0c9), the [code base is also open-source accessible](https://github.com/awegroup/Pointwise-Openfoam-toolchain).

As the three trained models, for Reynolds number = 1e6, 5e6 and 1e7 are too large (~2.3GB) for GitHub, they have to be downloaded separately, and added to the `data/ml_models` folder. They are accessible through [Zenodo](https://doi.org/10.5281/zenodo.16925758), and so is the [CFD data](https://doi.org/10.5281/zenodo.16925833) on which the models are trained. 

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
2. Download ML models from: [Zenodo](https://doi.org/10.5281/zenodo.16925758) and place inside the `data/ml_models` folder.

3. Download the rest of the supporing data from: [INSERT LINK HERE] and place inside the `data/` folder.
   
4. Run 
    ```bash
    python -m wes_aero_sim_for_kite_design.main
    ```

### Dependencies
- "numpy", 
-  "pandas>=1.5.3", 
-  "matplotlib>=3.7.1",
-  "odfpy",
-  "VSM @ git+https://github.com/awegroup/Vortex-Step-Method.git@v2.0.3",

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
> Copyright (c) [YEAR] Jelle Poland
