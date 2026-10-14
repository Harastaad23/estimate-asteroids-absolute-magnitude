# Asteroid Lightcurve & Phase Curve Analysis

This repository contains a Python program designed to analyze asteroid lightcurve data. It processes observational data to find rotation periods, fits Fourier series to the lightcurves, and calculates the absolute magnitude (H) and slope parameters (G1, G2) using the H-G1-G2 model.

## Features

* **Data Reductions:** Option to remove distance effects from apparent magnitude.
* **Phase Angle Interpolation:** Aligns ephemeris data with lightcurve observations.
* **Clustering:** Groups data based on phase angles using the Ckmeans algorithm.
* **Period Analysis:** Finds the best rotation period using the Lomb-Scargle Periodogram.
* **Fourier Fitting:** Automatically determines the best Fourier order and extracts the baseline magnitude ($c_0$).
* **Phase Curve Modeling:** Calculates parameters using both linear regression and the H-G1-G2 model.

## File Structure

* `main.py`: The main execution script that runs the step-by-step analysis.
* `asteroid_tools.py`: A modular file containing all the underlying math, processing, and plotting functions.
* Data folders (e.g., `19 Fortuna/`, `87 Sylvia/`): Expected to contain the `.csv` files for lightcurves and ephemeris data.

## Prerequisites

To run this program, you need Python installed on your computer along with several scientific libraries. You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn astropy scipy ckmeans-1d-dp
