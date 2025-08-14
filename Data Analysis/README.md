# Data Analysis - Results Processing and Visualization

## Overview

This directory contains the comprehensive analysis pipeline for processing simulation results from both the AMM and Double Auction models. The analysis generates statistical comparisons, welfare calculations, and visualizations that support the thesis conclusions about the "cost of decentralization."

## Key Components

### Analysis Files

- **`data_analysis.ipynb`**: Main Jupyter notebook containing complete analysis pipeline
- **`da_amm_combined.csv`**: Cleaned and merged dataset combining DA and AMM results
- **`images/`**: Generated plots and visualizations for thesis
- **`Latex Regression tables/`**: LaTeX-formatted regression output tables

### Generated Outputs

#### Visualization Directory (`images/`)
- **`amm_lag.png`**: AMM pricing dynamics with lag effects
- **`amm_low.png`**: AMM performance under low solar scenarios  
- **`d_surp.png`**: Demand agent surplus distributions
- **`da_excess_q.png`**: Double auction excess quantity analysis
- **`da_low_price.png`**: DA pricing under low production
- **`da_low.png`**: DA performance metrics
- **`inf_trader_high.png`**: Informed trader performance
- **`solar_surp.png`**: Solar agent surplus analysis
- **`surplus_battery.png`**: Battery surplus comparisons
- **`Tot_surp.png`**: Total social welfare comparisons
- **`util_surplus.png`**: Utility agent surplus distribution

#### LaTeX Tables (`Latex Regression tables/`)
- **`regression_results.tex`**: Main welfare comparison regressions
- **`amm_regression_results.tex`**: AMM-specific analysis
- **`da_informed_regression_results.tex`**: DA with informed trader analysis
