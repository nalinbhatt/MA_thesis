# The Cost of Decentralization? Welfare Comparisons between Double Auctions and Automated Market Makers in Transactive Energy Markets

## Overview

This repository contains the complete computational framework for my Master's thesis comparing the welfare implications of adopting Automated Market Makers (AMMs) versus Double Auctions in Transactive Energy markets. The research develops agent-based simulations of peer-to-peer electricity markets to quantify the "cost of decentralization" across different market mechanisms.

## Abstract

This research quantifies the welfare implications of adopting an Automated Market Maker (AMM) in place of the simultaneous time-delimited double auction used in the Transactive Energy Service System (TESS). We develop an agent-based simulation of a peer-to-peer electricity market populated by heterogeneous demand, solar, utility, and battery agents, and evaluate market outcomes under varying technical and operational parameters. Battery strategies are modeled both through value function iteration (VFI) dynamic programming and an alternative informed trader heuristic that exploits anticipated intertemporal price differentials.

## Project Structure

```
├── ABM Model/                      # Automated Market Maker (AMM) implementation
├── DA Model/                       # Double Auction (DA) implementation  
├── Data Analysis/                  # Results analysis and visualization
├── Professional Thesis Text/       # LaTeX thesis document
├── Rough Notebooks/               # Development and testing notebooks
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Key Features

- **Comparative Market Mechanisms**: Implementation of both AMM (Uniswap-style) and Double Auction systems
- **Multi-Agent Framework**: Heterogeneous agents including demand, solar, utility, and battery storage
- **Advanced Battery Strategies**: 
  - Value Function Iteration (VFI) with dynamic programming
  - Informed Trader heuristic exploiting price differentials
- **Comprehensive Analysis**: Parameter sweeps across battery capacity, solar production, and market volatility
- **Welfare Analysis**: Statistical comparison of surplus distribution and total social welfare

## Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/nalinbhatt/MA_thesis.git
cd MA_thesis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Simulations

- **Single Configuration**: Use individual model files for specific scenarios
- **Full Parameter Sweep**: Run complete thesis simulations using the multi-simulation runners
- **Data Analysis**: Process results using Jupyter notebooks in the Data Analysis folder

## Research Questions

1. **Welfare Comparison**: How do AMMs perform relative to Double Auctions in terms of total social welfare?
2. **Surplus Distribution**: How does the choice of market mechanism affect surplus distribution across different agent types?
3. **Battery Strategy Impact**: How do different battery trading strategies affect market outcomes and equity?
4. **Parametric Sensitivity**: Under what conditions do AMMs outperform Double Auctions?

## Results

Key findings from the research:
- AMMs deliver statistically significant gains in total social welfare relative to double auctions across most scenarios
- VFI battery strategy often generates negative surplus under AMMs, but these losses are offset by gains to other agents
- Informed trader battery strategy eliminates negative battery surplus while maintaining efficiency
- The "cost of decentralization" can be negative, implying net benefits from AMM adoption

## Directory Documentation

Each subdirectory contains detailed README files with specific instructions:

- **[ABM Model/](./ABM%20Model/README.md)**: AMM simulation framework
- **[DA Model/](./DA%20Model/README.md)**: Double Auction simulation framework  
- **[Data Analysis/](./Data%20Analysis/README.md)**: Results processing and visualization
- **[Professional Thesis Text/](./Professional%20Thesis%20Text/README.md)**: LaTeX thesis document

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{bhatt2025cost,
  title={The Cost of Decentralization? Welfare Comparisons between Double Auctions and Automated Market Makers in Transactive Energy Markets},
  author={Bhatt, Nalin},
  year={2025},
  school={University of Chicago},
  type={Master's Thesis},
  department={Computational Social Science}
}
```

## Keywords

Transactive Energy, Automated Market Makers, Double Auction, Distributed Energy Resources, Agent-based Modeling, Dynamic Programming

## Contact

**Author**: Nalin Bhatt  
**Institution**: University of Chicago, Master of Arts in Computational Social Science  
**Date**: August 2025 
