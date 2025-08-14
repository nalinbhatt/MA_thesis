# ABM Model - Automated Market Maker Implementation

## Overview

This directory contains the implementation of the Automated Market Maker (AMM) trading mechanism for the transactive energy market simulation. The AMM follows a Uniswap-style constant product formula (x * y = k) where energy tokens and money tokens are traded through a decentralized liquidity pool.

## Key Components

### Core Model Files

- **`model.py`**: Main simulation engine that orchestrates agent interactions and market dynamics
- **`amm.py`**: Automated Market Maker implementation with constant product formula
- **`data_tracker.py`**: Comprehensive data collection and analysis utilities
- **`simulate.py`**: Single simulation runner for testing specific configurations
- **`multi_simulation_runner.py`**: Batch simulation runner for parameter sweeps

### Agent Implementations

- **`demand.py`**: Elastic demand agent with utility maximization
- **`supply.py`**: Supply agents (utility and solar) with cost minimization
- **`battery.py`**: Two battery trading strategies:
  - **Informed Trader**: Heuristic strategy exploiting price differentials
  - **Optimal Policy (VFI)**: Value Function Iteration with dynamic programming

## Running Simulations

### Single Configuration Simulation

For testing specific parameter configurations:

```bash
python simulate.py
```

This script allows you to:
- Test specific agent configurations
- Debug model behavior
- Validate parameter settings
- Generate quick results for analysis

### Full Parameter Sweep

For running the complete thesis parameter grid:

```bash
python multi_simulation_runner.py
```

This script:
- Runs parallel simulations across all parameter combinations
- Generates comprehensive CSV outputs
- Uses multiprocessing for efficient execution
- Produces data for statistical analysis

**Note**: This may take several hours to complete depending on your system.

## Parameter Configuration

The simulations test combinations of:

- **Battery Capacity**: 5, 10, 15, 20 MWh
- **Initial Charge**: 20%, 50%, 80% of maximum capacity
- **Charge/Discharge Rate**: 1-5 MW
- **Solar Production**: Mean peak power of 5-20 MW
- **Production Volatility**: 0 and 50% of mean peak power
- **Battery Strategies**: Informed Trader vs. VFI Optimal

## Output Files

### Generated Data

- **`abm_sim_summary.csv`**: Summary statistics for each simulation run
- **`abm_period_summary.csv`**: Detailed period-by-period results


### Data Structure

Each simulation generates:
- Total social welfare metrics
- Agent-specific surplus calculations
- Price dynamics and trading volumes
- Battery state-of-charge trajectories
- AMM liquidity pool evolution

## Key Features

### AMM Implementation

- Constant product market maker (x * y = k)
- Dynamic pricing based on pool reserves
- Slippage calculation for large trades
- Liquidity provision mechanics

### Agent Strategies

#### Informed Trader Battery
- Anticipates future price movements
- Exploits intertemporal arbitrage opportunities
- Simple but effective heuristic approach

#### VFI Optimal Battery
- Dynamic programming solution
- Backward induction optimization
- Considers full state space and time horizon

### Solar Generation

- Cosine-based daily irradiance profiles
- Stochastic peak power with normal distribution
- Multi-day simulation support
- Ensures positive generation values

## Dependencies

Key Python packages required:
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `tqdm`: Progress tracking for simulations
- `multiprocessing`: Parallel execution

## Usage Examples

### Running a Quick Test

```python
# Example: Single day simulation with informed trader battery
from simulate import *

# Set parameters
C_max = 10  # Battery capacity
C_init = 5  # Initial charge
q_b_max = 2  # Max charge/discharge rate

# Run simulation
# (See simulate.py for full parameter configuration)
```

### Customizing Parameters

Modify the parameter grids in `multi_simulation_runner.py`:

```python
# Example: Test only high capacity batteries
C_max_grid = [15, 20]  # Focus on larger batteries
mean_pmax_grid = [10, 15]  # Medium solar production
```

## Technical Notes

- Simulations use 24-hour periods with hourly time steps
- Multi-day simulations ensure circular consistency
- Parallel processing utilizes all available CPU cores
- Memory-efficient data handling for large parameter sweeps

## Troubleshooting

### Common Issues

1. **Memory Usage**: Large parameter grids may require significant RAM
2. **Execution Time**: Full parameter sweep takes several hours
3. **Convergence**: VFI battery may require tuning for complex scenarios

### Performance Tips

- Start with smaller parameter grids for testing
- Monitor CPU and memory usage during long runs
- Use `simulate.py` for debugging before full sweeps

## Next Steps

After running simulations:
1. Process results using notebooks in `Data Analysis/`
2. Compare with Double Auction results from `DA Model/`
3. Generate visualization and statistical analysis
