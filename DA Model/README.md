# DA Model - Double Auction Implementation

## Overview

This directory contains the implementation of the Double Auction (DA) trading mechanism for the transactive energy market simulation. The DA follows the simultaneous time-delimited auction format used in the Transactive Energy Service System (TESS), where agents submit bids and offers that are cleared through a centralized matching mechanism.

## Key Components

### Core Files

- **`vfi_simulations.py`**: Main simulation runner implementing Value Function Iteration (VFI) for battery optimization
- **`DA_ABM_testing_notebook.ipynb`**: Jupyter notebook for testing and development of DA mechanisms

## Double Auction Mechanism

### Market Structure

The Double Auction implementation features:

- **Simultaneous Bidding**: All agents submit bids/offers simultaneously
- **Time-Delimited Clearing**: Markets clear at discrete time intervals
- **Price Discovery**: Equilibrium price determined by supply-demand intersection
- **Centralized Matching**: Order book maintained by central auctioneer

### Trading Process

1. **Bid Collection**: Agents submit quantity-price pairs
2. **Order Sorting**: Bids sorted by price (highest first), offers by price (lowest first)  
3. **Market Clearing**: Find intersection of supply and demand curves
4. **Trade Execution**: All trades execute at uniform clearing price
5. **Settlement**: Quantities allocated to successful bidders/sellers

## Battery Optimization

### Value Function Iteration (VFI)

The DA model implements sophisticated battery optimization using dynamic programming:

```python
# VFI solves: V(C_t, t) = max_{q_b} [profit_t + β * V(C_{t+1}, t+1)]
# Subject to: C_{t+1} = C_t + q_b (charging/discharging constraints)
```

#### Key Features

- **Backward Induction**: Solves from final period backwards
- **State Space**: Battery charge level and time period
- **Action Space**: Charge/discharge quantities within technical limits
- **Bellman Equation**: Optimizes expected future value
- **Perfect Foresight**: Assumes knowledge of future price distributions

## Running Simulations

### Main Simulation Script

```bash
python vfi_simulations.py
```

This script:

- Implements VFI battery optimization
- Runs parameter sweeps across multiple configurations
- Generates comparative data for welfare analysis
- Produces outputs comparable to AMM model results

### Development and Testing

Use the Jupyter notebook for:

- Interactive development and debugging
- Visualization of auction dynamics  
- Testing specific scenarios
- Algorithm validation

```bash
jupyter notebook DA_ABM_testing_notebook.ipynb
```

## Parameter Configuration

The DA simulations test:

- **Battery Parameters**: Same grid as AMM model for comparability
- **Market Structure**: Various bid/offer strategies
- **Agent Behavior**: Strategic vs. truthful bidding
- **Time Horizons**: Multi-period optimization scenarios

