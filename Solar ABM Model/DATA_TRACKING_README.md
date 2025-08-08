# Energy AMM ABM Data Tracking System Documentation

## Overview

The enhanced data tracking system provides comprehensive monitoring and analysis capabilities for your Energy Automated Market Maker (AMM) Agent-Based Model. It tracks trading activity at multiple levels of granularity and provides tools for analysis and export.

## Key Features

### 1. Multi-Level Data Tracking
- **Trading Round Level**: Individual transactions within each period
- **Period Level**: Aggregated data for each time period  
- **Agent Level**: Individual agent performance metrics
- **Simulation Level**: Overall simulation statistics

### 2. Comprehensive Metrics
- Electricity quantities bought/sold
- Money paid/received for transactions
- Price per unit for each trade
- Agent surplus calculations
- AMM state before/after trades
- Price impact measurements

### 3. Data Export and Analysis
- Automatic CSV export functionality
- Real-time analysis capabilities
- Summary statistics generation
- Agent performance comparisons

## Data Structures

### Trading Rounds Data
Each individual trade is recorded with:
```python
{
    'timestamp': datetime,
    'period': int,
    'round': int,
    'agent_type': str,
    'trade_type': 'buy' or 'sell',
    'quantity_electricity': float,
    'money_paid': float,
    'price_per_unit': float,
    'agent_surplus': float,
    'amm_reserve_x_before': float,
    'amm_reserve_y_before': float,
    'amm_price_before': float,
    'amm_reserve_x_after': float,
    'amm_reserve_y_after': float,
    'amm_price_after': float,
    'price_impact': float,
    'trade_successful': bool
}
```

### Period Summary Data
Aggregated data for each period:
```python
{
    'period': int,
    'solar_generation': float,
    'total_trades': int,
    'total_electricity_traded': float,
    'total_money_exchanged': float,
    'avg_price': float,
    'price_volatility': float,
    'final_amm_reserve_x': float,
    'final_amm_reserve_y': float,
    'final_amm_price': float,
    # Agent-specific aggregations
    'solar_total_electricity_sold': float,
    'battery_total_surplus': float,
    # etc.
}
```

## Usage Instructions

### 1. Basic Setup
The data tracker is automatically initialized when you create a Model instance:

```python
model = Model(T, agent_list, amm_agent, trades_per_period, summary_df, hourly_df, s_t)
```

### 2. Running Simulation
Simply run your simulation as before:
```python
model.simulate()
```

The data tracker automatically records all trades and aggregates period-level data.

### 3. Accessing Data
After simulation completion, access comprehensive data:

```python
# Get all trading data
trading_data = model.get_trading_data()

# Access specific DataFrames
trading_rounds_df = trading_data['trading_rounds']
period_summary_df = trading_data['period_summary'] 
agent_summary_df = trading_data['agent_summary']
summary_stats = trading_data['summary_statistics']

# Get agent-specific performance
battery_performance = model.get_agent_performance('informed_trader_battery')
```

### 4. Data Export
Export all data to CSV files:
```python
model.export_detailed_results("my_simulation")
```

This creates three CSV files:
- `my_simulation_trading_rounds_[timestamp].csv`
- `my_simulation_period_summary_[timestamp].csv`
- `my_simulation_agent_trades_[timestamp].csv`

## Analysis Capabilities

### 1. Trading Volume Analysis
- Total electricity traded per period
- Trading frequency by agent type
- Market participation rates

### 2. Price Analysis
- Price evolution over time
- Price volatility measurements
- Price impact of trades
- Average prices by agent type

### 3. Agent Performance
- Individual agent profitability
- Surplus calculations by agent type
- Trading behavior patterns
- Success rates for different trade types

### 4. Market Efficiency
- Social welfare calculations
- Market liquidity metrics
- Price discovery efficiency

## Key Advantages

### 1. Granular Tracking
- Every individual trade is recorded
- No data loss between trading rounds
- Complete audit trail of all transactions

### 2. Flexible Aggregation
- Data can be aggregated by period, agent type, or trade type
- Multiple levels of analysis possible
- Easy to calculate custom metrics

### 3. Safe Data Storage
- All data stored in structured DataFrames
- Automatic error handling for failed trades
- Consistent data types and formats

### 4. Analysis Ready
- Pre-calculated common metrics
- Easy integration with pandas/matplotlib
- Ready for statistical analysis

## Example Analysis Workflows

### 1. Price Evolution Analysis
```python
trading_rounds_df = model.get_trading_data()['trading_rounds']
price_by_period = trading_rounds_df.groupby('period')['price_per_unit'].mean()
price_volatility = trading_rounds_df.groupby('period')['price_per_unit'].std()
```

### 2. Agent Profitability Comparison
```python
agent_summary_df = model.get_trading_data()['agent_summary']
profit_by_agent = agent_summary_df.groupby('agent_type')['agent_surplus'].sum()
```

### 3. Market Activity Analysis
```python
period_summary_df = model.get_trading_data()['period_summary']
trading_volume = period_summary_df['total_electricity_traded']
market_activity = period_summary_df['total_trades']
```

## Integration with Existing Code

The new system is designed to be:
- **Backward Compatible**: Your existing `hourly_df` still works
- **Non-Intrusive**: Minimal changes to existing agent code
- **Additive**: Provides additional data without replacing current systems
- **Optional**: Can be used selectively based on analysis needs

## Files Created

1. **`data_tracker.py`**: Core data tracking functionality
2. **`analysis_example.py`**: Example analysis functions and workflows
3. **Enhanced `model.py`**: Updated with data tracking integration

This system gives you the comprehensive data structure you requested for tracking electricity trades, prices, and agent surplus at both the trading round and period levels, with safe storage and easy aggregation capabilities.
