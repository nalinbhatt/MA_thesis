# Example Usage of the Enhanced Data Tracking System

"""
This file demonstrates how to use the new comprehensive data tracking system
for the Energy AMM ABM model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_simulation_results(model):
    """
    Analyze the comprehensive results from a completed simulation
    
    Parameters:
    - model: The Model instance after simulation is complete
    """
    
    # Get all trading data
    trading_data = model.get_trading_data()
    
    # Extract DataFrames
    trading_rounds_df = trading_data['trading_rounds']
    period_summary_df = trading_data['period_summary']
    agent_summary_df = trading_data['agent_summary']
    summary_stats = trading_data['summary_statistics']
    
    print("=== Simulation Analysis ===")
    print(f"Total Periods: {summary_stats['total_periods']}")
    print(f"Total Trades: {summary_stats['total_trades']}")
    print(f"Total Electricity Traded: {summary_stats['total_electricity_traded']:.2f}")
    print(f"Average Price: {summary_stats['avg_price_overall']:.4f}")
    print(f"Total Agent Surplus: {summary_stats['total_agent_surplus']:.2f}")
    print(f"Successful Trade Rate: {summary_stats['successful_trade_rate']:.1f}%")
    
    # Analyze by agent type
    print("\n=== Agent Performance ===")
    for agent_type in ['solar', 'utility', 'demand', 'informed_trader_battery']:
        agent_trades = agent_summary_df[agent_summary_df['agent_type'] == agent_type]
        if not agent_trades.empty:
            total_surplus = agent_trades['agent_surplus'].sum()
            total_trades = len(agent_trades)
            avg_price = agent_trades['price_per_unit'].mean()
            print(f"{agent_type}: {total_trades} trades, Avg Price: {avg_price:.4f}, Total Surplus: {total_surplus:.2f}")
    
    # Period-by-period analysis
    print("\n=== Period Analysis ===")
    if not period_summary_df.empty:
        for _, period in period_summary_df.iterrows():
            print(f"Period {period['period']}: {period['total_trades']} trades, "
                  f"Avg Price: {period['avg_price']:.4f}, "
                  f"Electricity Traded: {period['total_electricity_traded']:.2f}")
    
    return trading_data

def plot_price_evolution(trading_rounds_df):
    """Plot price evolution over time"""
    if trading_rounds_df.empty:
        print("No trading data to plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot AMM price before each trade
    plt.subplot(1, 2, 1)
    plt.plot(trading_rounds_df['amm_price_before'], 'b-', alpha=0.7, label='AMM Price Before Trade')
    plt.plot(trading_rounds_df['price_per_unit'], 'r.', alpha=0.5, label='Trade Price')
    plt.xlabel('Trade Number')
    plt.ylabel('Price')
    plt.title('Price Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot price by period
    plt.subplot(1, 2, 2)
    period_prices = trading_rounds_df.groupby('period')['price_per_unit'].mean()
    plt.plot(period_prices.index, period_prices.values, 'go-', linewidth=2)
    plt.xlabel('Period')
    plt.ylabel('Average Price')
    plt.title('Average Price by Period')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_trading_volume(period_summary_df):
    """Plot trading volume by period"""
    if period_summary_df.empty:
        print("No period data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.bar(period_summary_df['period'], period_summary_df['total_electricity_traded'], 
            alpha=0.7, color='skyblue')
    plt.xlabel('Period')
    plt.ylabel('Total Electricity Traded')
    plt.title('Trading Volume by Period')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_agent_surplus(agent_summary_df):
    """Plot agent surplus by type"""
    if agent_summary_df.empty:
        print("No agent data to plot")
        return
    
    agent_surplus = agent_summary_df.groupby('agent_type')['agent_surplus'].sum()
    
    plt.figure(figsize=(10, 6))
    agent_surplus.plot(kind='bar', color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.xlabel('Agent Type')
    plt.ylabel('Total Surplus')
    plt.title('Total Surplus by Agent Type')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def generate_detailed_report(model, save_to_file=True):
    """Generate a detailed text report of the simulation"""
    trading_data = model.get_trading_data()
    
    report = []
    report.append("=" * 50)
    report.append("ENERGY AMM ABM SIMULATION REPORT")
    report.append("=" * 50)
    
    # Summary statistics
    summary_stats = trading_data['summary_statistics']
    report.append("\nSUMMARY STATISTICS:")
    for key, value in summary_stats.items():
        report.append(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Period analysis
    period_df = trading_data['period_summary']
    if not period_df.empty:
        report.append("\nPERIOD-BY-PERIOD ANALYSIS:")
        for _, period in period_df.iterrows():
            report.append(f"\nPeriod {period['period']}:")
            report.append(f"  Solar Generation: {period['solar_generation']:.2f}")
            report.append(f"  Total Trades: {period['total_trades']}")
            report.append(f"  Electricity Traded: {period['total_electricity_traded']:.2f}")
            report.append(f"  Average Price: {period['avg_price']:.4f}")
            report.append(f"  Price Volatility: {period['price_volatility']:.4f}")
    
    # Agent performance
    agent_df = trading_data['agent_summary']
    if not agent_df.empty:
        report.append("\nAGENT PERFORMANCE:")
        for agent_type in agent_df['agent_type'].unique():
            agent_data = agent_df[agent_df['agent_type'] == agent_type]
            report.append(f"\n{agent_type.replace('_', ' ').title()}:")
            report.append(f"  Total Trades: {len(agent_data)}")
            report.append(f"  Total Surplus: {agent_data['agent_surplus'].sum():.2f}")
            report.append(f"  Average Trade Price: {agent_data['price_per_unit'].mean():.4f}")
    
    report_text = "\n".join(report)
    
    if save_to_file:
        with open("simulation_report.txt", "w") as f:
            f.write(report_text)
        print("Report saved to 'simulation_report.txt'")
    
    return report_text

# Example usage in your notebook:
"""
# After running your simulation:
model.simulate()

# Analyze results
results = analyze_simulation_results(model)

# Generate plots
plot_price_evolution(results['trading_rounds'])
plot_trading_volume(results['period_summary'])
plot_agent_surplus(results['agent_summary'])

# Generate detailed report
report = generate_detailed_report(model)
print(report)

# Access specific data
battery_performance = model.get_agent_performance('informed_trader_battery')
print("Battery trading performance:")
print(battery_performance.head())

# Export all data to CSV
model.export_detailed_results("my_simulation_results")
"""
