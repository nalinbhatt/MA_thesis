import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import json

class TradingDataTracker:
    """
    Comprehensive data tracking system for the Energy AMM ABM model.
    Tracks trading data at both trading round and period levels.
    """
    
    def __init__(self,verbose = False):
        # Trading round level data (individual transactions)
        self.trading_rounds_data = []
        
        # Period level aggregated data
        self.period_data = []
        
        # Agent level tracking
        self.agent_data = defaultdict(list)
        
        # Current period and round counters
        self.current_period = 0
        self.current_round = 0
        
        # Temporary storage for current period
        self.current_period_trades = []

        self.verbose = verbose 
        
    def start_new_period(self, period_number, solar_generation=0):
        """Initialize a new trading period"""
        self.current_period = period_number
        self.current_round = 0
        self.current_period_trades = []
        
        if self.verbose: print(f"[DataTracker] Starting Period {period_number} with solar generation {solar_generation}")
        
    def record_trade(self, period, round_num, agent_type, trade_type, 
                    quantity_electricity, money_paid, price, 
                    amm_state_before, amm_state_after, agent_surplus=0,
                    trade_successful=True):
        """
        Record an individual trading round transaction
        
        Parameters:
        - period: Period number
        - round_num: Trading round within the period
        - agent_type: Type of agent (battery, solar, utility, demand)
        - trade_type: 'buy' or 'sell'
        - quantity_electricity: Amount of electricity traded (energy tokens)
        - money_paid: Amount of money paid/received (money tokens)
        - price: Effective price per unit
        - amm_state_before: AMM state before trade (dict with reserve_x, reserve_y)
        - amm_state_after: AMM state after trade
        - agent_surplus: Agent's surplus from this trade
        - agent_id: Unique identifier for the agent instance
        - trade_successful: Whether the trade was executed
        """
        
        trade_record = {
            'timestamp': datetime.now(),
            'period': period,
            'round': round_num,
            'agent_type': agent_type,
            'trade_type': trade_type,  # 'buy' or 'sell'
            'quantity_electricity': quantity_electricity,
            'money_paid': money_paid,
            'price_per_unit': price,
            'agent_surplus': agent_surplus,
            'trade_successful': trade_successful,
            
            # AMM state tracking
            'amm_reserve_x_before': amm_state_before['reserve_x'],
            'amm_reserve_y_before': amm_state_before['reserve_y'],
            'amm_price_before': amm_state_before['reserve_y'] / amm_state_before['reserve_x'],
            'amm_reserve_x_after': amm_state_after['reserve_x'],
            'amm_reserve_y_after': amm_state_after['reserve_y'],
            'amm_price_after': amm_state_after['reserve_y'] / amm_state_after['reserve_x'],
            
            # Price impact
            'price_impact': (amm_state_after['reserve_y'] / amm_state_after['reserve_x']) - 
                           (amm_state_before['reserve_y'] / amm_state_before['reserve_x'])
        }
        
        self.trading_rounds_data.append(trade_record)
        self.current_period_trades.append(trade_record)
        
        # Update agent-specific tracking
        self.agent_data[agent_type].append(trade_record)
        
        if self.verbose: print(f"[DataTracker] Recorded trade: {agent_type} {trade_type} {quantity_electricity} units at price {price:.4f}")
        
    def finalize_period(self, period, solar_generation, amm_final_state, agent_states):
        """
        Aggregate and finalize data for a completed period
        
        Parameters:
        - period: Period number
        - solar_generation: Solar generation for this period
        - amm_final_state: Final AMM state for the period
        - agent_states: Dictionary of final agent states {agent_type: state_dict}
        """
        
        # Aggregate trading data for this period
        period_trades = [t for t in self.current_period_trades if t['trade_successful']]
        
        # Calculate aggregated metrics
        total_electricity_traded = sum(t['quantity_electricity'] for t in period_trades)
        total_money_exchanged = sum(t['money_paid'] for t in period_trades)
        avg_price = total_money_exchanged / total_electricity_traded if total_electricity_traded > 0 else 0
        
        # Separate buy and sell transactions
        buy_trades = [t for t in period_trades if t['trade_type'] == 'buy']
        sell_trades = [t for t in period_trades if t['trade_type'] == 'sell']
        
        # Agent-specific aggregations
        agent_summaries = {}
        for agent_type in ['solar', 'utility', 'demand', 'informed_trader_battery', 'vfi_optimized_battery']:
            agent_trades = [t for t in period_trades if t['agent_type'] == agent_type]
            
            agent_summaries[agent_type] = {
                'total_electricity_bought': sum(t['quantity_electricity'] for t in agent_trades if t['trade_type'] == 'buy'),
                'total_electricity_sold': sum(t['quantity_electricity'] for t in agent_trades if t['trade_type'] == 'sell'),
                'total_money_paid': sum(t['money_paid'] for t in agent_trades if t['trade_type'] == 'buy'),
                'total_money_received': sum(t['money_paid'] for t in agent_trades if t['trade_type'] == 'sell'),
                'total_surplus': sum(t['agent_surplus'] for t in agent_trades),
                'num_trades': len(agent_trades),
                'avg_price_paid': np.mean([t['price_per_unit'] for t in agent_trades if t['trade_type'] == 'buy']) if any(t['trade_type'] == 'buy' for t in agent_trades) else 0,
                'avg_price_received': np.mean([t['price_per_unit'] for t in agent_trades if t['trade_type'] == 'sell']) if any(t['trade_type'] == 'sell' for t in agent_trades) else 0
            }
        
        period_summary = {
            'period': period,
            'solar_generation': solar_generation,
            'total_trades': len(period_trades),
            'successful_trades': len(period_trades),
            'total_electricity_traded': total_electricity_traded,
            'total_money_exchanged': total_money_exchanged,
            'avg_price': avg_price,
            'total_electricity_bought': sum(t['quantity_electricity'] for t in buy_trades),
            'total_electricity_sold': sum(t['quantity_electricity'] for t in sell_trades),
            'price_volatility': np.std([t['price_per_unit'] for t in period_trades]) if period_trades else 0,
            
            # AMM state
            'final_amm_reserve_x': amm_final_state['reserve_x'],
            'final_amm_reserve_y': amm_final_state['reserve_y'],
            'final_amm_price': amm_final_state['reserve_y'] / amm_final_state['reserve_x'],
            
            # Agent summaries
            **{f"{agent}_{key}": value for agent, summary in agent_summaries.items() 
               for key, value in summary.items()},
            
            # Agent states (if provided)
            **{f"{agent}_final_state": state for agent, state in agent_states.items()}
        }
        
        self.period_data.append(period_summary)
        if self.verbose: print(f"[DataTracker] Finalized Period {period}: {len(period_trades)} trades, {total_electricity_traded:.2f} electricity traded")
    
    def finalize_circular_period(self):
        if not self.period_data or not self.trading_rounds_data:
            if self.verbose: print("[DataTracker] No data to make circular.")
            return

        # Get first period number and last period number
        first_period_num = self.period_data[0]['period']
        last_period_num = self.period_data[-1]['period']
        new_period_num = last_period_num + 1

        # 1. Append period summary for new_period_num (copy of period 0)
        first_period_data = self.period_data[0].copy()
        first_period_data['period'] = new_period_num
        self.period_data.append(first_period_data)

        # 2. Append all trades from period 0, changing 'period' to new_period_num
        first_period_trades = [trade.copy() for trade in self.trading_rounds_data if trade['period'] == first_period_num]
        for trade in first_period_trades:
            trade['period'] = new_period_num
            trade['timestamp'] = datetime.now()  # update timestamp to now (optional)
        self.trading_rounds_data.extend(first_period_trades)

        if self.verbose: print(f"[DataTracker] Added circular period summary and trades for period {new_period_num}.")

    def get_trading_rounds_df(self):
        """Return DataFrame of all trading round data"""
        return pd.DataFrame(self.trading_rounds_data)
    
    def get_period_summary_df(self):
        """Return DataFrame of period-level aggregated data"""
        return pd.DataFrame(self.period_data)
    
    def get_agent_summary_df(self, agent_type=None):
        """Return DataFrame of agent-specific data"""
        if agent_type:
            return pd.DataFrame(self.agent_data[agent_type])
        else:
            # Return all agents' data with agent_type column
            all_data = []
            for agent, trades in self.agent_data.items():
                all_data.extend(trades)
            return pd.DataFrame(all_data)
    
    def export_to_csv(self, base_filename="simulation_results"):
        """Export all data to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export trading rounds data
        rounds_df = self.get_trading_rounds_df()
        if not rounds_df.empty:
            rounds_df.to_csv(f"{base_filename}_trading_rounds_{timestamp}.csv", index=False)
        
        # Export period summary data
        period_df = self.get_period_summary_df()
        if not period_df.empty:
            period_df.to_csv(f"{base_filename}_period_summary_{timestamp}.csv", index=False)
        
        # Export agent-specific data
        agent_df = self.get_agent_summary_df()
        if not agent_df.empty:
            agent_df.to_csv(f"{base_filename}_agent_trades_{timestamp}.csv", index=False)
        
        if self.verbose: print(f"[DataTracker] Exported data to CSV files with timestamp {timestamp}")

    def safe_stringify(self, value):
        if isinstance(value, (list, np.ndarray, pd.Series)):
            return json.dumps(list(value))  # Properly format as JSON string
        return value

    
    def get_summary_statistics(self):
        """Return summary statistics for the entire simulation"""
        rounds_df = self.get_trading_rounds_df()
        period_df = self.get_period_summary_df()
        
        if rounds_df.empty or period_df.empty:
            return {}
        
        # Precompute values for summary statistics
        total_periods = len(period_df)
        total_trades = len(rounds_df)
        total_electricity_traded = rounds_df['quantity_electricity'].sum()
        total_money_exchanged = rounds_df['money_paid'].sum()
        avg_price_overall = total_money_exchanged / total_electricity_traded if total_electricity_traded > 0 else 0
        price_volatility_overall = rounds_df['price_per_unit'].std()
        total_agent_surplus = rounds_df['agent_surplus'].sum()
        avg_trades_per_period = total_trades / total_periods if total_periods > 0 else 0
        successful_trade_rate = (rounds_df['trade_successful'].sum() / total_trades) * 100 if total_trades > 0 else 0
        final_amm_reserve_x = period_df["final_amm_reserve_x"]
        max_amm_reserve_x = final_amm_reserve_x.max()
        final_amm_reserve_y = period_df["final_amm_reserve_y"]
        max_amm_reserve_y = final_amm_reserve_y.max()

        p = period_df["final_amm_price"]
        s_t = period_df['solar_generation']
        q_d = period_df['demand_total_electricity_bought']
        q_s = period_df['solar_total_electricity_sold']
        q_u = period_df['utility_total_electricity_sold']
        q_b_inf = period_df['informed_trader_battery_total_electricity_sold'] - period_df['informed_trader_battery_total_electricity_bought']
        q_b_vfi = period_df['vfi_optimized_battery_total_electricity_sold'] - period_df["vfi_optimized_battery_total_electricity_bought"]


        soc_inf = [battery_dict['soc'] if isinstance(battery_dict, dict) and 'soc' in battery_dict else None 
               for battery_dict in period_df['battery_inf_final_state']] if 'battery_inf_final_state' in period_df.columns else [None]*total_periods
        soc_vfi = [battery_dict['soc'] if isinstance(battery_dict, dict) and 'soc' in battery_dict else None 
               for battery_dict in period_df['battery_vfi_final_state']] if 'battery_vfi_final_state' in period_df.columns else [None]*total_periods


        surplus_solar_ts = period_df["solar_total_surplus"]
        surplus_utility_ts = period_df["utility_total_surplus"]
        surplus_demand_ts = period_df["demand_total_surplus"]
        surplus_battery_inf_ts = period_df["informed_trader_battery_total_surplus"]
        surplus_battery_vfi_ts = period_df["vfi_optimized_battery_total_surplus"]
        surplus_total_ts = (period_df["solar_total_surplus"] + period_df["utility_total_surplus"] +
                    period_df["demand_total_surplus"] + period_df["vfi_optimized_battery_total_surplus"] +
                    period_df["informed_trader_battery_total_surplus"])

        total_surplus_battery_vfi = period_df["vfi_optimized_battery_total_surplus"].sum()
        total_surplus_battery_inf = period_df["informed_trader_battery_total_surplus"].sum()
        total_surplus_utility = period_df["utility_total_surplus"].sum()
        total_surplus_demand = period_df["demand_total_surplus"].sum()
        total_surplus_solar = period_df["solar_total_surplus"].sum()
        total_surplus_all = (total_surplus_solar + total_surplus_utility + total_surplus_demand +
                     total_surplus_battery_vfi + total_surplus_battery_inf)
        

        return {
            'total_periods': total_periods,
            'total_trades': total_trades,
            'total_electricity_traded': total_electricity_traded,
            'total_money_exchanged': total_money_exchanged,
            'avg_price_overall': avg_price_overall,
            'price_volatility_overall': price_volatility_overall,
            'total_agent_surplus': total_agent_surplus,
            'avg_trades_per_period': avg_trades_per_period,
            'successful_trade_rate': successful_trade_rate,
            'max_amm_reserve_x': max_amm_reserve_x,
            'max_amm_reserve_y': max_amm_reserve_y,
            'final_amm_reserve_y':  self.safe_stringify(final_amm_reserve_y),
            'final_amm_reserve_x':  self.safe_stringify(final_amm_reserve_x),
            'prices': self.safe_stringify(p), 
            's_t': self.safe_stringify(s_t),
            'q_d': self.safe_stringify(q_d),
            'q_s': self.safe_stringify(q_s),
            'q_u': self.safe_stringify(q_u),
            'q_b_inf': self.safe_stringify(q_b_inf),
            'q_b_vfi': self.safe_stringify(q_b_vfi),
            'soc_inf': self.safe_stringify(soc_inf),
            'soc_vfi': self.safe_stringify(soc_vfi),
            'surplus_solar_ts': self.safe_stringify(surplus_solar_ts),
            'surplus_utility_ts': self.safe_stringify(surplus_utility_ts),
            'surplus_demand_ts': self.safe_stringify(surplus_demand_ts),
            'surplus_battery_inf_ts': self.safe_stringify(surplus_battery_inf_ts),
            'surplus_battery_vfi_ts': self.safe_stringify(surplus_battery_vfi_ts),
            'surplus_total_ts': self.safe_stringify(surplus_total_ts),
            'total_surplus_battery_vfi': total_surplus_battery_vfi,
            'total_surplus_battery_inf': total_surplus_battery_inf,
            'total_surplus_utility': total_surplus_utility,
            'total_surplus_demand': total_surplus_demand,
            'total_surplus_solar': total_surplus_solar,
            'total_surplus_all': total_surplus_all
        }

        # return {
        #     'total_periods': total_periods,
        #     'total_trades': total_trades,
        #     'total_electricity_traded': total_electricity_traded,
        #     'total_money_exchanged': total_money_exchanged,
        #     'avg_price_overall': avg_price_overall,
        #     'price_volatility_overall': price_volatility_overall,
        #     'total_agent_surplus': total_agent_surplus,
        #     'avg_trades_per_period': avg_trades_per_period,
        #     'successful_trade_rate': successful_trade_rate,
        #     'max_amm_reserve_x': max_amm_reserve_x,
        #     'max_amm_reserve_y': max_amm_reserve_y,
        #     'final_amm_reserve_y':  final_amm_reserve_y,
        #     'final_amm_reserve_x':  final_amm_reserve_x,
        #     'prices':p, 
        #     's_t': s_t,
        #     'q_d': q_d,
        #     'q_s': q_s,
        #     'q_u': q_u,
        #     'q_b_inf': q_b_inf,
        #     'q_b_vfi': q_b_vfi,
        #     'soc_inf': soc_inf,
        #     'soc_vfi': soc_vfi,
        #     'surplus_battery_ts': surplus_battery_ts,
        #     'surplus_utility_ts': surplus_utility_ts,
        #     'surplus_demand_ts': surplus_demand_ts,
        #     'surplus_battery_inf_ts': surplus_battery_inf_ts,
        #     'surplus_battery_vfi_ts': surplus_battery_vfi_ts,
        #     'surplus_total_ts': surplus_total_ts,
        #     'total_surplus_battery_vfi': total_surplus_battery_vfi,
        #     'total_surplus_battery_inf': total_surplus_battery_inf,
        #     'total_surplus_utility': total_surplus_utility,
        #     'total_surplus_demand': total_surplus_demand,
        #     'total_surplus_solar': total_surplus_solar,
        #     'total_surplus_all': total_surplus_all
        # }
