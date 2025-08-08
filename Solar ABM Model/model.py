import random
import numpy as np 
import pandas as pd 
import math
from data_tracker import TradingDataTracker


class Model(): 

    def __init__(self, T, agent_list, amm_instance, trades_per_period,  s_t, verbose=False): 
        self.total_periods = T 
        self.agent_list = agent_list
        self.trades_per_period = trades_per_period
        self.amm_instance = amm_instance
        self.verbose = verbose
        self.s_t = s_t 
        
        # Initialize the comprehensive data tracker
        self.data_tracker = TradingDataTracker()
        
        self.amm_period_final_state = [] #logs the state of the amm at the end of the period
    
    def simulate(self, finalize_circular_period = False): 
        if self.verbose: print(f"[Model][simulate] Starting simulation for {self.total_periods} periods, {self.trades_per_period} trades per period.")



        for t in range(self.total_periods - 1): #model runs till 24 
            if self.verbose: print(f"\n[Model][simulate] === Period {t+1}/{self.total_periods} ===")
            
            # Initialize period in data tracker
            solar_gen_t = self.s_t[t] if t < len(self.s_t) else 0
            self.data_tracker.start_new_period(t, solar_gen_t)

            trade_count = 0
            for trade in range(self.trades_per_period):
                if self.verbose: print(f"[Model][simulate] -- Trade {trade_count+1}/{self.trades_per_period} --")
                random.shuffle(self.agent_list)
                if self.verbose:
                    print(f"[Model][simulate][VERBOSE] Agent list order: {[type(a).__name__ for a in self.agent_list]}")
                all_none = True
                
                for agent in self.agent_list:
                    if self.verbose: print(f"[Model][simulate] Agent: {type(agent).__name__}")
                    
                    # Get AMM state before trade
                    amm_state_before = self.amm_instance.request_info()
                    E = amm_state_before["reserve_x"]
                    M = amm_state_before["reserve_y"]
                    if self.verbose: print(f"[Model][simulate] AMM State: E={E}, M={M}")
                    if self.verbose:
                        print(f"[Model][simulate][VERBOSE] AMM full state: {amm_state_before}")

                    amm_txs_request = agent.make_decision(E, M, t)
                    if self.verbose: print(f"[Model][simulate] Agent decision: {amm_txs_request}")
                    
                    if amm_txs_request:
                        all_none = False
                        decision = amm_txs_request["decision"]
                        
                        if decision == "sell_tokens_min_price":
                            token = amm_txs_request["token"]
                            quantity = amm_txs_request["quantity"]
                            min_price = amm_txs_request["min_price"]
                            if self.verbose: print(f"[Model][simulate] Agent requests to SELL {quantity} {token} with min_price {min_price}")

                            txs_result = self.amm_instance.sell_tokens_min_price(token, quantity, min_price)
                            if self.verbose: print(f"[Model][simulate] AMM sell_tokens_min_price result: {txs_result}")
                            
                            # Get AMM state after trade
                            amm_state_after = self.amm_instance.request_info()
                            
                            if txs_result and "quantity_returned" in txs_result and txs_result["quantity_returned"] != "NoTrade": 
                                if self.verbose: print(f"[Model][simulate] Trade executed. Passing result to agent.")
                                
                                # Calculate metrics for data tracking
                                quantity_returned = txs_result["quantity_returned"] #Amount of M_tokens or money returned
                                price_per_unit = quantity_returned / quantity if quantity > 0 else 0
                                
                                
                                # Call transaction_result with appropriate parameters
                                if hasattr(agent, 'transaction_result'):
                                    try:
                                        agent.transaction_result(txs_result, t)
                                    except TypeError:
                                        # Some agents may not accept the 't' parameter
                                        agent.transaction_result(txs_result)
                                

                                agent_surplus = getattr(agent, 'sw_arr', [0])[-1] if hasattr(agent, 'sw_arr') and agent.sw_arr else 0
                                
                                # Record the trade
                                self.data_tracker.record_trade(
                                    period=t,
                                    round_num=trade_count,
                                    agent_type=agent.agent_type,
                                    trade_type='sell',
                                    quantity_electricity=quantity,
                                    money_paid=quantity_returned,
                                    price=price_per_unit,
                                    amm_state_before=amm_state_before,
                                    amm_state_after=amm_state_after,
                                    agent_surplus=agent_surplus,
                                    trade_successful=True
                                )

                            else:
                                if self.verbose: print(f"[Model][simulate] Trade not executed (NoTrade).")
                                # Record failed trade
                                self.data_tracker.record_trade(
                                    period=t,
                                    round_num=trade_count,
                                    agent_type=agent.agent_type,
                                    trade_type='sell',
                                    quantity_electricity=quantity,
                                    money_paid=0,
                                    price=0,
                                    amm_state_before=amm_state_before,
                                    amm_state_after=amm_state_before,  # No change
                                    agent_surplus=0,
                                    trade_successful=False
                                )

                        elif decision == "buy_tokens_max_price":
                            token = amm_txs_request["token"] 
                            quantity = amm_txs_request["quantity"]
                            max_price = amm_txs_request["max_price"]
                            if self.verbose: print(f"[Model][simulate] Agent requests to BUY {quantity} {token} with max_price {max_price}")

                            txs_result = self.amm_instance.buy_tokens_max_price(token, quantity, max_price)
                            if self.verbose: print(f"[Model][simulate] AMM buy_tokens_max_price result: {txs_result}")

                            # Get AMM state after trade
                            amm_state_after = self.amm_instance.request_info()
                            
                            if txs_result and "quantity_needed" in txs_result and txs_result["quantity_needed"] != "NoTrade": 
                                if self.verbose: print(f"[Model][simulate] Trade executed. Passing result to agent.")
                                
                                # Calculate metrics for data tracking
                                quantity_needed = txs_result["quantity_needed"]
                                price_per_unit = quantity_needed / quantity if quantity > 0 else 0
                               
                                # Call transaction_result with appropriate parameters
                                if hasattr(agent, 'transaction_result'):
                                    try:
                                        agent.transaction_result(txs_result, t)
                                    except TypeError:
                                        # Some agents may not accept the 't' parameter
                                        agent.transaction_result(txs_result)
                                
                                agent_surplus = getattr(agent, 'sw_arr', [0])[-1] if hasattr(agent, 'sw_arr') and agent.sw_arr else 0
                                
                                # Record the trade
                                self.data_tracker.record_trade(
                                    period=t,
                                    round_num=trade_count,
                                    agent_type=agent.agent_type,
                                    trade_type='buy',
                                    quantity_electricity=quantity,
                                    money_paid=quantity_needed,
                                    price=price_per_unit,
                                    amm_state_before=amm_state_before,
                                    amm_state_after=amm_state_after,
                                    agent_surplus=agent_surplus,
                                    trade_successful=True
                                )
                                
                            else:
                                if self.verbose: print(f"[Model][simulate] Trade not executed (NoTrade).")
                                # Record failed trade
                                self.data_tracker.record_trade(
                                    period=t,
                                    round_num=trade_count,
                                    agent_type=agent.agent_type,
                                    trade_type='buy',
                                    quantity_electricity=quantity,
                                    money_paid=0,
                                    price=0,
                                    amm_state_before=amm_state_before,
                                    amm_state_after=amm_state_before,  # No change
                                    agent_surplus=0,
                                    trade_successful=False
                                )
                        elif decision == "buy_tokens":
                            token = amm_txs_request["token"] 
                            quantity = amm_txs_request["quantity"]
                  
                            if self.verbose: print(f"[Model][simulate] Agent requests to BUY {quantity} {token}")

                            txs_result = self.amm_instance.buy_tokens(token, quantity)
                            if self.verbose: print(f"[Model][simulate] AMM buy_tokens result: {txs_result}")

                            # Get AMM state after trade
                            amm_state_after = self.amm_instance.request_info()
                            
                            if txs_result and "quantity_needed" in txs_result and txs_result["quantity_needed"] != "NoTrade": 
                                if self.verbose: print(f"[Model][simulate] Trade executed. Passing result to agent.")
                                
                                # Calculate metrics for data tracking
                                quantity_needed = txs_result["quantity_needed"]
                                price_per_unit = quantity_needed / quantity if quantity > 0 else 0
                                
                                # Call transaction_result with appropriate parameters
                                if hasattr(agent, 'transaction_result'):
                                    try:
                                        agent.transaction_result(txs_result, t)
                                    except TypeError:
                                        # Some agents may not accept the 't' parameter
                                        agent.transaction_result(txs_result)
                                
                                agent_surplus = getattr(agent, 'sw_arr', [0])[-1] if hasattr(agent, 'sw_arr') and agent.sw_arr else 0
                                
                                # Record the trade
                                self.data_tracker.record_trade(
                                    period=t,
                                    round_num=trade_count,
                                    agent_type=agent.agent_type,
                                    trade_type='buy',
                                    quantity_electricity=quantity,
                                    money_paid=quantity_needed,
                                    price=price_per_unit,
                                    amm_state_before=amm_state_before,
                                    amm_state_after=amm_state_after,
                                    agent_surplus=agent_surplus,
                                    trade_successful=True
                                )
                                
                            else:
                                if self.verbose: print(f"[Model][simulate] Trade not executed (NoTrade).")
                                # Record failed trade
                                self.data_tracker.record_trade(
                                    period=t,
                                    round_num=trade_count,
                                    agent_type=agent.agent_type,
                                    trade_type='buy',
                                    quantity_electricity=quantity,
                                    money_paid=0,
                                    price=0,
                                    amm_state_before=amm_state_before,
                                    amm_state_after=amm_state_before,  # No change
                                    agent_surplus=0,
                                    trade_successful=False
                                )
           
                        else: 
                            if self.verbose: print(f"[Model][simulate] ERROR: Unknown decision type: {decision}")
                            raise Exception("Unknown decision type encountered in Model.simulate()")
                    else:
                        if self.verbose: print(f"[Model][simulate] No trade decision from agent.")

                if all_none: #each agent responded with No trade, we want to move to next period 
                    break
                    
                trade_count += 1 


            # Collect agent states
            agent_states = {}
            for agent in self.agent_list: 
                if agent.agent_type == "solar":
                    q_s = agent.q_dispatched[t] if t < len(agent.q_dispatched) else 0
                    agent_states['solar'] = {
                        'q_s': q_s,
                        'remaining_supply': getattr(agent, 'remaining_supply', 0),
                        'profit': getattr(agent, 'profit', 0)
                    }
                elif agent.agent_type == "utility":
                    q_u = agent.q_dispatched[t] if t < len(agent.q_dispatched) else 0
                    agent_states['utility'] = {
                        'q_u': q_u,
                        'remaining_supply': getattr(agent, 'remaining_supply', 0),
                        'profit': getattr(agent, 'profit', 0)
                    }
                elif agent.agent_type == "demand": 
                    q_d = getattr(agent, 'q_purchased', [0])
                    q_d = q_d[t] if t < len(q_d) else 0
                    agent_states['demand'] = {
                        'q_d': q_d,
                        'remaining_demand': getattr(agent, 'remaining_demand', 0),
                        'profit': getattr(agent, 'profit', 0)
                    }
                elif agent.agent_type == "informed_trader_battery":
                    q_b = agent.q_exchanged[t] if t < len(agent.q_exchanged) else 0
                    C = agent.C
                    agent_states['battery_inf'] = {
                        'q_b_inf': q_b,
                        'soc': C,
                        'profit': getattr(agent, 'profit', 0)
                    }

                elif agent.agent_type == "vfi_optimized_battery":
                    q_b = agent.q_exchanged[t] if t < len(agent.q_exchanged) else 0
                    C = agent.C
                    agent_states['battery_vfi'] = {
                        'q_b_vfi': q_b,
                        'soc': C,
                        'profit': getattr(agent, 'profit', 0)
                    }
                
                else:
                    if self.verbose: print(f"Warning: unrecognized agent {agent.agent_type}")
            
            # Finalize period in data tracker
            amm_final_state = self.amm_instance.request_info()
            self.data_tracker.finalize_period(
                period=t,
                solar_generation=solar_gen_t,
                amm_final_state=amm_final_state,
                agent_states=agent_states
            )
            
        # Export comprehensive data at the end of simulation
        #self.data_tracker.export_to_csv("energy_abm_simulation")

        #Appends the first period trades at the end of the simulation to make the 
        #simulation circular the 0th hour is equivalant to the T+1st hour 
        if finalize_circular_period: self.data_tracker.finalize_circular_period()

        # Print summary statistics
        #summary_stats = self.data_tracker.get_summary_statistics()
       # if self.verbose: print("\n[Model][simulate] Simulation Summary Statistics:")
        #for key, value in summary_stats.items():
        #    if self.verbose: print(f"  {key}: {value}")

        if self.verbose: print("[Model][simulate] Simulation complete.")
    
    def get_trading_data(self):
        """Return comprehensive trading data"""
        return {
            'trading_rounds': self.data_tracker.get_trading_rounds_df(),
            'period_summary': self.data_tracker.get_period_summary_df(),
            'agent_summary': self.data_tracker.get_agent_summary_df(),
            'summary_statistics': self.data_tracker.get_summary_statistics()
        }
    
    def get_agent_performance(self, agent_type=None):
        """Get performance metrics for specific agent type or all agents"""
        return self.data_tracker.get_agent_summary_df(agent_type)
    
    def export_detailed_results(self, filename_prefix="detailed_results"):
        """Export all detailed results to CSV files"""
        self.data_tracker.export_to_csv(filename_prefix)


