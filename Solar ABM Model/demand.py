import random
import numpy as np 
import pandas as pd 




class ElasticDemand: 
    def __init__(self, v_max_arr, q_max_arr, agent_type, verbose=False): 
        # state variables 
        self.v_max = v_max_arr # per kwh reservation price 
        self.q_max = q_max_arr # max demand or total demand per period (CONSTANT)
        
        self.agent_type = agent_type

        # FIXED: remaining_demand should be a copy of the array, not a reference
        self.remaining_demand = q_max_arr
        
        # welfare 
        self.profit = 0 
        self.sw_arr = []

        # txs request 
        self.txs_request = None 
        
        # debugging
        self.verbose = verbose

        #AMM state 
        self.amm_state = {}

        if self.verbose:
            print(f"[{self.agent_type}][INIT] v_max={self.v_max}, q_max={self.q_max}, agent_type={agent_type}, verbose={verbose}")
            print(f"[{self.agent_type}][INIT] remaining_demand initialized as: {self.remaining_demand}")

    def value_function(self,q,t): 
        """ Marginal Value for unit q purchased"""
        return self.v_max[t]*(1- q/self.q_max[t]) 
    
    def util_func(self, q, token_required,t):
        """ 
        Calculates the utility or consumer surplus from trade
        """

        # area under demand
        surplus = self.v_max[t] * (q - q**2/(2*self.q_max[t]))
        # total paid
        paid = token_required
        consumer_surplus = surplus - paid

        return consumer_surplus

    def find_optimal_e(self, E, M, v_max, q_max):
        """
        Solves for e in:
        e^3 -(q_max + 2E)e^2 + (E^2 + 2E q_max)e + q_max*E*(M/v_max - E) = 0
        Returns the unique real root 0 < e < E.
        """
        # Coefficients of the monic cubic
        coeffs = [
            1,
            -(q_max + 2*E),
            E**2 + 2*E*q_max,
            q_max*E*(M/v_max - E)
        ]
        roots = np.roots(coeffs)
        # Filter for a real root in (0, E)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < E]
        if not real_roots:
            if self.verbose: print("No valid root found in (0, E)")
        return real_roots 


    def update_profit(self, q, token_required,t): 
        utils = self.util_func(q, token_required,t)
        self.profit += utils
        self.sw_arr.append(utils)
        if self.verbose: print(f"[{self.agent_type}][update_profit] Updated profit by {utils}, total profit now {self.profit}")
        if self.verbose:
            print(f"[{self.agent_type}][update_profit][VERBOSE] token_required = {token_required}  q={q}, utils={utils}, sw_arr={self.sw_arr}")
    
    
    def make_decision(self, E, M,t):
        """Compute the optimal e* and issue a buy_tokens_max_price at that e."""
        if self.remaining_demand[t] <= 0:
            if self.verbose: print(f"[{self.agent_type}] No demand left.")
            return None
        
        #AMM_state 
        self.update_amm_state(E, M)
        e_star = self.find_optimal_e(E, M, self.v_max[t], self.q_max[t])
        
        if not e_star: return # if no real roots, we return 

        if self.verbose: print(e_star)
        e_star = e_star[0]
        e_req = min(e_star, self.remaining_demand[t])
        p_star = self.marginal_price(E, M, e_req)

        tx = {
            "decision":   "buy_tokens_max_price",
            "token":      "x",
            "quantity":   e_req,
            "max_price":  p_star * e_req, 
            "agent_type": self.agent_type
        }

        if self.verbose:
            print(f"[{self.agent_type}][make_decision] e*={e_star:.4f}, e_req={e_req:.4f}, p*={p_star:.4f}")
            print(f"[{self.agent_type}][make_decision][VERBOSE] tx = {tx}")

        self.txs_request = tx
        return tx
    
    def update_amm_state(self, E, M): 

        self.amm_state = {"reserve_x" :E, "reserve_y": M}
    def marginal_price(self, E, M, e):
        """Marginal cost MC(e) = d/d e [ total_cost(e) ]."""
        K = E * M
        return K / (E - e)**2

    def transaction_result(self, txs_result,t): 
        if self.verbose: print(f"[{self.agent_type}][transaction_result] txs_result: {txs_result}")
        e_requested = self.txs_request["quantity"]
        m_required = txs_result["quantity_needed"]
        
        #Calculating the Average Price 
        avg_price = m_required/e_requested
        if self.verbose: print(f"[{self.agent_type}][transaction_result] avg_price = {avg_price}")
        
        #Figuring out the Marginal value and Cost 
        E , M = self.amm_state["reserve_x"], self.amm_state["reserve_y"]
        marginal_cost = self.marginal_price(E, M, e_requested)
        marginal_value= self.value_function(e_requested, t)

        if self.verbose: print(f"[{self.agent_type}][transaction_result] Marginal Value = {marginal_value}, Marginal Cost = {marginal_cost}")

        self.update_profit(e_requested, m_required,t)
        #The position of this value function matters, because it relies on 
        #q_max_t not being updated, we need 
        self.v_max[t] = self.value_function(e_requested,t)
        self.remaining_demand[t] -= e_requested
        self.q_max[t] -= e_requested 
        

        if self.verbose: print(f"[{self.agent_type}][transaction_result] Updated remaining_demand: {self.remaining_demand[t]}")

        