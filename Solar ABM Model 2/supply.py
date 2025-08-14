import random
import numpy as np 
import pandas as pd 
import math 


class Supply:
    def __init__(self, c, q_sell_arr, agent_type, verbose=False):
        # constant per-unit cost, total supply endowment
        self.c = c
        self.q_sell = q_sell_arr
        self.remaining_supply = q_sell_arr #array of values to sell 

        # welfare tracking
        self.profit = 0.0
        self.sw_arr = []

        # last request
        self.txs_request = None
        self.agent_type = agent_type
        self.verbose = verbose

        self.q_dispatched = [0] #tracks the quantity dispatched each period 

        #AMM 
        self.amm_state = {}

        if self.verbose: print(f"[{self.agent_type}][INIT] c={c}, q_sell={self.q_sell[0]}, verbose={verbose}")

    def update_amm_state(self, E, M): 

        self.amm_state = {"reserve_x": E, "reserve_y": M}

    def marginal_price(self, E, M, e):
        """AMM marginal revenue when selling e of x into the E-reserve."""
        K = E * M
        return K / (E + e) ** 2

    def find_optimal_e(self, E, M,t):
        """
        Closed-form optimal sell quantity when MC=c:
          if c>0: solve c = K/(E+e)^2  =>  e = sqrt(K/c) − E
          if c<=0: supply everything up to your remaining endowment
        """
        if self.c <= 0:
            # zero (or negative) marginal cost → willing to sell all you have
            return self.remaining_supply[t]

        K = E * M
        e_star = np.sqrt(K / self.c) - E
        return max(0.0, e_star)

    def util_func(self, p, q):
        """(price – cost) × quantity."""
        return (p - self.c) * q

    def update_profit(self, p, q):
        utils = self.util_func(p, q)
        self.profit += utils
        self.sw_arr.append(utils)
        if self.verbose:
            print(f"[{self.agent_type}][update_profit] profit={utils:.4f}, total={self.profit:.4f}")

    def make_decision(self, E, M, t):
        """Compute e* and issue a sell_tokens_min_price order."""
        self.update_amm_state(E,M)

        if self.remaining_supply[t] <= 0:
            if self.verbose: print(f"[{self.agent_type}] No remaining supply.")
            return None

        e_star = self.find_optimal_e(E, M,t)
        e_req  = min(e_star, self.remaining_supply[t])

        if e_req <= 0:
            if self.verbose: print(f"[{self.agent_type}] No profitable quantity to sell.")
            return None

        p_star = self.marginal_price(E, M, e_req)

        if self.c <= 0: p_star = 0 # you are willing to accept any price 

        tx = {
            "decision":   "sell_tokens_min_price",
            "token":      "x",
            "quantity":   e_req,
            # specify total minimum y-tokens you must get back
            "min_price":  p_star * e_req,
            "agent_type": self.agent_type
        }
        if self.verbose:
            print(f"[{self.agent_type}][make_decision] e*={e_star:.4f}, e_req={e_req:.4f}, p*={p_star:.4f}")
            print(f"[{self.agent_type}][make_decision][VERBOSE] tx = {tx}")

        self.txs_request = tx
        return tx

    def transaction_result(self, txs_result,t):
        """Update profit and remaining_supply after execution."""
        e_sold      = self.txs_request["quantity"]
        m_received  = txs_result["quantity_returned"]
        avg_price   = m_received / e_sold

        self.q_dispatched.append(e_sold) #amount of quantity dispatched this period 

        #Figuring out Marginal Cost and Marginal Revenue 
        E = self.amm_state["reserve_x"]
        M = self.amm_state["reserve_y"]
        mr_rev = self.marginal_price(E, M, e_sold)
        
        self.update_profit(avg_price, e_sold)
        self.remaining_supply[t] -= e_sold
        if self.verbose:
            print(f"[{self.agent_type}][transaction_result] Marginal Cost = {self.c} Marginal Revenue = {mr_rev}")
            print(f"[{self.agent_type}][transaction_result] remaining = {self.remaining_supply[t]:.4f}")
