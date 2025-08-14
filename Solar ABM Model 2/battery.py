import random
import numpy as np 
import pandas as pd 



class InformedTraderBattery: 

    def __init__(self, C_init, C_max, agent_type , s_t, v_max, q_max, c_u, q_u_max, q_b_max, verbose = False): 

        self.C_init = C_init # this is the initial and final state of the battery
        self.C_max = C_max 
        self.C = C_init

        self.agent_type = agent_type

        self.verbose = verbose
        
        self.q_b_max = q_b_max #Maximum amount that the battery charge or discharge in a given period


        # welfare tracking
        self.profit = 0.0
        self.sw_arr = []
        self.amm_state = {}

        self.q_exchanged = np.zeros(len(s_t)) #same number of indices as s_t
        
        if self.verbose: print(f"[{self.agent_type}][init] ", 
                               f"C_init = {self.C_init}",
                               f"C_max = {self.C_max}",
                               f"agent_type = {self.agent_type}",)
            
        #buyer and seller params to calc equilibrium prices 
        self.s_t = s_t  #array of solar generation in all periods t 
        self.v_max = v_max # max value of demand 
        self.q_max = q_max # max quantity demanded 
        self.c_u = c_u #per unit marginal cost of supply 
        self.q_u_max = q_u_max # max procurement by the utility this should be q_max

    def util_func(self, p, q):
        """price * quantity."""
        return p * q
    
    def update_profit(self, p, q):

        if self.verbose: print(f"[{self.agent_type}] p = {p}, {q}")
        utils = self.util_func(p, q)
        self.profit += utils
        self.sw_arr.append(utils)
        if self.verbose:
            print(f"[{self.agent_type}][update_profit] profit={utils:.4f}, total={self.profit:.4f}")

    def update_amm_state(self, E, M): 
        """reserve_x are the energy reserves, and reserve y are the money reserves"""
        self.amm_state = {"reserve_x": E, "reserve_y": M}

    def make_decision(self, E, M, t):
        """
        1) Compute spot p_t and forecast p_{t+1}.
        2) Decide buy/sell/hold and q_req.
        3) Execute via amm.buy_tokens_max_price or amm.sell_tokens_min_price.
        """
        self.update_amm_state(E, M)
        
        # 1) true spot price
        p_t = M / E

        K = E * M
        # 2) forecast next period equilibrium *demand-side* price
        #    you can still use your comp_eq_price to get p_{t+1} from solar+utility stack
        p_nxt, _ = self.comp_eq_price_quantity(t + 1)

        if self.verbose: print(f"[{self.agent_type}][make_decision] E={E}, M={M}, t={t}, p_t={p_t}, K={K}, p_nxt={p_nxt}, SOC={self.C}, C_max={self.C_max}")

        if self.verbose:
            print(f"[{self.agent_type}] p_t={p_t:.4f}, p_{t+1}={p_nxt:.4f}, SOC={self.C:.4f}")

        # 1) BUY (charge)
        if (p_nxt > p_t) and (p_nxt > 0):
           
            # correct unconstrained q* for buying E-tokens
            q_uncon = E - np.sqrt(K / p_nxt)

            # check if buying this much would violate q_exchanged bound
            max_q_allowed = self.q_exchanged[t] + self.q_b_max  # buying lowers q_exchanged
            q_bounded = max_q_allowed  # since q_exchanged[t] - q_req >= -q_b_max → q_req <= q_b_max + q_exchanged[t]

            # clamp to available battery capacity and to not drain pool completely
            #also ensure quantity bought is less than max charge allowed per period q_b_max
            q_req = float(np.clip(q_uncon, 0.0, min(self.C_max - self.C, E - 1e-8,  q_bounded)))

           
            #q_req = float(np.clip(q_uncon, 0.0, min(self.C_max - self.C, E - 1e-8, self.q_b_max)))
            
            if q_req <= 0:
                return None

            # worst-case final per-unit cost at e = q_req
            p_final   = K / (E - q_req)**2
            max_total = p_final * q_req

            tx = {
            "decision":   "buy_tokens_max_price",
            "token":      "x",
            "quantity":   q_req,
            "max_price":  max_total,
            "agent_type": self.agent_type
            }
            
            self.txs_request = tx
            return tx

        # 2) SELL (discharge)
        elif p_nxt < p_t:
            if p_nxt > 0: 
                q_uncon = np.sqrt(K / p_nxt) - E
            else: # if next period price is 0 we sell all our holdings. 
                q_uncon = self.C

            if self.verbose: print(f"[{self.agent_type}][make_decision][SELL] q_uncon={q_uncon}")

            # check if selling this much would violate q_exchanged bound
            max_q_allowed = self.q_b_max - self.q_exchanged[t]  # selling increases q_exchanged
            q_bounded = max_q_allowed  # since q_exchanged[t] + q_req <= q_b_max → q_req <= q_b_max - q_exchanged[t]

            # updated q_req with additional constraint
            q_req = float(np.clip(q_uncon, 0.0, min(self.C,  q_bounded)))

            # clamp into [0, SOC], ensures max amount sold is less than or equal to q_b_max
            #q_req = float(np.clip(q_uncon, 0.0, min(self.C, self.q_b_max)))

            if self.verbose: print(f"[{self.agent_type}][make_decision][SELL] q_req={q_req}")

            if q_req <= 0:
                if self.verbose: print(f"[{self.agent_type}][make_decision][SELL] q_req <= 0, returning None")
                return None
            
            p_final = K / (E + q_req)**2
            min_total = p_final * q_req

            tx = {
            "decision":   "sell_tokens_min_price",
            "token":      "x",
            "quantity":   q_req,
            "min_price":  min_total,
            "agent_type": self.agent_type
            }
            self.txs_request = tx 
            if self.verbose: print(f"[{self.agent_type}][make_decision][SELL] tx={tx}")
            return tx

        # 3) HOLD
        else:
            if self.verbose: print(f"[{self.agent_type}][make_decision][HOLD] No action taken.")
            if self.verbose:
                print(f"[{self.agent_type}] hold")
            return None
    
    def transaction_result(self, txs_result,t):
        """Update profit and remaining_supply after execution."""

        if self.verbose: print(f"[{self.agent_type}][transaction_result] txs_result: {txs_result}")

        #if your previous decision was to sell tokens
        if self.txs_request["decision"] == "sell_tokens_min_price":
            e_sold      = self.txs_request["quantity"]
            m_received  = txs_result["quantity_returned"]
            avg_price   = m_received / e_sold

            self.q_exchanged[t] += e_sold
            if self.verbose: print(f"[{self.agent_type}][transaction_result] e_sold = {e_sold}")
            if self.verbose: print(f"[{self.agent_type}][transaction_result] m_received = {m_received}")
            if self.verbose: print(f"[{self.agent_type}][transaction_result] avg_price = {avg_price}")

            self.update_profit(avg_price, e_sold)

            #NOTE: could be a bug if e_sold is greater than self.C
            self.C -= e_sold #you update your battery state to reflect amount of e that was discharged

            if self.verbose: print(f"[{self.agent_type}][transaction_result] Updated SOC (C) = {self.C} q_b_net = {self.q_exchanged[t]}")

        elif self.txs_request["decision"] == "buy_tokens_max_price":
            e_requested = self.txs_request["quantity"]
            m_required = txs_result["quantity_needed"]

            if self.verbose: print(f"[{self.agent_type}][transaction_result] e_requested = {e_requested}")
            if self.verbose: print(f"[{self.agent_type}][transaction_result] m_required = {m_required}")

            #Calculating the Average Price 
            avg_price = m_required / e_requested

            self.q_exchanged[t] -= e_requested 

            if self.verbose: print(f"[{self.agent_type}][transaction_result] avg_price = {avg_price}")
            
            self.update_profit(-avg_price, e_requested)
            self.C += e_requested 
            self.C = min(self.C_max, self.C) #incase there is an overcharge 

            if self.verbose: print(f"[{self.agent_type}][transaction_result] Updated SOC (C) = {self.C} , q_b_net = {self.q_exchanged[t]}")

        else: 
            raise Exception(f"[{self.agent_type}][transaction_result] Unknown txs_request decision {self.txs_request}")
                
    
    def comp_eq_price_quantity(self, t):
        """
        Find (p*, q*) solving demand v(q) = supply stack:
        - Solar free up to s_t[t]
        - Utility at cost c_u up to q_u_max
        Demand: v(q)=v_max*(1-q/q_max)
        """

        # 1) If solar alone covers full demand at p=0
        if self.q_max <= self.s_t[t]:
            return 0.0, self.q_max

        # 2) If solar < full demand, but the next marginal value is below c_u:
        #    v(s_t) = v_max*(1 - s_t/q_max) < c_u
        #    nobody is willing to pay the utility price, so
        #    market clears on free solar alone

        if self.verbose: print(self.s_t[t])

        v_at_solar = self.v_max * (1 - self.s_t[t] / self.q_max)
        if v_at_solar < self.c_u:
            return 0.0, self.s_t[t] #market clears at last marginal supply 

        # 3) Otherwise demand at price c_u (utility kicks in)
        q_star = self.q_max * (1 - self.c_u / self.v_max)
        total_supply = self.s_t[t] + self.q_u_max[t]

        # 3a) If that demand exceeds total capacity, clear at choke price
        if q_star > total_supply:
            #p_star = self.v_max * (1 - total_supply / self.q_max)
            p_star = self.c_u
            return p_star, total_supply

        # 3b) Otherwise clear at (c_u, q_star)
        return self.c_u, q_star
    
        

class OptimalPolicyBattery:

    def __init__(self, v_max,q_max, c_u, q_u_max, C_init, C_final, C_max, s_t, beta, T, q_b_max, agent_type, verbose = False):

        self.v_max = v_max
        self.q_max = q_max
        self.c_u = c_u
        self.q_u_max = q_u_max
        self.C_init = C_init
        self.C_final = C_final
        self.C_max = C_max
        self.C = C_init #this the charge that is updated accross time
        self.s_t = s_t
        self.beta = beta
        self.T = T #number of time periods in the model 
        self.agent_type = agent_type
        self.txs_request = {} #dictionary that holds the last transaction requested 

        self.q_exchanged = np.zeros(len(s_t)) #same number of indices as s_t
        self.q_b_max = q_b_max

        # welfare tracking
        self.profit = 0.0
        self.sw_arr = []
        self.amm_state = {}

        self.verbose = verbose

        self.compute_optimal_strategy()

    def util_func(self, p, q):
        """price * quantity."""
        return p * q
    
    def update_profit(self, p, q):

        if self.verbose: print(f"[{self.agent_type}] p = {p}, {q}")
        utils = self.util_func(p, q)
        self.profit += utils
        self.sw_arr.append(utils)
        if self.verbose:
            print(f"[{self.agent_type}][update_profit] profit={utils:.4f}, total={self.profit:.4f}")

    def update_amm_state(self, E, M): 
        """reserve_x are the energy reserves, and reserve y are the money reserves"""
        self.amm_state = {"reserve_x": E, "reserve_y": M}

    
    def compute_optimal_strategy(self):

        # Setup grids
        C_grid = np.arange(0, self.C_max + 1) #integer 
        q_b_grid = np.arange(-self.q_b_max,self.q_b_max + 1) #integer 

        
        # Initialize value function arrays
        V = np.zeros((len(C_grid), self.T+1))
        P = np.zeros_like(V)
        QS = np.zeros_like(V)
        QB = np.zeros_like(V)
        QU = np.zeros_like(V)

        # Terminal constraint: return to initial SOC
        V[:, self.T] = -np.inf
        i_init = np.searchsorted(C_grid, self.C_init)
        V[i_init, self.T] = 0

        # Value function iteration loop (same as before)
        for t in reversed(range(self.T)):
            solar = self.s_t[t]
            for i, C in enumerate(C_grid):
                best = -np.inf
                bp = bqs = bqb = bqu = 0.0
                for q_b_val in q_b_grid:
                    C_next = C - q_b_val
                    
                    if t == self.T - 1 and not np.isclose(C_next, self.C_init, atol=0):
                        continue
                    if not (0 <= C_next <= self.C_max):
                        continue
                    
                    if q_b_val < 0:
                        p_star, q_d, q_s, q_b_ex, q_u = self.equilibrium_with_battery_bid(
                            solar, -q_b_val, self.v_max, self.q_max, self.c_u, self.q_u_max
                        )
                        q_b = -q_b_ex

                        # Skip if partial fill, partial fills move the VFI away from the grid 
                        if not np.isclose(q_b_ex, -q_b_val, atol=1e-6):
                            continue
                    
                    else:
                        p_star, q_d, q_s, q_b_ex, q_u = self.equilibrium_with_battery_offer(
                            solar, q_b_val, self.v_max, self.q_max, self.c_u, self.q_u_max
                        )
                        q_b = q_b_ex

                         # Skip if partial fill, partial fills move the VFI away from the grid 
                        if not np.isclose(q_b_ex, q_b_val, atol=1e-6):
                            continue
                        
                    inst = p_star * q_b
                    idx = np.searchsorted(C_grid, C_next)
                    idx = min(max(idx, 0), len(C_grid)-1)
                    val = inst + self.beta*V[idx, t+1]
                    if val > best:
                        best = val
                        bp, bqs, bqb, bqu = p_star, q_s, q_b, q_u
                V[i, t] = best
                P[i, t], QS[i, t], QB[i, t], QU[i, t] = bp, bqs, bqb, bqu

        # Forward pass for optimal policy
        i = i_init
        C = self.C_init
        prices = []
        socs = []
        q_s_list = []
        q_b_list = []
        q_u_list = []

        for t in range(self.T):
            socs.append(C)
            prices.append(P[i, t])
            q_s_list.append(QS[i, t])
            q_b_list.append(QB[i, t])
            q_u_list.append(QU[i, t])

            #Advance to next state 
            # C = C - QB[i, t]
            # i = np.searchsorted(C_grid, C)
            # i = min(max(i, 0), len(C_grid)-1)

            #valid 
            q_b_t = QB[i, t]
            C_next = C - q_b_t

            # Clip to [0, C_max] range to ensure SOC validity
            C_next = max(0, min(self.C_max, C_next))

            # Snap to nearest valid grid point
            i = np.abs(C_grid - C_next).argmin()
            C = C_grid[i]  # Update to snapped value

        self.optimal_q_b = q_b_list

        if self.verbose: print(f"[{self.agent_type}][compute_optimal_strategy] self.optimal_q_b: {self.optimal_q_b}")


    def make_decision(self, E, M, t): 
        """ Sells and buys tokens according 
            to optimal strategy 
        """
        q_b  = self.optimal_q_b[t]

        if self.verbose: print(f"q_b = {q_b}")
        if q_b  > 0: 
            q_b = min(q_b, self.C) # the amount you sell cannot be greater than current charge

            if self.verbose: print(f"q_b clipped = {q_b}")
            tx = {
                "decision": "sell_tokens_min_price",
                "token": "x", 
                "quantity": q_b, 
                "min_price": 0 ,
                "agent_type": self.agent_type}
            
            self.txs_request = tx
        
        elif q_b < 0: 
            q_b = -q_b #convert it into positive, only allowed to buy positive quantities 
            if q_b >= E: 
                # If q_b is negative (buying), but exceeds available E, buy as much as possible with a small tolerance
                eps = 1e-6  # small tolerance
                tx = {
                    "decision": "buy_tokens",
                    "token": "x", 
                    "quantity": max(E - eps, 0.0) #ensure you don't drain the pool
                }
                self.txs_request = tx
            
            else: 
                tx = {
                    "decision": "buy_tokens",
                    "token": "x", 
                    "quantity": q_b #ensure you don't drain the pool
                }
                self.txs_request = tx

        else: 
            return  
        
        return tx 


    def transaction_result(self, txs_result, t): 
        """ Battery agent processes to result of it's transaction"""

        if self.verbose: print(f"[{self.agent_type}][transaction_result] txs_result: {txs_result}")

        #if your previous decision was to sell tokens
        if self.txs_request["decision"] == "sell_tokens_min_price":
            e_sold      = self.txs_request["quantity"]
            m_received  = txs_result["quantity_returned"]
            avg_price   = m_received / e_sold

            self.q_exchanged[t] += e_sold

            if self.verbose: print(f"[{self.agent_type}][transaction_result] e_sold = {e_sold}")
            if self.verbose: print(f"[{self.agent_type}][transaction_result] m_received = {m_received}")
            if self.verbose: print(f"[{self.agent_type}][transaction_result] avg_price = {avg_price}")

            self.update_profit(avg_price, e_sold)

            #NOTE: could be a bug if e_sold is greater than self.C
            self.C -= e_sold #you update your battery state to reflect amount of e that was discharged

            self.optimal_q_b[t] -= e_sold

            if self.verbose: print(f"[{self.agent_type}][transaction_result] Updated SOC (C) = {self.C} q_b_net = {self.q_exchanged[t]}")

        elif self.txs_request["decision"] == "buy_tokens":
            e_requested = self.txs_request["quantity"]
            m_required = txs_result["quantity_needed"]

            if self.verbose: print(f"[{self.agent_type}][transaction_result] e_requested = {e_requested}")
            if self.verbose: print(f"[{self.agent_type}][transaction_result] m_required = {m_required}")

            #Calculating the Average Price 
            avg_price = m_required / e_requested

            self.q_exchanged[t] -= e_requested 

            if self.verbose: print(f"[{self.agent_type}][transaction_result] avg_price = {avg_price}")
            
            self.optimal_q_b[t] += e_requested 

            self.update_profit(-avg_price, e_requested)
            self.C += e_requested 
            self.C = min(self.C_max, self.C) #incase there is an overcharge, we consider the rest to be wasted or curtailed, or simply never settled

            if self.verbose: print(f"[{self.agent_type}][transaction_result] Updated SOC (C) = {self.C} , q_b_net = {self.q_exchanged[t]}")

        else: 
            raise Exception(f"[{self.agent_type}][transaction_result] Unknown txs_request decision {self.txs_request}")
                

    def equilibrium_with_battery_offer(
        self,
        s_t: float,
        q_b_offer: float,
        v_max: float,
        q_max: float,
        c_u: float,
        q_u_max: float
    ):
        """
        Compute equilibrium price/quantities given:
        - solar supply at price 0: s_t
        - battery supply at price 0: q_b_offer
        - utility supply at price c_u: capacity q_u_max
        - linear inverse‐demand v(q)=v_max*(1 - q/q_max)

        Returns:
        p_star, q_d, q_s, q_b, q_u
        """
        # 1) total zero‐price supply
        free_supply = s_t + q_b_offer

        # 2) demand at zero price
        q_d0 = q_max  # v(0)=v_max>0 ⇒ q at p=0 is q_max

        # 3) if demand <= free_supply, clear at p=0
        if q_d0 <= free_supply:
            q_d = q_d0
            p_star = 0.0
            # allocate: solar up to its capacity
            q_s = min(s_t, q_d)
            q_b = max(0.0, q_d - q_s)  # avoid negatives due to precision
            q_u = 0.0
            return p_star, q_d, q_s, q_b, q_u

        # 4) check whether the marginal value at free_supply is below c_u
        #    if so, price remains 0 but quantity = free_supply
        v_at_free = v_max * (1 - free_supply / q_max)
        if v_at_free < c_u:
            q_d = free_supply
            p_star = 0.0
            q_s = s_t
            q_b = q_b_offer
            q_u = 0.0
            return p_star, q_d, q_s, q_b, q_u

        # 5) otherwise bring utility online at cost c_u
        #    demand at p=c_u
        q_d_cu = q_max * (1 - c_u / v_max)
        total_capacity = free_supply + q_u_max

        # 5a) if that demand exceeds total capacity, capacity‐constrained
        if q_d_cu > total_capacity:
            q_d = total_capacity
            p_star = c_u
        else:
            q_d = q_d_cu
            p_star = c_u

        # 6) allocate across sources in dispatch order
        #    solar first, then battery, then utility
        remaining = q_d
        q_s = min(s_t, remaining)
        remaining -= q_s

        q_b = min(q_b_offer, remaining)
        remaining -= q_b

        q_u = min(q_u_max, max(0.0, remaining))  # enforce non-negativity


        return p_star, q_d, q_s, q_b, q_u


    def equilibrium_with_battery_bid(
        self, 
        s_t: float,
        q_b_bid: float,
        v_max: float,
        q_max: float,
        c_u: float,
        q_u_max: float
    ):
        """
        Supply stack:
        1) Solar at p=0, capacity s_t
        2) Utility at p=c_u, capacity q_u_max

        Demand:
        1) Battery bids q_b_bid at p=0 (charges only if price=0 and solar > consumer)
        2) Consumers with inverse‐demand v(q)=v_max*(1 - q/q_max)

        Returns:
        p_star : equilibrium price
        q_d    : consumer quantity
        q_s    : solar dispatched
        q_b    : battery charged (bid filled)
        q_u    : utility dispatched
        """
        # Consumer demand at zero price
        q_d0 = q_max  # since v(q)=0 => q=q_max
        
        # 1) Check if market clears at p=0 on solar alone:
        if s_t >= q_d0:
            # Solar > consumer demand: price=0, consumers get q_max, battery fills from excess
            p_star = 0.0
            q_d = q_d0
            # battery charges only from leftover solar
            q_b = min(q_b_bid, s_t - q_d)
            q_s = q_d + q_b 

            q_u = 0.0
            return p_star, q_d, q_s, q_b, q_u

        # 2) Check if solar alone is binding but consumers' marginal value at s_t is below utility cost:
        #    v(s_t) < c_u ⇒ no one pays c_u, so price still 0, consumers get only solar, battery can't charge
        v_at_solar = v_max * (1 - s_t / q_max)
        if v_at_solar < c_u:
            p_star = 0.0
            q_d = s_t
            q_s = s_t
            q_b = 0.0        # no excess to charge into battery
            q_u = 0.0
            return p_star, q_d, q_s, q_b, q_u

        # 3) Otherwise utility steps in at p=c_u
        #    Find consumers' demand at that price
        q_d_cu = q_max * (1 - c_u / v_max)
        total_capacity = s_t + q_u_max

        # 3a) If demand > total capacity, clear at choke price > c_u
        if q_d_cu > total_capacity:
            q_d = total_capacity
            p_star = c_u #price of the last marginal unit sold
        else:
            # clear at utility price
            q_d = q_d_cu
            p_star = c_u

        # 4) Allocate dispatch:
        #    solar first (always price=0), then utility (battery can't charge at p>0)
        remaining = q_d
        q_s = min(s_t, remaining)
        remaining -= q_s

        q_b = 0.0     # battery bid only fills at p=0
        
        q_u = min(q_u_max, max(0.0, remaining))  # enforce non-negativity

        return p_star, q_d, q_s, q_b, q_u
    
    # def compute_optimal_strategy(self): 
    #     """ 
    #     Computes the Optimal Policy of trades for the 
    #     battery based on the perfect forecast information 
    #     about solar, aggregate demand, utility cost. 
    #     """

    #     print(f"[{self.agent_type}][compute_optimal_strategy] Starting computation of optimal strategy.")

    #     C_grid = np.linspace(0, self.C_max, self.C_max + 1)      # SOC: 0, 1, ..., 10
    #     q_b_grid = np.linspace(-self.C_max, self.C_max, 2*self.C_max + 1)    # Battery dispatch: -1, 0, 1
    #     # Error tolerance: half your grid spacing
    #     atol = (C_grid[1] - C_grid[0]) / 2   # e.g., 0.25 if grid step is 0.5  
    #     atol = 0
 

    #     print(f"[{self.agent_type}][compute_optimal_strategy] C_grid: {C_grid}")
    #     print(f"[{self.agent_type}][compute_optimal_strategy] q_b_grid: {q_b_grid}")
    #     print(f"[{self.agent_type}][compute_optimal_strategy] atol: {atol}")

    #     V = np.zeros((len(C_grid), self.T+1))
    #     P = np.zeros_like(V)
    #     QS = np.zeros_like(V)
    #     QB = np.zeros_like(V)
    #     QU = np.zeros_like(V)

    #     # Terminal constraint (battery returns to initial state)
    #     V[:, self.T] = -np.inf
    #     i_init = np.searchsorted(C_grid, self.C_init)
    #     V[i_init, self.T] = 0

    #     print(f"[{self.agent_type}][compute_optimal_strategy] Terminal constraint set at index {i_init} for C_init={self.C_init}")

    #     for t in reversed(range(self.T)):
    #         solar = self.s_t[t]
    #         #print(f"[{self.agent_type}][compute_optimal_strategy] Time step t={t}, solar={solar}")
    #         for i, C in enumerate(C_grid):
    #             best = -np.inf
    #             bp = bqs = bqb = bqu = 0.0

    #             for q_b_val in q_b_grid:
    #                 C_next = C - q_b_val
                    
    #                 # Ensure C_next is exactly at C_init when at final step (t = T-1)
    #                 if t == self.T - 1 and not np.isclose(C_next, self.C_init, atol=atol):
    #                     continue
                    
    #                 # ensure state is within bounds for all other time steps
    #                 if not (0 <= C_next <= self.C_max):
    #                     continue

    #                 if q_b_val < 0:
    #                     p_star, q_d, q_s, q_b_ex, q_u = self.equilibrium_with_battery_bid(
    #                         solar, -q_b_val, self.v_max, self.q_max, self.c_u, self.q_u_max
    #                     )
    #                     q_b = -q_b_ex
    #                     #print(f"[{self.agent_type}][compute_optimal_strategy][t={t}][C={C}] BID: q_b_val={q_b_val}, p_star={p_star}, q_b={q_b}")
    #                 else:
    #                     p_star, q_d, q_s, q_b_ex, q_u = self.equilibrium_with_battery_offer(
    #                         solar, q_b_val, self.v_max, self.q_max, self.c_u, self.q_u_max
    #                     )
    #                     q_b = q_b_ex
    #                     #print(f"[{self.agent_type}][compute_optimal_strategy][t={t}][C={C}] OFFER: q_b_val={q_b_val}, p_star={p_star}, q_b={q_b}")

    #                 inst = p_star * q_b_ex

    #                 idx = np.searchsorted(C_grid, C_next)
    #                 idx = min(max(idx, 0), len(C_grid)-1)

    #                 val = inst + self.beta * V[idx, t+1]

    #                 #print(f"[{self.agent_type}][compute_optimal_strategy][t={t}][C={C}] q_b_val={q_b_val}, C_next={C_next}, inst={inst}, val={val}, best={best}")

    #                 if val > best:
    #                     best = val
    #                     bp, bqs, bqb, bqu = p_star, q_s, q_b, q_u

    #             V[i, t] = best
    #             P[i, t], QS[i, t], QB[i, t], QU[i, t] = bp, bqs, bqb, bqu
    #             #print(f"[{self.agent_type}][compute_optimal_strategy][t={t}][C={C}] Best: V={best}, P={bp}, QS={bqs}, QB={bqb}, QU={bqu}")
        
    #     # Assume: C_grid, QB, C_init, T, P are already defined as above.
    #     C_path = [self.C_init]
    #     Q_b_path = []
    #     Q_u_path = []
    #     Q_s_path = []
    #     Q_d_path = []
    #     P_path = []

    #     current_C = self.C_init
    #     print(f"[{self.agent_type}][compute_optimal_strategy] Tracing optimal path...")
    #     for t in range(self.T):
    #         idx = np.abs(C_grid - current_C).argmin()  # Snap to closest grid point
    #         q_b = QB[idx, t]
    #         q_s = QS[idx, t]
    #         q_u = QU[idx, t]
    #         price = P[idx, t]
    #         q_d = self.q_max * (1 - price / self.v_max)

    #         Q_b_path.append(q_b)
    #         Q_u_path.append(q_u)
    #         Q_s_path.append(q_s)
    #         Q_d_path.append(q_d)
    #         P_path.append(price)    

    #         #print(f"[{self.agent_type}][compute_optimal_strategy][t={t}] C={current_C}, q_b={q_b}, q_s={q_s}, q_u={q_u}, price={price}, q_d={q_d}")

    #         current_C = C_grid[idx] - q_b  # Move from snapped value, not from possibly off-grid current_C
    #         C_path.append(current_C)
        
    #     self.optimal_q_b = Q_b_path
    #     print(f"[{self.agent_type}][compute_optimal_strategy] Optimal q_b path: {Q_b_path}")
    #     print(f"[{self.agent_type}][compute_optimal_strategy] Optimal q_b path: {Q_b_path}")
    

            


            




