import numpy as np
import pandas as pd
from itertools import product
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing


def generate_daily_solar_profile(sunrise=6, sunset=20, mean_pmax=15, std_pmax=1, seed=None):
    """
    Generate a 24-hour solar irradiance profile using a cosine model for daylight hours.
    
    Parameters:
    - sunrise: Hour of sunrise (e.g., 6)
    - sunset: Hour of sunset (e.g., 18)
    - mean_pmax: Mean of P_max (e.g., 1000 W/m²)
    - std_pmax: Standard deviation of P_max (e.g., 100 W/m²)
    - seed: Random seed for reproducibility
    
    Returns:
    - irradiance: numpy array of length 24, representing hourly solar irradiance
    - p_max: the actual peak irradiance drawn for the day
    """
    if seed is not None:
        np.random.seed(seed)
    
    hours = np.arange(24 + 1)  # 0 to 25 
    irradiance = np.zeros_like(hours, dtype=float)
    
    t_mid = (sunrise + sunset) / 2
    daylight_duration = sunset - sunrise

    # Draw P_max from normal distribution
    p_max = np.random.normal(loc=mean_pmax, scale=std_pmax)

    # Apply cosine model during daylight hours
    for h in range(24 + 1):
        
        if sunrise <= h <= sunset:
            val = p_max * np.cos(np.pi * (h - t_mid) / daylight_duration)
            irradiance[h] = max(0, val)  # clip negative values to zero

        #ensuring that the 25th hour is the same as the 0th hour
        if (h == 25): val[h] == val[0] 

    return irradiance, p_max

# Example usage
# irradiance_profile, p_max = generate_daily_solar_profile(seed=0)
# print(f"P_max for the day: {p_max:.2f}")
# print("Hourly irradiance values:")
# print(irradiance_profile)

def generate_nday_solar(sunrise=6, sunset=20, mean_pmax=14, std_pmax=1, days= 1):
    """Generate nday solar production"""

    irradiance_profile = []
    p_max_list = []
    for day in range(days): 
        #print(f"Day = {day}")
        p_max = -1 
        while p_max <=0:  #ensures all days have a positive p_max
            power_gen_day, p_max = generate_daily_solar_profile(sunrise, sunset, mean_pmax, std_pmax, day)
        #print(f"P_max for the day: {p_max:.2f}")
        #print("Hourly irradiance values:")


        if day == 0: #we want 25 values for day 1 
            irradiance_profile.extend(power_gen_day)
        else: #we want less than 25 values for day 2 
            irradiance_profile.extend(power_gen_day[1:])

        p_max_list.append(p_max)
    
    return irradiance_profile, p_max_list


def equilibrium_with_battery_offer(
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
        p_star = c_u #last marginal unit cleared in the auction 
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

    # remaining should be ~0
    return p_star, q_d, q_s, q_b, q_u


def equilibrium_with_battery_bid(
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
        p_star = c_u #last marginal unit price
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


# ---- 1. Simulation function for one parameter combination ----
def run_simulation(args):
    (C_max, C_init, q_b_max, mean_pmax, std_pmax, days, T, sunrise, sunset,v_max,q_max, c_u, q_u_max,beta) = args

    period_rows = []

    # Setup grids
    C_grid = np.arange(0, C_max + 1) #integer 
    q_b_grid = np.arange(-q_b_max, q_b_max + 1) #integer 

    # Solar generation
    solar_gen_profile, _ = generate_nday_solar(
        sunrise, sunset, mean_pmax, std_pmax, days=days
    )
    s_t = np.array(solar_gen_profile[:T])

    # Initialize value function arrays
    V = np.zeros((len(C_grid), T+1))
    P = np.zeros_like(V)
    QS = np.zeros_like(V)
    QB = np.zeros_like(V)
    QU = np.zeros_like(V)

    # Terminal constraint: return to initial SOC
    V[:, T] = -np.inf
    i_init = np.searchsorted(C_grid, C_init)
    V[i_init, T] = 0

    # Value function iteration loop (same as before)
    for t in reversed(range(T)):
        solar = s_t[t]
        for i, C in enumerate(C_grid):
            best = -np.inf
            bp = bqs = bqb = bqu = 0.0
            for q_b_val in q_b_grid:
                C_next = C - q_b_val
                
                if t == T - 1 and not np.isclose(C_next, C_init, atol=0):
                    continue
                if not (0 <= C_next <= C_max):
                    continue
                
                if q_b_val < 0:
                    p_star, q_d, q_s, q_b_ex, q_u = equilibrium_with_battery_bid(
                        solar, -q_b_val, v_max, q_max, c_u, q_u_max
                    )
                    q_b = -q_b_ex
                    
                    # Skip if partial fill, partial fills move the VFI away from the grid 
                    if not np.isclose(q_b_ex, -q_b_val, atol=1e-6):
                        continue

                else:
                    p_star, q_d, q_s, q_b_ex, q_u = equilibrium_with_battery_offer(
                        solar, q_b_val, v_max, q_max, c_u, q_u_max
                    )
                    q_b = q_b_ex

                    # Skip if partial fill, partial fills move the VFI away from the grid 
                    if not np.isclose(q_b_ex, q_b_val, atol=1e-6):
                        continue
                    
                inst = p_star * q_b
                idx = np.searchsorted(C_grid, C_next)
                idx = min(max(idx, 0), len(C_grid)-1)
                val = inst + beta*V[idx, t+1]
                if val > best:
                    best = val
                    bp, bqs, bqb, bqu = p_star, q_s, q_b, q_u
            V[i, t] = best
            P[i, t], QS[i, t], QB[i, t], QU[i, t] = bp, bqs, bqb, bqu

    # Forward pass for optimal policy
    i = i_init
    C = C_init
    prices = []
    socs = []
    q_s_list = []
    q_b_list = []
    q_u_list = []
    q_d_list = []
    surplus_battery = []
    surplus_utility = []
    surplus_solar = []
    surplus_demand = []
    surplus_total = []

    for t in range(T):
        socs.append(C)
        prices.append(P[i, t])
        q_s_list.append(QS[i, t])
        q_b_list.append(QB[i, t])
        q_u_list.append(QU[i, t])

        #Determining Demand 
        p = P[i,t] #current price
        q_d = q_max*(1-p/v_max)
        q_d_list.append(q_d)

        # Surplus calculations 
        surplus_battery.append(P[i, t] * QB[i, t])  
        surplus_solar.append(P[i, t] * QS[i, t])   
        surplus_utility.append((P[i, t] - c_u) * QU[i, t])  
        surplus_demand.append(v_max * q_d - (v_max / (2 * q_max)) * q_d**2 - p * q_d)                  
        surplus_total.append(surplus_battery[-1] + surplus_utility[-1] + surplus_demand[-1] + surplus_solar[-1])

        # #Advance to next state 
        # C = C - QB[i, t]
        # i = np.searchsorted(C_grid, C)
        # i = min(max(i, 0), len(C_grid)-1)

        #valid 
        q_b_t = QB[i, t]
        C_next = C - q_b_t

        # Clip to [0, C_max] range to ensure SOC validity
        C_next = max(0, min(C_max, C_next))

        # Snap to nearest valid grid point
        i = np.abs(C_grid - C_next).argmin()
        C = C_grid[i]  # Update to snapped value


    #Ensure a closed loop, the first period T=0 matches the final period T+1
    socs.append(C)
    prices.append(prices[0])
    q_s_list.append(q_s_list[0])
    q_b_list.append(q_b_list[0])
    q_u_list.append(q_u_list[0])
    q_d_list.append(q_d_list[0])
    
    # Surplus calculations 
    surplus_battery.append(surplus_battery[0])  
    surplus_solar.append(surplus_solar[0])   
    surplus_utility.append(surplus_utility[0])  
    surplus_demand.append(surplus_demand[0])                  
    surplus_total.append(surplus_battery[0] + surplus_utility[0] + surplus_demand[0] + surplus_solar[0])


    summary_row = {
        'v_max': v_max, 
        'q_max': q_max, 
        'c_u': c_u, 
        'q_u_max': q_u_max, 
        'beta':beta, 
        'C_max': C_max,
        'C_init': C_init,
        'q_b_max': q_b_max,
        'mean_pmax': mean_pmax,
        'std_pmax': std_pmax,
        'prices': prices,
        'socs': socs,
        'q_s': q_s_list,
        'q_b': q_b_list,
        'q_u': q_u_list,
        'q_d': q_d_list,
        's_t': solar_gen_profile,
        'surplus_battery_ts': surplus_battery,
        'surplus_utility_ts': surplus_utility,
        'surplus_demand_ts': surplus_demand,
        'surplus_solar_ts': surplus_solar,
        'surplus_total_ts': surplus_total,
        'total_surplus_battery': sum(surplus_battery),
        'total_surplus_solar': sum(surplus_solar),
        'total_surplus_utility': sum(surplus_utility),
        'total_surplus_demand': sum(surplus_demand),
        'total_surplus_all': sum(surplus_total)
    }

   

    for t in range(T + 1): # T+1st period loops around and is the same as first period
        period_rows.append({
            'v_max': v_max, 
            'q_max': q_max, 
            'c_u': c_u, 
            'q_u_max': q_u_max, 
            'beta': beta, 
            'C_max': C_max,
            'C_init': C_init,
            'q_b_max': q_b_max,
            'mean_pmax': mean_pmax,
            'std_pmax': std_pmax,
            'period': t,
            'price': prices[t],
            'soc': socs[t],
            'q_s': q_s_list[t],
            'q_b': q_b_list[t],
            'q_u': q_u_list[t],
            'q_d': q_d_list[t],
            's_t': solar_gen_profile[t],
            'surplus_battery': surplus_battery[t],
            'surplus_utility': surplus_utility[t],
            'surplus_demand': surplus_demand[t],
            'surplus_solar': surplus_solar[t],
            'surplus_total': surplus_total[t]
        })

    return summary_row, period_rows


# ---- 3. Parallel execution and CSV writing ----
def write_csv_row(df, file):
    df.to_csv(file, mode='a', index=False, header=False)


if __name__ == "__main__":
    
    C_max_grid = [5, 10, 15, 20]
    C_init_fracs = [0.2, 0.5, 0.8]             # 20%, 50%, 80%
    q_b_max_grid = [1, 2, 3, 4, 5]
    mean_pmax_grid = [5, 10, 15, 20]
    days = 7
    T = 24 * days
    sunrise = 6
    sunset = 20
    v_max = 10
    q_max = 10
    c_u = 5
    q_u_max = q_max
    beta = 1

    param_grid = []
    for C_max in C_max_grid:
        # For each C_max, compute all valid integer initial charges
        C_init_grid = [int(round(C_max * frac)) for frac in C_init_fracs]
        for C_init, q_b_max, mean_pmax in product(C_init_grid, q_b_max_grid, mean_pmax_grid):
            std_pmax_grid = [0, mean_pmax / 2]    # For each P_max, try 0 and P_max/2
 
            for std_pmax in std_pmax_grid:
                param_grid.append(
                    (C_max, C_init, q_b_max, mean_pmax, std_pmax, days, T, sunrise, sunset,v_max, q_max, c_u, q_u_max,beta)
                )
    #print(f"Total simulations: {len(param_grid)}")
    # You can inspect the first few rows if desired:
    #for p in param_grid[:5]: print(p) 


    # Output file paths
    summary_csv = "simulation_summary.csv"
    period_csv = "period_summary.csv"

    # Remove old files if present (optional)
    for fname in [summary_csv, period_csv]:
        if os.path.exists(fname):
            os.remove(fname)
    
    # 1. List of columns for each DataFrame
    summary_columns = [
        'v_max','q_max','c_u', 'q_u_max', 'beta','C_max','C_init','q_b_max','mean_pmax','std_pmax',
        'prices','socs','q_s','q_b','q_u','q_d', 's_t',
        'surplus_battery_ts','surplus_utility_ts','surplus_demand_ts','surplus_solar_ts', 'surplus_total_ts',
        'total_surplus_battery','total_surplus_solar', 'total_surplus_utility','total_surplus_demand','total_surplus_all'
    ]

    period_columns = [
        'v_max','q_max','c_u', 'q_u_max', 'beta','C_max','C_init','q_b_max','mean_pmax','std_pmax','period',
        'price','soc','q_s','q_b','q_u','q_d','s_t',
        'surplus_battery','surplus_utility','surplus_demand', 'surplus_solar','surplus_total'
    ]

    # 2. Write empty DataFrames with headers
    pd.DataFrame(columns=summary_columns).to_csv(summary_csv, index=False)
    pd.DataFrame(columns=period_columns).to_csv(period_csv, index=False)



    # Build param_grid, etc...
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(run_simulation, args) for args in param_grid]

            # ...CSV write loop...
        for future in tqdm(as_completed(futures), total=len(futures)):
            summary_row, period_rows = future.result()

            # Write summary row
            summary_df = pd.DataFrame([summary_row])
            write_csv_row(summary_df, summary_csv)

            # Write period rows
            period_df = pd.DataFrame(period_rows)
            write_csv_row(period_df, period_csv)


    print("All simulations complete. Results written to CSV.")