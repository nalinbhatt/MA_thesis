import numpy as np
import pandas as pd
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

# ABM imports
from model import Model  
from data_tracker import TradingDataTracker
from demand import ElasticDemand
from battery import InformedTraderBattery, OptimalPolicyBattery
from supply import Supply 
from amm import AMM
from model import Model
import os



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




def run_abm_simulation(args):
    # --- Unpack params ---
    (C_max, C_init, q_b_max, mean_pmax, std_pmax, days, T, sunrise, sunset,v_max, q_max, c_u, q_u_max,beta,trades_per_period, battery_type, reserve_x_init, reserve_y_init) = args
    

    # --- Create solar profile ---
    solar_gen_profile, _ = generate_nday_solar(
        sunrise=sunrise, sunset=sunset, mean_pmax=mean_pmax, std_pmax=std_pmax, days=days
    )
    s_t = np.array(solar_gen_profile)

    finalize_circular_period = True #ensures the last period is the same as the first  
    
    #we simulate the informed trader for an additional period rather than
    #ensuring a circular loop in the case of the vfi optimized battery
    #we do this because we cannot gurantee that the state of charge of the informed trader
    #would permit the trades in the first period unlike vfi. 
    if battery_type =='informed':
        T = T+1 
        finalize_circular_period = False 


    #---- Initialize agent input arr ----
    q_u_max_arr= np.ones(T)*q_u_max #max capacity that the utility can supply 
    q_max_arr = np.ones(T)*q_max
    v_max_arr = np.ones(T)*v_max


    # --- Initialize agents ---
    # (Update this section to match your agent creation signatures)
    demand_agent = ElasticDemand(v_max_arr, q_max_arr, "demand",False)
    utility_agent = Supply(c_u, q_u_max_arr, "utility", False)
    solar_agent = Supply(0, s_t, "solar", False)

    #---Initialize battery agent-----
    if battery_type == 'informed':
        battery_agent = InformedTraderBattery(
            C_init, C_max, "informed_trader_battery", s_t, v_max, q_max, c_u, q_u_max_arr, q_b_max, verbose=False
        )

    elif battery_type == 'optimal':
        battery_agent = OptimalPolicyBattery(
            v_max, q_max, c_u, q_u_max, C_init, C_init, C_max, s_t, beta, T, q_b_max, "vfi_optimized_battery"
        )
    else:
        raise ValueError(f"Unknown battery_type: {battery_type}")

    agent_list = [utility_agent, demand_agent, solar_agent, battery_agent]
    amm_agent = AMM()
    # (Set AMM reserves as desired)
    amm_agent.setup_pool(reserve_x_init, reserve_y_init)

    # --- Run simulation ---
    model = Model(T, agent_list, amm_agent, trades_per_period, s_t)
    model.simulate(finalize_circular_period)
    
    # --- Collect results ---
    # You can select what to aggregate (example below)
    summary = model.data_tracker.get_summary_statistics()
    period_df = model.data_tracker.get_period_summary_df()
    # Optionally, add parameter info as columns:
    for k, v in zip(['C_max', 'C_init', 'q_b_max', 'mean_pmax', 'std_pmax', 'days', 'T', 'sunrise', 'sunset','v_max', 'q_max', 'c_u', 'q_u_max','beta','trades_per_period', 'battery_type', 'reserve_x_init', 'reserve_y_init'], args):
        summary[k] = v
        period_df[k] = v

    return summary, period_df.to_dict('records')

# ==== Main parallel execution ====

if __name__ == "__main__":

    # --- Build parameter grid as needed ---
    C_max_grid = [5, 10, 15, 20]
    C_init_fracs = [0.2, 0.5, 0.8]             # 20%, 50%, 80%
    q_b_max_grid = [1, 2, 3, 4, 5]
    mean_pmax_grid = [5, 10, 15, 20]
    days = 7
    T = 24 * days
    trades_per_period = 20
    sunrise = 6
    sunset = 20
    v_max = 10
    q_max = 10
    c_u = 5
    q_u_max = q_max
    beta = 1

    #Ensures a starting AMM spot price of 5  (M_token/E_token)
    k = 100 
    reserve_x_init = np.sqrt(k/5)
    reserve_y_init = k/reserve_x_init

    battery_types = ['informed', 'optimal'] #informed trader battery and vfi optimized

    param_grid = []
    for C_max in C_max_grid:
        C_init_grid = [int(round(C_max * frac)) for frac in C_init_fracs]
        for C_init, q_b_max, mean_pmax, battery_type  in product(C_init_grid, q_b_max_grid, mean_pmax_grid, battery_types):
            std_pmax_grid = [0, mean_pmax / 2]    # For each P_max, try 0 and P_max/2
            for std_pmax in std_pmax_grid:
                param_grid.append((C_max, C_init, q_b_max, mean_pmax, std_pmax, days, T, sunrise, sunset,v_max, q_max, c_u, q_u_max,beta,trades_per_period, battery_type, reserve_x_init, reserve_y_init))

    # --- Set up CSV output ---
    summary_csv = "abm_sim_summary.csv"
    period_csv = "abm_period_summary.csv"

    for fname in [summary_csv, period_csv]:
        if os.path.exists(fname):
            os.remove(fname)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(run_abm_simulation, args) for args in param_grid]
        for future in tqdm(as_completed(futures), total=len(futures)):
            summary, period_rows = future.result()
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(summary_csv, mode='a', header=not os.path.exists(summary_csv), index=False)
            period_df = pd.DataFrame(period_rows)
            period_df.to_csv(period_csv, mode='a', header=not os.path.exists(period_csv), index=False)

    print("All ABM simulations complete. Results written to CSV.")
