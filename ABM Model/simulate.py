import numpy as np 
import random 
import pandas as pd
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
        power_gen_day, p_max = generate_daily_solar_profile(sunrise, sunset, mean_pmax, std_pmax, day)
        #print(f"P_max for the day: {p_max:.2f}")
        #print("Hourly irradiance values:")

        if day == 0: 
    #we want 25 values for day 1 
            irradiance_profile.extend(power_gen_day)
        else: #we want less than 25 values for day 2 
            irradiance_profile.extend(power_gen_day[1:])

        p_max_list.append(p_max)
    
    return irradiance_profile, p_max


if __name__ == "__main__": 

    # === Parameters ===
    days = 1
    T = 24*days + 1
    v_max = 10 #max value for the demand purchased
    q_max = 10 #max quantity that demand agent would purchase 
    c_u = 5 #utility per unit marginal cost 
    q_u_max = 10 #max procurement by the utility
    C_init = 2
    C_final = 2
    C_max = 10

    # === Simulated solar irradiance ===
    sunrise = 6 
    sunset = 20
    mean_pmax = 10
    std_pmax = 1 

    seed = 1
    np.random.seed(seed)#comeback to this

    solar_gen_profile, p_max_list = generate_nday_solar(sunrise, sunset, mean_pmax, std_pmax, days)

    #np.random.seed(0)
    s_t = np.array(solar_gen_profile)  
    #s_t = np.array([6,6,6])
    q_u_max = np.ones(T)*q_max #max capacity that the utility can supply 


    k = 100

    reserve_x = 4 #number of Energy Tokens in the pool 
    reserve_y = k/reserve_x #number of Money Tokens in the pool
    p = reserve_y/reserve_x

    summary_df = pd.DataFrame([{
                            "v_max": v_max,
                            "C_max": C_max,
                            "p_max": p_max_list,
                            "seed": seed,
                            "social_welfare": None
                        }])

    hourly_df = pd.DataFrame([{
                            "v_max": v_max,
                            "C_max": C_max,
                            "p_max": p_max_list,
                            "seed": seed,
                            "hour": 0, #starts at 0th hour 
                            "p": p,
                            "q_s": 0,
                            "q_b": 0,
                            "q_u": 0 ,
                            "C": C_init,
                            "q_d": 0,
                            "s_t": 0 
                        }])
    

    #Demand/Consumer/Load agent 
    demand_agent = ElasticDemand(v_max, q_max, "demand", True )
    utility_agent = Supply(c_u, q_u_max, "utility", True)
    solar_agent = Supply(0, s_t, "solar", True )
    battery_agent = InformedTraderBattery(C_init, C_max,"informed_trader_battery", s_t, v_max, q_max, c_u, q_u_max, verbose= True)


    #AMM 
    amm_agent = AMM()
    amm_agent.setup_pool(reserve_x, reserve_y)
    agent_list = [utility_agent, demand_agent, utility_agent, battery_agent]

    trades_per_period = 100

    #Model Agent 
    model = Model(T, agent_list, amm_agent, trades_per_period, summary_df, hourly_df, s_t)
    model.simulate()
        






