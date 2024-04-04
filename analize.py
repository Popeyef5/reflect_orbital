from safetensors.numpy import load_file
import matplotlib.pyplot as plt
from settings import *
import numpy as np

def analize():  
  tensors = load_file(DEFAULT_RESULTS_DIR)
  optimal_sellable_power = tensors["power"]

  annual_sellable_energy = np.trapz(optimal_sellable_power, dx=EARTH_ROTATION_PERIOD/POINTS_PER_DAY/60/60, axis=0) # how much energy can be sold in one day (orbit_phases, flavours, season_points)
  annual_sellable_energy = np.trapz(annual_sellable_energy, dx=365/POINTS_PER_YEAR, axis=2) # (orbit_phases, flavours)

  annual_revenue = annual_sellable_energy * ENERGY_PRICE

  total_sat_cost = SAT_COST + SAT_WEIGHT * KG_COST_TO_ORBIT

  lifetime_energy_output = annual_sellable_energy * SAT_LIFE
  LCOE = total_sat_cost / lifetime_energy_output

  annual_returns = annual_revenue / total_sat_cost
  print(annual_returns[:, 0, :])
  

if __name__ == "__main__":
  analize()

  # Post sim variables: launch_cost, operational_status, 
  # Post sim analisis: annual_returns, LCOE, theta?, variability on phase and flavour, best orbit transmission