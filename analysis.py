import matplotlib.pyplot as plt
from settings import *
from util import get_farm_area_stats
import numpy_financial as npf
from safetensors.torch import load_file
import numpy as np
import torch


def analyze(points_per_day=POINTS_PER_DAY, points_per_year=POINTS_PER_YEAR, orbit_phases=ORBIT_PHASES, orbit_flavours=ORBIT_FLAVOURS):
  farm_area, ground_utilization_rate = get_farm_area_stats()
  PANEL_SPACING_LOSSES = 1-ground_utilization_rate

  t = torch.linspace(start=0, end=EARTH_ROTATION_PERIOD, steps=points_per_day+1, dtype=torch.float32) # torch missing endpoint kwarg in linspace
  t = t[:-1] # (points_per_day,) seconds in a day
  
  solar_power  = SPACE_SOLAR_kWpm2 # how much power reaches LEO from the sun per m^2		
  solar_power *= SAT_SURFACE # how much power can go through the satellite based on its area
  solar_power *= MYLAR_REFLECTIVITY # how much power the satelite can reflect based on its reflectivity
  solar_power *= SUNNY_SKY_FREQ # what percentage of the time will the climate conditions be optimal for transfer
  solar_power *= 1-PANEL_SPACING_LOSSES # what percentage of the light actually hits a solar panel
  solar_power *= PV_CELL_EFFICIENCY # how much power is harnessed by solar panels
  solar_power *= 1-SOLAR_INVERTER_LOSSES # how much power reaches the grid per satellite in kW
  
  tensors = load_file(DEFAULT_RESULTS_DIR)
  optimal_sellable_energy = tensors["transmission"] * solar_power
  
  daily_sellable_energy = torch.trapz(optimal_sellable_energy, dx=EARTH_ROTATION_PERIOD/points_per_day/60/60, axis=1) # how much energy can be sold in one day (dropouts, orbit_phases, orbit_flavours, points_per_year, farm_type)
  annual_sellable_energy = torch.trapz(daily_sellable_energy, dx=365/points_per_year, axis=3) # (dropouts, orbit_phases, orbit_flavours, farm_type)
  
  annual_revenue = annual_sellable_energy * ENERGY_PRICE * (1-SOLAR_FARM_REV_SHARE)

  total_sat_cost = SAT_COST + SAT_WEIGHT * KG_COST_TO_ORBIT

  lifetime_energy_output = annual_sellable_energy * SAT_LIFE
  LCOE = total_sat_cost / lifetime_energy_output

  apr_equivalent = np.zeros_like(annual_revenue.numpy())
  for index in np.ndindex(apr_equivalent.shape): # unfortunately vectorized operation yielded all NaNs for some reason
    apr_equivalent[index] = npf.rate(
      nper=12*SAT_LIFE,
      pmt=-np.maximum(annual_revenue[index]/12 - SAT_OPEX, 1),
      pv=total_sat_cost,
      fv=0,
    )
  apr_equivalent *= 12*100

  customer_percentage = np.round((1 - np.array(DROPOUTS))*100)
  
  ##########
  # Plotting
  ##########
   
  cmap = plt.get_cmap('coolwarm_r')   

  # ======================
  # Interest rate analisis
  # ======================
  
  plt.figure("Interest rate vs Customer percentage")
  for i in range(4):
    plt.plot(customer_percentage, np.amax(apr_equivalent, axis=(1, 2))[:, i])
  plt.xlabel("Customer %")
  plt.ylabel("Maximum tolerable fixed interest rate")
  plt.legend(["operating", "construction", "pre-construction", "announced"])
  plt.title("Interest rate vs customer %")
  plt.savefig("media/interest_rate.jpg")
  
  
  plt.figure("Optimal orbits interest rate heatmap")
  vmax = np.max(np.abs(np.amax(apr_equivalent, axis=(1, 2))))  # Find the maximum absolute value in the data
  vmin = -vmax
  cax = plt.imshow(np.amax(apr_equivalent, axis=(1, 2)), cmap=cmap, vmax=vmax, vmin=vmin)
  plt.colorbar(cax)
  plt.xlabel("Max farm type")
  plt.xticks(ticks=np.arange(len(INCLUDE_FARM_STATUS)), labels=INCLUDE_FARM_STATUS, rotation=20, ha='right')
  plt.ylabel("Customer %")
  plt.yticks(ticks=np.arange(len(DROPOUTS)), labels=customer_percentage)
  plt.tight_layout()
  plt.savefig("media/interest_rate_heatmap.jpg")

  # ==================================
  # Energy transmission along an orbit
  # ==================================
  
  plt.figure("Transmission along optimal orbit")
  for i in range(4):
    dropout_idx, phase_idx, flavour_idx = np.unravel_index(np.argmax(apr_equivalent[:, :, :, i]), apr_equivalent[:, :, :, i].shape)
    plt.plot(t / 60 / 60, optimal_sellable_energy[dropout_idx, :, phase_idx, flavour_idx, 0, i])
  plt.legend(["operating", "construction", "pre-construction", "announced"])
  plt.title("Energy transmission along optimal orbit")
  plt.savefig("media/daily_energy_transmission.jpg")

  # =============================
  # Orbit performance variability
  # =============================
  
  best_orbit_by_farm_type, _ = torch.max(annual_revenue[dropout_idx, :, :, :].reshape((orbit_phases*orbit_flavours, -1)), axis=0)  # Find the maximum absolute value in the data

  plt.figure("Orbit variability")
  vmax, vmin = 1, 0
  cax = plt.imshow((annual_revenue[dropout_idx, :, :, :].reshape((orbit_phases*orbit_flavours, -1)) / best_orbit_by_farm_type[None, :]).T, cmap=cmap, vmax=vmax, vmin=vmin, aspect='auto')
  plt.colorbar(cax)
  plt.xlabel("Orbit phase")
  plt.ylabel("Max farm type")
  plt.yticks(ticks=np.arange(len(INCLUDE_FARM_STATUS)), labels=INCLUDE_FARM_STATUS)
  plt.tight_layout()
  plt.title("Orbit revenue normalized row-wise")
  plt.savefig("media/orbit_revenue_variability.jpg")
  
  # ==============================
  # Farm cumulative power received
  # ==============================
 
  cumulative_farm_distribution = tensors["cumulative"] # how much transmission occurred over a year in each farm
  cumulative_farm_distribution *= solar_power
  cumulative_farm_distribution *= ENERGY_PRICE * SOLAR_FARM_REV_SHARE # how much money can each farm potentially make per year

  max_allocation, flat_idx = cumulative_farm_distribution.view(-1).max(0)
  phase_idx, flavour_idx, _ = torch.unravel_index(flat_idx, cumulative_farm_distribution.shape)
  filtered_allocation = cumulative_farm_distribution[phase_idx, flavour_idx, :]
  sorted_allocation, _ = filtered_allocation.sort(descending=True)

  sat_number = torch.linspace(1, 100000, 50)
  farm_sat_allocation = sat_number[None, :] * filtered_allocation[:, None] * ENERGY_PRICE
  farm_sat_allocation_count = (farm_sat_allocation > MIN_FARM_ALLOCATION).sum(dim=0)

  plt.figure("Farm potential revenue distribution")
  plt.plot(sorted_allocation)
  plt.xlabel("Farm number")
  plt.ylabel("Yearly farm extra revenue [$]")
  plt.title("Farm potential revenue distribution")
  plt.savefig("media/farm_revenue_distribution.jpg")

  plt.figure("Sat number analysis")
  ax = plt.axes()
  ax.plot(farm_sat_allocation_count, sat_number)
  secax = ax.secondary_yaxis('right', functions=(lambda x: x * (total_sat_cost/1e6), lambda x: x / (total_sat_cost/1e6)))
  secax.set_ylabel("CAPEX [$M]")
  plt.xlabel("Number of farms")
  plt.ylabel("Number of satellites")
  plt.title("Amount of farms that can receive the minimum annual revenue required")
  plt.savefig("media/sat_number_analisis.jpg")

  # ==========
  # Show plots
  # ==========

  plt.show()

if __name__ == "__main__":
  analyze()
 

