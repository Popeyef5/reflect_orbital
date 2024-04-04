from settings import *
from util_numpy import load_farm_data, LatLon_to_ECEF, ECEF_to_Corotating, Tilted_to_ECI, ECI_to_Corotating, get_farm_area_stats

import numpy as np

def orbital_efficiency(
  points_per_day=POINTS_PER_DAY,
  points_per_year=POINTS_PER_YEAR,
  orbit_phases=ORBIT_PHASES, 
  orbit_flavours=ORBIT_FLAVOURS, 
  dropout=DROPOUTS[0], 
  farm_cosine_losses=FARM_COSINE_LOSSES,
  farm_area_losses=FARM_AREA_LOSSES
):
  #######
  # Setup
  #######

  try:
    from tqdm import tqdm
  except ImportError:
    tqdm = lambda x: x
  
  dtype = np.float32
  
  available_mem = 2 * 1024**3 # 2GB

  np.random.seed(1337)

  ###########
  # Variables
  ###########
 
  R = EARTH_MEAN_RADIUS
  h = ORBIT_ALTITUDE 
  a = R + h
  w_s = 2 * np.pi / ORBIT_PERIOD # satellite angular velocity
  i = ORBIT_INCLINATION * np.pi  / 180 # in radians
  T_earth = EARTH_ROTATION_PERIOD
  sun_subtended = SUN_SUBTENDED_ANGLE * np.pi / 180 # in radians
  
  ###########
  # Farm data
  ###########
  
  farm_data = load_farm_data()

  dropout_mask = np.random.rand(len(farm_data.T)) > dropout
  farm_data = farm_data[:, dropout_mask] # keep each farm as a client with p = (1-DROPOUT)

  farm_types = np.unique(farm_data[3])
  farm_area, ground_utilization_rate = get_farm_area_stats(farm_data)
  
  PANEL_SPACING_LOSSES = 1-ground_utilization_rate
  
  FARM_NUMBER = len(farm_area)
  CHUNK_SIZE = int(available_mem / orbit_phases / orbit_flavours / FARM_NUMBER / points_per_year // np.array([], dtype=dtype).itemsize)

  #######
  # Start
  #######
  
  # Time variables
  t = np.linspace(start=0, stop=T_earth, num=points_per_day, endpoint=False, dtype=dtype) # (points_per_day,) seconds in a day
  orbit_phase = np.linspace(start=0, stop=ORBIT_PERIOD, num=orbit_phases, endpoint=False, dtype=dtype) # (orbit_phases,) seconds of offset between orbits
  tau = t[:, np.newaxis] - orbit_phase[np.newaxis, :] # (points_per_day, orbit_phases)
  orbit_flavour = np.linspace(start=0, stop=2*np.pi, num=orbit_flavours, endpoint=False, dtype=dtype) # (orbit_flavours,) angle of the orbit wrt the sun direction. 0 is dawn/dusk with ascending node at ~6AM local time
  season = np.linspace(start=0, stop=2*np.pi, num=points_per_year, endpoint=False, dtype=dtype) # (points_per_year,) one full year rotation
  
  tau = tau[:, :, np.newaxis] # prepare for broadcast
  orbit_flavour = orbit_flavour[np.newaxis, np.newaxis, :] # prepare for broadcast
  
  optimal_sellable_power = np.zeros((points_per_day, orbit_phases, orbit_flavours, points_per_year, len(farm_types)), dtype=dtype)
  optimal_farm_distribution = np.zeros((points_per_day, orbit_phases, orbit_flavours, points_per_year, len(farm_types)), dtype=dtype)
  
  # Sun coordinates in space
  sun_tilted_x =  np.sin(season) # (points_per_year,) sun geocentric tilted x
  sun_tilted_y = -np.cos(season) # (points_per_year,) sun geocentric tilted y
  sun_tilted_z = np.zeros_like(season) # (points_per_year,) sun geocentric tilted z
  sun_tilted = np.stack((sun_tilted_x, sun_tilted_y, sun_tilted_z), axis=0) # (3, points_per_year)
  sun_eci = Tilted_to_ECI(sun_tilted) # (3, points_per_year)
  sun_corotating = ECI_to_Corotating(sun_eci, season) # (3, points_per_year) sun cartesian

  for j in tqdm(range(int(np.ceil(points_per_day / CHUNK_SIZE)))):
    # Chunks
    tau_chunk = tau[j*CHUNK_SIZE:(j+1)*CHUNK_SIZE]
    t_chunk = t[j*CHUNK_SIZE:(j+1)*CHUNK_SIZE]

    # Farms coordinates in space
    farms_ecef = LatLon_to_ECEF(farm_data[:2]) # (3, FARM_NUMBER)
    farms_corotating = ECEF_to_Corotating(np.repeat(farms_ecef[:, np.newaxis, :], len(t_chunk), axis=1), t_chunk) # (3, CHUNK_SIZE, FARM_NUMBER) 
  
    # Satellite coordinates in space
    sat_corotating_x = a*(np.cos(w_s*tau_chunk)*np.cos(orbit_flavour)-np.sin(w_s*tau_chunk)*np.sin(orbit_flavour)*np.cos(i)) # (CHUNK_SIZE, orbit_phases, orbit_flavours) satellite cartesian x
    sat_corotating_y = a*(np.cos(w_s*tau_chunk)*np.sin(orbit_flavour)+np.sin(w_s*tau_chunk)*np.cos(orbit_flavour)*np.cos(i)) # (CHUNK_SIZE, orbit_phases, orbit_flavours) satellite cartesian y
    sat_corotating_z = np.repeat((a*np.sin(w_s*tau_chunk)*np.sin(i)), orbit_flavours, axis=2) # (CHUNK_SIZE, orbit_phases, orbit_flavours) satellite cartesian z
    sat_corotating = np.stack((sat_corotating_x, sat_corotating_y, sat_corotating_z), axis=0) # (3, CHUNK_SIZE, orbit_phases, orbit_flavours) satellite cartesian  
  
    # Farms to satellites
    farms_to_sat_corotating = sat_corotating[:, :, :, :, np.newaxis] - farms_corotating[:, :, np.newaxis, np.newaxis, :] # (3, CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) vectors from each farm to each satellite
    norm_farms_to_sat_corotating = np.sqrt(np.einsum('ijklm, ijklm->jklm', farms_to_sat_corotating, farms_to_sat_corotating)) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) distances from each farm to each satellite
    
    # Sat to farm transmission
    filter_fov = norm_farms_to_sat_corotating > np.sqrt((R+h)**2 - R**2) # which farms are out of sight for each point along the orbit?
  
    scalar = np.einsum('ijklm, ijkl->jklm', farms_to_sat_corotating, sat_corotating) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) # cos(farm_sat_angle) * ||sat_to_farm||
    unit_scalar = scalar / (a * norm_farms_to_sat_corotating) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) * ||sat_to_farm||
    theta = np.pi/2 - np.arccos(np.clip(unit_scalar, -1, 1)) # Angle from the farm's surface to the satellite
    sat_farm_transmission = 0.1283 + 0.7548*np.exp(-0.3866/np.maximum(np.sin(theta), 0.01)) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) transmission from each satellite to each farm due to longer atmospheric path
  
    beam_area = np.pi * (norm_farms_to_sat_corotating * np.tan(sun_subtended) * 0.5)**2 # 
    area_losses = np.minimum(farm_area[np.newaxis, np.newaxis, np.newaxis, np.newaxis :] / beam_area, 1) if farm_area_losses else 1 # Losses due to beam being bigger than farm area
    cos_losses = np.sin(theta) if farm_cosine_losses else 1 # first order approx. We could be a bit more gentle and include it before taking the min before, but doesn't seem to do much difference 
    sat_farm_transmission = sat_farm_transmission * area_losses * cos_losses
    sat_farm_transmission[filter_fov] = 0 # disregard farms out of reach
  
    # Sun to farm transmission
    farm_sun_cos2 = np.einsum('ijklm, in->jklmn', -farms_to_sat_corotating, sun_corotating) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) cos(2*theta) * ||sat_to_farm|| (orbit_points, orbit_phases, flavours, farms, season_points)
    farm_sun_cos2 = farm_sun_cos2 / norm_farms_to_sat_corotating[:, :, :, :, np.newaxis] # cos(2*theta)
    sun_farm_transmission = np.sqrt(0.5*(np.maximum(farm_sun_cos2, -1) + 1)) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) cos(theta)
  
    sat_sun_cos = np.einsum('ijkl, im -> jklm', sat_corotating, sun_corotating) # (CHUNK_SIZE, orbit_phases, orbit_flavours) cosine of the angle between the sun and the satellite
    threshold = (R+h)**2 - R**2
    filter_sun = (sat_sun_cos < 0) & (sat_sun_cos**2 > threshold) # the satellite is behind the earth wrt the sun
    sun_farm_transmission[np.repeat(filter_sun[:, :, :, np.newaxis, :], FARM_NUMBER, axis=3)] = 0
  
    # Full transmission
    sun_farm_transmission *= sat_farm_transmission[:, :, :, :, np.newaxis]
  
    for k, type in enumerate(farm_types):
      # Optimal transmission and farm selection
      farm_mask = farm_data[3] == type
      max_transmission = np.max(sun_farm_transmission[:, :, :, farm_mask, :], axis=3)
      optimal_farms = np.argmax(sun_farm_transmission[:, :, :, farm_mask, :], axis=3)
      transmission_condition = np.any(sun_farm_transmission[:, :, :, farm_mask, :] != 0, axis=3)
      optimal_farms = np.where(transmission_condition, optimal_farms, -1)
      optimal_sellable_power[j*CHUNK_SIZE:(j+1)*CHUNK_SIZE, :, :, :, k] = max_transmission
      optimal_farm_distribution[j*CHUNK_SIZE:(j+1)*CHUNK_SIZE, :, :, :, k] = optimal_farms

  optimal_sellable_power *= SPACE_SOLAR_kWpm2 # how much power reaches LEO from the sun per m^2		
  optimal_sellable_power *= SAT_SURFACE # how much power can go through the satellite based on its area
  optimal_sellable_power *= MYLAR_REFLECTIVITY # how much power the satelite can reflect based on its reflectivity
  optimal_sellable_power *= SUNNY_SKY_FREQ # what percentage of the time will the climate conditions be optimal for transfer
  optimal_sellable_power *= 1-PANEL_SPACING_LOSSES # what percentage of the light actually hits a solar panel
  optimal_sellable_power *= PV_CELL_EFFICIENCY # how much power is harnessed by solar panels
  optimal_sellable_power *= 1-SOLAR_INVERTER_LOSSES # how much power reaches the grid per satellite in kW

  return optimal_sellable_power, optimal_farm_distribution


if __name__ == "__main__":
  from safetensors.numpy import save_file
  
  power = None
  for dropout in DROPOUTS:
    tmp_power, tmp_farms = orbital_efficiency(dropout=dropout)
    if power is None:
      power = tmp_power[np.newaxis, :]
      farms = tmp_farms[np.newaxis, :]
    else:
      power = np.concatenate((power, tmp_power[np.newaxis, :]), axis=0)
      farms = np.concatenate((farms, tmp_farms[np.newaxis, :]), axis=0)
  save_file({"power": power, "farms": farms}, DEFAULT_RESULTS_DIR)

