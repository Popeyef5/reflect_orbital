from settings import *
from util import load_farm_data, LatLon_to_ECEF, ECEF_to_Corotating, Tilted_to_ECI, ECI_to_Corotating, get_farm_area_stats

import torch

def simulate(
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
  
  dtype = torch.float32
  
  if torch.cuda.is_available():
    device = 'cuda'
    free, _ = torch.cuda.mem_get_info()
    available_mem = free * 0.25
  else:
    device = 'cpu'
    available_mem = 4 * 1024**3 # 4GB
  available_mem = 2 * 1024**3 # 2GB TODO: fix
  
  torch.manual_seed(1337)

  ###########
  # Variables
  ###########

  R = EARTH_MEAN_RADIUS
  h = ORBIT_ALTITUDE 
  a = R + h
  w_s = 2 * torch.pi / ORBIT_PERIOD # satellite angular velocity
  i = torch.tensor([ORBIT_INCLINATION * torch.pi  / 180], dtype=dtype, device=device) # in radians
  T_earth = EARTH_ROTATION_PERIOD
  sun_subtended = torch.tensor([SUN_SUBTENDED_ANGLE * torch.pi / 180], dtype=dtype, device=device) # in radians
  
  ###########
  # Farm data
  ###########
  
  farm_data = load_farm_data(device=device, dtype=dtype)

  dropout_mask = torch.rand_like(farm_data[0]) > dropout
  farm_data = farm_data[:, dropout_mask] # keep each farm as a client with p = (1-DROPOUT)

  farm_types, _ = torch.unique(farm_data[3]).sort()
  farm_area, ground_utilization_rate = get_farm_area_stats(farm_data, device=device, dtype=dtype)
  
  PANEL_SPACING_LOSSES = 1-ground_utilization_rate
  
  FARM_NUMBER = len(farm_area)
  CHUNK_SIZE = int(available_mem / orbit_phases / orbit_flavours / FARM_NUMBER / points_per_year // torch.tensor([], dtype=dtype).element_size())
 
  #######
  # Start
  #######

  # Time tensors
  t = torch.linspace(start=0, end=T_earth, steps=points_per_day+1, dtype=dtype, device=device) # torch missing endpoint kwarg in linspace
  t = t[:-1] # (points_per_day,) seconds in a day
  orbit_phase = torch.linspace(start=0, end=ORBIT_PERIOD, steps=orbit_phases+1, dtype=dtype, device=device)
  orbit_phase = orbit_phase[:-1] # (orbit_phases,) seconds of offset between orbits
  tau = t[:, None] - orbit_phase[None, :] # (points_per_day, orbit_phases)
  orbit_flavour = torch.linspace(start=0, end=2*torch.pi, steps=orbit_flavours+1, dtype=dtype, device=device)
  orbit_flavour = orbit_flavour[:-1] # (orbit_flavours,) angle of the orbit wrt the sun direction. 0 is dawn/dusk with ascending node at ~6AM local time
  season = torch.linspace(start=0, end=2*torch.pi, steps=points_per_year+1, dtype=dtype, device=device)
  season = season[:-1] # (points_per_year,) one full year rotation
  
  tau = tau[:, :, None] # prepare for broadcast
  orbit_flavour = orbit_flavour[None, None, :] # prepare for broadcast
  
  optimal_transmission = torch.zeros((points_per_day, orbit_phases, orbit_flavours, points_per_year, len(farm_types)), dtype=dtype, device=device)
  optimal_farm_distribution = torch.zeros((points_per_day, orbit_phases, orbit_flavours, points_per_year, len(farm_types)), dtype=torch.int, device=device)
  cumulative_farm_allocation = torch.zeros((orbit_phases, orbit_flavours, FARM_NUMBER), dtype=dtype, device=device)
  
  # Sun coordinates in space
  sun_tilted_x =  torch.sin(season) # (points_per_year,) sun geocentric tilted x
  sun_tilted_y = -torch.cos(season) # (points_per_year,) sun geocentric tilted y
  sun_tilted_z = torch.zeros_like(season) # (points_per_year,) sun geocentric tilted z
  sun_tilted = torch.stack((sun_tilted_x, sun_tilted_y, sun_tilted_z), dim=0) # (3, points_per_year)
  sun_eci = Tilted_to_ECI(sun_tilted) # (3, points_per_year)
  sun_corotating = ECI_to_Corotating(sun_eci, season) # (3, points_per_year) sun cartesian

  tau_chunks = torch.split(tau, CHUNK_SIZE) 
  t_chunks = torch.split(t, CHUNK_SIZE)

  for j, (tau_chunk, t_chunk) in enumerate(tqdm(zip(tau_chunks, t_chunks), total=len(tau_chunks))): 
    # Farms coordinates in space
    farms_ecef = LatLon_to_ECEF(farm_data[:2]) # (3, FARM_NUMBER)
    farms_corotating = ECEF_to_Corotating(torch.repeat_interleave(farms_ecef[:, None, :], len(t_chunk), dim=1), t_chunk) # (3, CHUNK_SIZE, FARM_NUMBER) 
  
    # Satellite coordinates in space
    sat_corotating_x = a*(torch.cos(w_s*tau_chunk)*torch.cos(orbit_flavour)-torch.sin(w_s*tau_chunk)*torch.sin(orbit_flavour)*torch.cos(i)) # (CHUNK_SIZE, orbit_phases, orbit_flavours) satellite cartesian x
    sat_corotating_y = a*(torch.cos(w_s*tau_chunk)*torch.sin(orbit_flavour)+torch.sin(w_s*tau_chunk)*torch.cos(orbit_flavour)*torch.cos(i)) # (CHUNK_SIZE, orbit_phases, orbit_flavours) satellite cartesian y
    sat_corotating_z = torch.repeat_interleave((a*torch.sin(w_s*tau_chunk)*torch.sin(i)), orbit_flavours, dim=2) # (CHUNK_SIZE, orbit_phases, orbit_flavours) satellite cartesian z
    sat_corotating = torch.stack((sat_corotating_x, sat_corotating_y, sat_corotating_z), axis=0) # (3, CHUNK_SIZE, orbit_phases, orbit_flavours) satellite cartesian  
  
    # Farms to satellites
    farms_to_sat_corotating = sat_corotating[:, :, :, :, None] - farms_corotating[:, :, None, None, :] # (3, CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) vectors from each farm to each satellite
    norm_farms_to_sat_corotating = torch.sqrt(torch.einsum('ijklm, ijklm->jklm', farms_to_sat_corotating, farms_to_sat_corotating)) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) distances from each farm to each satellite
    
    # Sat to farm transmission
    filter_fov = norm_farms_to_sat_corotating > torch.sqrt(torch.tensor([(R+h)**2 - R**2], dtype=dtype, device=device)) # which farms are out of sight for each point along the orbit?
  
    scalar = torch.einsum('ijklm, ijkl->jklm', farms_to_sat_corotating, sat_corotating) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) # cos(farm_sat_angle) * ||sat_to_farm||
    unit_scalar = scalar / (a * norm_farms_to_sat_corotating) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) * ||sat_to_farm||
    theta = torch.pi/2 - torch.arccos(torch.clamp(unit_scalar, min=-1, max=1)) # Angle from the farm's surface to the satellite
    sat_farm_transmission = 0.1283 + 0.7548*torch.exp(-0.3866/torch.clamp(torch.sin(theta), min=0.01)) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER) transmission from each satellite to each farm due to longer atmospheric path
  
    beam_area = torch.pi * (norm_farms_to_sat_corotating * torch.tan(sun_subtended) * 0.5)**2 # 
    area_losses = torch.minimum(farm_area[None, None, None, :] / beam_area, torch.ones_like(beam_area)) if farm_area_losses else 1 # Losses due to beam being bigger than farm area
    cos_losses = torch.sin(theta) if farm_cosine_losses else 1 # first order approx. We could be a bit more gentle and include it before taking the min before, but doesn't seem to do much difference 
    sat_farm_transmission = sat_farm_transmission * area_losses * cos_losses
    sat_farm_transmission[filter_fov] = 0 # disregard farms out of reach
  
    # Sun to farm transmission
    farm_sun_cos2 = torch.einsum('ijklm, in -> jklmn', -farms_to_sat_corotating, sun_corotating) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER, points_per_year) cos(2*theta) * ||sat_to_farm|| (orbit_points, orbit_phases, flavours, farms, season_points)
    farm_sun_cos2 = farm_sun_cos2 / norm_farms_to_sat_corotating[:, :, :, :, None] # cos(2*theta)
    sun_farm_transmission = torch.sqrt(0.5*(torch.clamp(farm_sun_cos2, min=-1) + 1)) # (CHUNK_SIZE, orbit_phases, orbit_flavours, FARM_NUMBER, points_per_year) cos(theta)
  
    sat_sun_cos = torch.einsum('ijkl, im -> jklm', sat_corotating, sun_corotating) # (CHUNK_SIZE, orbit_phases, orbit_flavours, points_per_year) cosine of the angle between the sun and the satellite
    threshold = (R+h)**2 - R**2
    filter_sun = (sat_sun_cos < 0) & (sat_sun_cos**2 > threshold) # the satellite is behind the earth wrt the sun
    sun_farm_transmission[torch.repeat_interleave(filter_sun[:, :, :, None, :], FARM_NUMBER, dim=3)] = 0 # no transmission for blocked satellites
  
    # Full transmission
    sun_farm_transmission *= sat_farm_transmission[:, :, :, :, None]

    for k, type in enumerate(farm_types):
      # Optimal transmission and farm selection
      farm_mask = farm_data[3] <= type # consider farms with status at most 'type'
      max_transmission, optimal_farms = torch.max(sun_farm_transmission[:, :, :, farm_mask, :], dim=3) # TODO fix optimal farms: index is wrong as it relates to the masked array instead of the complete one (besides dropout)
      transmission_condition = torch.any(sun_farm_transmission[:, :, :, farm_mask, :] != 0, dim=3)
      optimal_farms = torch.where(transmission_condition, optimal_farms, -1)
      optimal_transmission[j*CHUNK_SIZE:(j+1)*CHUNK_SIZE, :, :, :, k] = max_transmission
      optimal_farm_distribution[j*CHUNK_SIZE:(j+1)*CHUNK_SIZE, :, :, :, k] = optimal_farms.to(dtype=torch.int)

      chunk_farm_transmission = torch.trapz(sun_farm_transmission, dx=EARTH_ROTATION_PERIOD/points_per_day/60/60, axis=0) # how much energy can be sold in one day (dropouts, orbit_phases, orbit_flavours, points_per_year, farm_type)
      annual_farm_transmission = torch.trapz(chunk_farm_transmission, dx=365/points_per_year, axis=3) # (dropouts, orbit_phases, orbit_flavours, farm_type)
      cumulative_farm_allocation += annual_farm_transmission
  
  return optimal_transmission.cpu(), optimal_farm_distribution.cpu(), cumulative_farm_allocation.cpu()


if __name__ == "__main__":
  from safetensors.torch import save_file

  cumulative = None 
  transmission = None
  for dropout in DROPOUTS:
    tmp_transmission, tmp_farms, tmp_cumulative = simulate(dropout=dropout)
    if transmission is None:
      transmission = tmp_transmission[None, :]
      farms = tmp_farms[None, :]
    else:
      transmission = torch.cat((transmission, tmp_transmission[None, :]), dim=0)
      farms = torch.cat((farms, tmp_farms[None, :]), dim=0)

    if cumulative is None:
      cumulative = tmp_cumulative
      
  save_file({"transmission": transmission, "farms": farms, "cumulative": cumulative}, DEFAULT_RESULTS_DIR)
