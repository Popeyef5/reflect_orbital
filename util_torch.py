from settings import *

import torch
import numpy as np


def cart_to_sph(p):
  """Cartesian to spherical coordinates"""
  ret = torch.zeros_like(p)
  xy = p[0]**2 + p[1]**2
  ret[0] = torch.sqrt(xy + p[2]**2)
  ret[1] = torch.arctan2(torch.sqrt(xy), p[2])
  ret[2] = torch.arctan2(p[1], p[0])
  return ret


def sph_to_cart(p):
  """Spherical to cartesian coordinates"""
  ret = torch.zeros_like(p)
  r, theta, phi = p[0], p[1], p[2]
  ret[0] = r * torch.sin(theta) * torch.cos(phi)
  ret[1] = r * torch.sin(theta) * torch.sin(phi)
  ret[2] = r * torch.cos(theta)
  return ret


def ECI_to_ECEF(p, t):
  """Earth centered inertial to Earth centered, Earth fixed coordinates"""
  angle = 2*torch.pi / EARTH_ROTATION_PERIOD * t # (orbit_points,)
  cos_vals = torch.cos(angle)
  sin_vals = torch.sin(angle)
  rot = torch.stack([
    torch.stack([ cos_vals               , sin_vals               , torch.zeros_like(angle)], dim=0),
    torch.stack([-sin_vals               , cos_vals               , torch.zeros_like(angle)], dim=0),
    torch.stack([ torch.zeros_like(angle), torch.zeros_like(angle), torch.ones_like(angle) ], dim=0)
  ], dim=0) # (3, 3, angle_points)
  return torch.einsum('ijk, jk->ik', rot, p) # (3, 3, orbit_points).(3, orbit_points) -> (3, orbit_points)


def ECEF_to_LatLon(p):
  """Earth centered, Earth fixed coordinates to Latitude and Longitude"""
  ret = cart_to_sph(p)
  lat = torch.pi/2 - ret[1]
  lon = ret[2] % (2*torch.pi) - torch.pi
  return torch.tensor([lat, lon], dtype=p.dtype)


def LatLon_to_ECEF(p, r=EARTH_MEAN_RADIUS):
  """Earth centered, Earth fixed coordinates to Latitude and Longitude"""
  radius = r * torch.ones_like(p[0])
  theta = torch.pi/2 - p[0]
  phi = p[1] + torch.pi
  ret = torch.vstack([radius, theta, phi])
  ret = sph_to_cart(ret)
  return ret
  

def ECI_to_LatLon(p, t):
  """Earth centered inertial coordinates to Latitude and Longitude"""
  ret = ECI_to_ECEF(p, t)
  return ECEF_to_LatLon(ret)


def LatLon_to_Mercator(p):
  """Latitude and Longitude to Mercator coordinates"""
  ret = torch.zeros_like(p)
  lat, lon = p
  ret[0] = lon*EARTH_MEAN_RADIUS
  ret[1] = torch.log(torch.tan(0.25*torch.pi + 0.5*lat))*EARTH_MEAN_RADIUS
  return ret


def Mercator_to_LatLon(p):
  """Mercator coordinates to Latitude and Longitude"""
  ret = torch.zeros_like(p)
  x, y = p
  ret[0] = 2*torch.arctan(torch.exp(y / EARTH_MEAN_RADIUS)) - torch.pi * 0.5
  ret[1] = x / EARTH_MEAN_RADIUS
  return ret


def Tilted_to_ECI(p):
  """Tilted to Earth centered inertial coordinates"""
  earth_tilt = np.deg2rad(EARTH_TILT)
  rot = torch.tensor([
    [ 1, 0                 ,  0                 ],
    [ 0, np.cos(earth_tilt), -np.sin(earth_tilt)],
    [ 0, np.sin(earth_tilt),  np.cos(earth_tilt)]
  ], dtype=p.dtype, device=p.device) # (3, 3)
  return torch.einsum('ij, jk->ik', rot, p) # (3, 3).(3, points) -> (3, points)


def ECI_to_Corotating(p, anomaly):
  """Earth centered inertial to Corotating coordinates"""
  cos_vals = torch.cos(anomaly)
  sin_vals = torch.sin(anomaly)
  rot = torch.stack([
    torch.stack([ cos_vals                 , sin_vals                 , torch.zeros_like(anomaly)], dim=0),
    torch.stack([-sin_vals                 , cos_vals                 , torch.zeros_like(anomaly)], dim=0),
    torch.stack([ torch.zeros_like(anomaly), torch.zeros_like(anomaly), torch.ones_like(anomaly) ], dim=0)
  ], dim=0) # (3, 3, anomaly_points)
  return torch.einsum('ijk, jk->ik', rot, p) # (3, 3, anomaly_points).(3, points) -> (3, points)
 

def ECEF_to_Corotating(p, t):
  """Earth centered, Earth fixed to Corotating coordinates"""
  angle = 2*torch.pi / EARTH_ROTATION_PERIOD * t # (orbit_points,)
  cos_vals = torch.cos(angle)
  sin_vals = torch.sin(angle)
  rot = torch.stack([
    torch.stack([ cos_vals               ,-sin_vals               , torch.zeros_like(angle)], dim=0),
    torch.stack([ sin_vals               , cos_vals               , torch.zeros_like(angle)], dim=0),
    torch.stack([ torch.zeros_like(angle), torch.zeros_like(angle), torch.ones_like(angle) ], dim=0)
  ], dim=0) # (3, 3, angle_points)
  return torch.einsum('ijk, jk...->ik...', rot, p) # (3, 3, orbit_points).(3, orbit_points) -> (3, orbit_points)


def dot(x, y):
  """Regular dot product"""
  return torch.einsum('ik, ik -> k', x, y)


def load_farm_data(device='cpu', dtype=torch.float32, save_new=True):
  """Load pre-computed numpy array if it exists, else load from excel file."""
  import pandas as pd
  import pathlib
  from safetensors.numpy import load_file, save_file

  if pathlib.Path(PRECOMPUTED_FARMS_DIR).exists():
    try:
      tensors = load_file(PRECOMPUTED_FARMS_DIR)
      farm_data = torch.from_numpy(tensors["farm_data"]).to(device=device, dtype=dtype)
      if farm_data.dtype == dtype:
        return farm_data
    except:
      pass

  excel = pd.ExcelFile(EXCEL_FARMS_DIR)
  df_large = pd.read_excel(excel, "Large Utility-Scale") # farms over 20MW (10MW for Arab countries)
  df_medium = pd.read_excel(excel, "Medium Utility-Scale") # farms lower than 10MW (10MW for Arab countries)
  df = pd.concat([df_large, df_medium], ignore_index=True, axis=0) # all farms
  df = df.loc[
      (~df['Country'].isin(EXCLUDE_COUNTRIES))
    & (df['Status'].isin(INCLUDE_FARM_STATUS))
  ].reset_index(drop=True) # filter to exclude forbidden countries and focus on selected statuses

  def process_same(group):
    tot_power = group['Capacity (MW)'].sum()
    lat = group['Latitude'] * group['Capacity (MW)'] / tot_power
    lon = group['Longitude'] * group['Capacity (MW)'] / tot_power
    return pd.DataFrame({'Capacity (MW)': tot_power, 'Latitude': lat.sum(), 'Longitude': lon.sum()}, index=[group.index[0]])

  def process_cumulative(group):
    group['Latitude'] = (group['Latitude']*group['Capacity (MW)']).cumsum()
    group['Longitude'] = (group['Longitude']*group['Capacity (MW)']).cumsum()
    group['Capacity (MW)'] = group['Capacity (MW)'].cumsum()
    group['Latitude'] = group['Latitude'] / group['Capacity (MW)']
    group['Longitude'] = group['Longitude'] / group['Capacity (MW)']
    return group
  
  df['Status'] = df['Status'].map(lambda x: farm_status_encode[x]) # encode farm status into integer for sorting
  df = df.groupby(['Project Name', 'Status']) # fetch different instances of the same farm project with the same operational status
  df = df.apply(process_same) # sum their power capacities and do a weighted average of their coordinates
  df = df.reset_index(level=['Project Name', 'Status']).reset_index(drop=True)
  df = df.sort_values(['Project Name', 'Status']) # order farms according to their status
  df = df.groupby('Project Name') # fetch different instances of the same farm project with different operational status
  df = df.apply(process_cumulative) # cumulatively sum their power capacities and do a weighted average of their coordinates
  df = df.filter(items=['Latitude', 'Longitude', 'Capacity (MW)', 'Status']) # keep relevant data
  
  farm_data = torch.tensor(df.values, dtype=dtype, device=device)
  farm_data = farm_data.T # (4, ALL_FARMS)
  farm_data[:2] = torch.deg2rad(farm_data[:2]) # Lat/Lon in radians
  farm_data[2] *= 1000 # power in kW
  
  if save_new:
    save_file({"farm_data": farm_data.cpu().numpy()}, PRECOMPUTED_FARMS_DIR) # save pre-computed array

  return farm_data
  
  
def get_farm_area_stats(farm_data, device='cpu', dtype=torch.float32):
  old_farms = []
  # Note: a higher threshold helps Reflect because larger farms cover more m2 per MW. This particular number was coarsely calculated by averaging the optimal farms to transfer to along the best orbit with the top 800 farms that are either alrady operating or in construction.
  threshold = 300
  with open(OLD_FARMS_DIR, "r") as f:
    for line in f.readlines():
      data = line.split("~")
      power, pv_coverage, ground_coverage = [float(p) for p in data[2:5]]
      if power > threshold: 
        old_farms.append([power, pv_coverage, ground_coverage])
  
  old_farms = torch.tensor(old_farms, dtype=dtype, device=device)
  farm_km2_per_MW = torch.mean(old_farms[:, 2]/old_farms[:, 0])
  ground_utilization_rate = torch.mean(old_farms[:, 1]/old_farms[:, 2])
  
  farm_m2_per_kw = farm_km2_per_MW * 1000
  
  return farm_data[2] * farm_m2_per_kw, ground_utilization_rate