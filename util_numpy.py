from settings import *

import numpy as np


def cart_to_sph(p):
  """Cartesian to spherical coordinates"""
  ret = np.zeros(p.shape, dtype=p.dtype)
  xy = p[0]**2 + p[1]**2
  ret[0] = np.sqrt(xy + p[2]**2)
  ret[1] = np.arctan2(np.sqrt(xy), p[2])
  ret[2] = np.arctan2(p[1], p[0])
  return ret


def sph_to_cart(p):
  """Spherical to cartesian coordinates"""
  ret = np.zeros(p.shape, dtype=p.dtype)
  r, theta, phi = p[0], p[1], p[2]
  ret[0] = r * np.sin(theta) * np.cos(phi)
  ret[1] = r * np.sin(theta) * np.sin(phi)
  ret[2] = r * np.cos(theta)
  return ret


def ECI_to_ECEF(p, t):
  """Earth centered inertial to Earth centered, Earth fixed coordinates"""
  angle = 2*np.pi / EARTH_ROTATION_PERIOD * t # (orbit_points,)
  rot = np.array([
    [ np.cos(angle)        , np.sin(angle)        , np.zeros(angle.shape)],
    [-np.sin(angle)        , np.cos(angle)        , np.zeros(angle.shape)],
    [ np.zeros(angle.shape), np.zeros(angle.shape), np.ones(angle.shape) ]
  ], dtype=p.dtype) # (3, 3, orbit_points)
  return np.einsum('ijk, jk->ik', rot, p) # (3, 3, orbit_points).(3, orbit_points) -> (3, orbit_points)


def ECEF_to_LatLon(p):
  """Earth centered, Earth fixed coordinates to Latitude and Longitude"""
  ret = cart_to_sph(p)
  lat = np.pi/2 - ret[1]
  lon = ret[2] % (2*np.pi) - np.pi
  return np.array([lat, lon], dtype=p.dtype)


def LatLon_to_ECEF(p, r=EARTH_MEAN_RADIUS):
  """Earth centered, Earth fixed coordinates to Latitude and Longitude"""
  radius = r * np.ones(p[0].shape, dtype=p.dtype)
  theta = np.pi/2 - p[0]
  phi = p[1] + np.pi
  ret = np.vstack([radius, theta, phi])
  ret = sph_to_cart(ret)
  return ret
  

def ECI_to_LatLon(p, t):
  """Earth centered inertial coordinates to Latitude and Longitude"""
  ret = ECI_to_ECEF(p, t)
  return ECEF_to_LatLon(ret)


def LatLon_to_Mercator(p):
  """Latitude and Longitude to Mercator coordinates"""
  ret = np.zeros(p.shape, dtype=p.dtype)
  lat, lon = p
  ret[0] = lon*EARTH_MEAN_RADIUS
  ret[1] = np.log(np.tan(0.25*np.pi + 0.5*lat))*EARTH_MEAN_RADIUS
  return ret


def Mercator_to_LatLon(p):
  """Mercator coordinates to Latitude and Longitude"""
  ret = np.zeros(p.shape, dtype=p.dtype)
  x, y = p
  ret[0] = 2*np.arctan(np.exp(y / EARTH_MEAN_RADIUS)) - np.pi * 0.5
  ret[1] = x / EARTH_MEAN_RADIUS
  return ret


def Tilted_to_ECI(p):
  """Tilted to Earth centered inertial coordinates"""
  earth_tilt = np.deg2rad(EARTH_TILT)
  rot = np.array([
    [ 1, 0                 ,  0                 ],
    [ 0, np.cos(earth_tilt), -np.sin(earth_tilt)],
    [ 0, np.sin(earth_tilt),  np.cos(earth_tilt)]
  ], dtype=p.dtype) # (3, 3)
  return np.einsum('ij, jk->ik', rot, p) # (3, 3).(3, points) -> (3, points)


def ECI_to_Corotating(p, anomaly):
  """Earth centered inertial to Corotating coordinates"""
  rot = np.array([
    [ np.cos(anomaly)        , np.sin(anomaly)        , np.zeros(anomaly.shape)],
    [-np.sin(anomaly)        , np.cos(anomaly)        , np.zeros(anomaly.shape)],
    [ np.zeros(anomaly.shape), np.zeros(anomaly.shape), np.ones(anomaly.shape) ]
  ], dtype=p.dtype) # (3, 3, anomaly_points)
  return np.einsum('ijk, jk->ik', rot, p) # (3, 3, anomaly_points).(3, points) -> (3, points)
 

def ECEF_to_Corotating(p, t):
  """Earth centered, Earth fixed to Corotating coordinates"""
  angle = 2*np.pi / EARTH_ROTATION_PERIOD * t # (orbit_points,)
  rot = np.array([
    [ np.cos(angle)        , -np.sin(angle)        , np.zeros(angle.shape)],
    [ np.sin(angle)        ,  np.cos(angle)        , np.zeros(angle.shape)],
    [ np.zeros(angle.shape),  np.zeros(angle.shape), np.ones(angle.shape) ]
  ], dtype=p.dtype) # (3, 3, orbit_points)
  return np.einsum('ijk, jk...->ik...', rot, p) # (3, 3, orbit_points).(3, orbit_points) -> (3, orbit_points)


def dot(x, y):
  """Regular dot product"""
  return np.einsum('ik, ik -> k', x, y)


def load_farm_data():
  """Load pre-computed numpy array if it exists, else load from excel file."""
  import pandas as pd
  import pathlib
  from safetensors.numpy import load_file, save_file

  if pathlib.Path(PRECOMPUTED_FARMS_DIR).exists():
    tensors = load_file(PRECOMPUTED_FARMS_DIR)
    return tensors["farm_data"]
    
  excel = pd.ExcelFile(EXCEL_FARMS_DIR)
  df_large = pd.read_excel(excel, "Large Utility-Scale") # farms over 20MW (10MW for Arab countries)
  df_medium = pd.read_excel(excel, "Medium Utility-Scale") # farms lower than 10MW (10MW for Arab countries)
  df = pd.concat([df_large, df_medium], ignore_index=True, axis=0) # all farms
  df = df.loc[
      (~df['Country'].isin(EXCLUDE_COUNTRIES))
    & (df['Status'].isin(INCLUDE_FARM_STATUS))
  ] # filter to exclude forbidden countries and focus on selected statuses
  
  df = df.filter(items=['Latitude', 'Longitude', 'Capacity (MW)', 'Status']) # keep relevant data
  df['Status'] = df['Status'].map(lambda x: farm_status_encode[x])
  
  farm_data = df.to_numpy(dtype=dtype)
  farm_data = farm_data.T # (4, ALL_FARMS)
  farm_data[:2] = np.deg2rad(farm_data[:2]) # Lat/Lon in radians
  farm_data[2] *= 1000 # power in kW
  
  save_file({"farm_data": farm_data}, PRECOMPUTED_FARMS_DIR) # save pre-computed array

  return farm_data
  
  
def get_farm_area_stats(farm_data):
  old_farms = []
  # Note: a higher threshold helps Reflect because larger farms cover more m2 per MW. This particular number was coarsely calculated by averaging the optimal farms to transfer to along the best orbit with the top 800 farms that are either alrady operating or in construction.
  threshold = 300
  with open(OLD_FARMS_DIR, "r") as f:
    for line in f.readlines():
      data = line.split("~")
      power, pv_coverage, ground_coverage = [float(p) for p in data[2:5]]
      if power > threshold: 
        old_farms.append([power, pv_coverage, ground_coverage])
  
  old_farms = np.array(old_farms)
  farm_km2_per_MW = np.average(old_farms[:, 2]/old_farms[:, 0])
  ground_utilization_rate = np.average(old_farms[:, 1]/old_farms[:, 2])
  
  farm_m2_per_kw = farm_km2_per_MW * 1000
  
  return farm_data[2] * farm_m2_per_kw, ground_utilization_rate