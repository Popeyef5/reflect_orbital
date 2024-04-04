import os

##################
# General settings
##################

# Sim
POINTS_PER_DAY = 86400 // 4 # ~ 1 point every 4s in a daily orbit
POINTS_PER_YEAR = 365 // 15 # 1 point every 15 days in a yearly orbit around the Sun
ORBIT_PHASES = 64 # orbit phases can be considered phase shifts in the ground tracks: which longitude does the RAAN cross at time=0?
ORBIT_FLAVOURS = 2

FARM_COSINE_LOSSES = True # consider losses due to the satellite not being perfectly on top of the farm
FARM_AREA_LOSSES = True # consider losses due to the beam area being larger than the farm area

# Data
EXCLUDE_COUNTRIES = ["CHN", "China"] # no farms from the countries in this list 
INCLUDE_FARM_STATUS = "operating", "construction", "pre-construction", "announced" # choose all that apply
DROPOUTS = 1-1, 1-4/5, 1-3/5, 1-2/5, 1-1/5, 1-1/10 # each farm is a client with P=1-DROPOUT

farm_status_encode = {
  "operating": 1 << 0,
  "construction": 1 << 1,
  "pre-construction": 1 << 2,
  "announced": 1 << 3
}

# Storage
USE_PRECOMPUTED_FARMS = int(os.environ.get("USE_PRECOMPUTED_FARMS", 1))
PRECOMPUTED_FARMS_DIR = os.environ.get("PRECOMPUTED_FARMS_DIR", "extra/farm_data.safetensors")
EXCEL_FARMS_DIR = os.environ.get("EXCEL_FARMS_DIR", "extra/farm_data.xlsx")
OLD_FARMS_DIR = os.environ.get("OLD_FARMS_DIR", "extra/area_farm_data.txt")

DEFAULT_RESULTS_DIR = os.environ.get("OUT", f"out/results.safetensors")

###################
# General constants
###################

# Earth
EARTH_ROTATION_PERIOD = 86400
EARTH_MEAN_RADIUS = 6367444.65
EARTH_TILT = 23.4

# Sun
SUN_SUBTENDED_ANGLE = 0.5

# Satellite
ORBIT_ALTITUDE = 600000 # (https://en.wikipedia.org/wiki/Sun-synchronous_orbit)
ORBIT_PERIOD = EARTH_ROTATION_PERIOD / 15 # (https://en.wikipedia.org/wiki/Sun-synchronous_orbit)
ORBIT_INCLINATION = 97.7 # (https://en.wikipedia.org/wiki/Sun-synchronous_orbit)

# Beam
MIN_BEAM_RADIUS = 15000

# Solar power
SPACE_SOLAR_kWpm2 = 1.361 # (kW/m^2) https://en.wikipedia.org/wiki/Solar_irradiance
MYLAR_REFLECTIVITY = 0.9 # https://www.hydroponics.eu/mylar-silver-reflective-sheeting-15-x-1-2mt~31966.html

# Solar farms
PV_CELL_EFFICIENCY = 0.2 # https://en.wikipedia.org/wiki/Solar-cell_efficiency
SOLAR_INVERTER_LOSSES = 0.085 # ChatGPT and https://www.pv-magazine.com/2023/03/02/guide-to-understanding-solar-production-losses/

# Space industry
#KG_COST_TO_ORBIT = 2800 # Falcon 9
KG_COST_TO_ORBIT = 1500 # Falcon Heavy
KG_COST_TO_ORBIT = 200 # Starship

# Reflect
ENERGY_PRICE = 83e-3
SUNNY_SKY_FREQ = 0.766

SAT_COST = 40e3
SAT_WEIGHT = 160
SAT_LIFE = 20
SAT_SURFACE = 54**2