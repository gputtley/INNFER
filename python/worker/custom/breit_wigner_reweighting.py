from functools import partial
import os
import pickle
import yaml
import numpy as np

def bw_reweight(df, mass_to="bw_mass", mass_from="sim_mass", gen_mass="GenTop1_mass", clip_min=0.1, clip_max=10.0):
  df.loc[:,"wt"] *= np.clip(bw(df.loc[:,mass_to], df.loc[:,gen_mass]) / bw(df.loc[:,mass_from], df.loc[:,gen_mass]), clip_min, clip_max)
  return df


def width(m):
  return ((0.0270*m) - 3.3455)


def bw(m, gen_m):
  return (1/(((gen_m**2)-(m**2))**2 + ((m*width(m))**2)))


def bw_fractions(df, spline_locations="spline_locations.yaml", mass_to="bw_mass", mass_from="sim_mass", category="run2", ignore_quantile=0.05):
  
  # Load splines (yaml file)
  with open(spline_locations, "r") as f:
    splines = yaml.safe_load(f)[category]

  # Loop through mass_from values
  spl = {}
  for k, v in splines.items():

    # Check if spline file exists
    if not os.path.isfile(v):
      raise FileNotFoundError(f"Spline file not found: {v}")
    
    # Load spline
    with open(v, "rb") as f:
      spl[k] = pickle.load(f)

    # Cap spline function to a minimum of 0
    def SplineWithMinZero(x, spline_func=spl[k]):
      result = spline_func(x)
      result[result < ignore_quantile] = 0.0
      return result
    spl[k] = partial(SplineWithMinZero, spline_func=spl[k])

    # Get mask for mass_from
    mask = (df.loc[:,mass_from] == float(k))

    # Apply reweighting
    df.loc[mask,"wt"] *= spl[k](df.loc[mask,mass_to])
    
  return df