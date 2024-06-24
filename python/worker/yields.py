import copy
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class Yields():

  def __init__(self, yield_dataframe, pois, nuisances, file_name, rate_params=[], method="default", column_name="yields"):

    self.yield_dataframe = yield_dataframe
    self.shape_pois = [poi for poi in pois if poi in self.yield_dataframe.columns]
    self.nuisances = [nuisance for nuisance in nuisances if nuisance in self.yield_dataframe.columns]
    self.method = method
    self.column_name = column_name
    self.file_name = file_name

    self.rate_scale = 1.0

  def _CheckIfInDataFrame(self, Y, df, return_matching=False):
    sorted_columns = sorted(list(Y.columns))
    Y = Y.loc[:, sorted_columns]
    df = df.loc[:, sorted_columns]
    matching = (df == Y.iloc[0])
    if not return_matching:
      return matching.any().iloc[0]
    else:
      if len(list(matching.columns)) == 0:
        return matching.index.to_list()
      else:
        return matching[matching[matching.columns[0]]].index.to_list()


  def GetYield(self, Y, ignore_rate=False):

    # Separate shape and rate Y terms
    if f"mu_{self.file_name}" in Y.columns and not ignore_rate:
      self.rate_scale = Y.loc[:,f"mu_{self.file_name}"].iloc[0]

    # Get shape varying Y parameters
    Y = Y.loc[:, [col for col in Y.columns if not col.startswith("mu_")]]

    # If no POIs or nuisances just return the total scale
    if len(self.shape_pois) + len(self.nuisances) == 0:
      return self.rate_scale * float(self.yield_dataframe.loc[:,self.column_name].iloc[0])

    # If Y already in the unique values list then just use that
    if self._CheckIfInDataFrame(Y, self.yield_dataframe):
      matched_rows = self._CheckIfInDataFrame(Y, self.yield_dataframe, return_matching=True)
      return self.rate_scale * float(self.yield_dataframe.loc[matched_rows,self.column_name].iloc[0])

    # If no POI or POI in unique values list, and nuisance is not, do nuisance interpolation
    if (len(self.shape_pois) == 0) or self._CheckIfInDataFrame(Y.loc[:,self.shape_pois], self.yield_dataframe):
      return self.rate_scale * float(self.DoNuisanceInterpolation(Y))

    # Do POI interpolation
    if self.method == "default":
      if len(self.shape_pois) > 1:
        raise ValueError('Default model only works for 0 or 1 shape POIs. Please write your own yield function for your case.')
      yield_from_interp = self.Default(Y)
      if yield_from_interp is not None:  
        return self.rate_scale * yield_from_interp
      else:
        return np.nan

  def Default(self, Y):

    # Get Y with nuisances all equal to 0
    nominal_values_to_match = pd.DataFrame([[0.0]*len(self.nuisances)], columns = self.nuisances)
    matched_rows = self._CheckIfInDataFrame(nominal_values_to_match, self.yield_dataframe, return_matching=True)
    poi_vals = self.yield_dataframe.loc[matched_rows,self.shape_pois]

    # Find values either side of Y
    index = np.searchsorted(sorted(poi_vals.to_numpy().flatten()), Y.loc[:,self.shape_pois].iloc[0])
    if (index == 0) or (index == len(poi_vals)): 
      return None
    min_val = poi_vals.to_numpy().flatten()[index-1]
    max_val = poi_vals.to_numpy().flatten()[index]
    min_Y = copy.deepcopy(Y)
    max_Y = copy.deepcopy(Y)
    min_Y.loc[:,self.shape_pois] = min_val
    max_Y.loc[:,self.shape_pois] = max_val

    # If nuisances, do nuisance interpolation
    min_Y_yield = self.NuisanceInterpolation(min_Y)
    max_Y_yield = self.NuisanceInterpolation(max_Y)

    # Interpolate pois
    f = interp1d([min_val[0], max_val[0]], [min_Y_yield, max_Y_yield])
    return f(Y.loc[:,self.shape_pois].iloc[0])[0]


  def NuisanceInterpolation(self, Y):

    # Get nominal yield (where nuisances parameters are zero)
    nominal_values_to_match = pd.DataFrame([[Y.loc[:,poi].iloc[0] for poi in self.shape_pois] + [0.0]*len(self.nuisances)], columns = self.shape_pois + self.nuisances)
    matched_rows = self._CheckIfInDataFrame(nominal_values_to_match, self.yield_dataframe, return_matching=True)
    nominal = self.yield_dataframe.loc[matched_rows, self.column_name].iloc[0]

    # Set this to the initial total yield
    total_yield = copy.deepcopy(nominal)

    # Loop thrpugh nuisance parameters
    for col in self.nuisances:

      # Get yields when only this nuisance is varied
      other_nuisances = [k for k in self.nuisances if k != col]
      values_to_match = pd.DataFrame([[Y.loc[:,poi].iloc[0] for poi in self.shape_pois] + [0.0]*len(other_nuisances)], columns = self.shape_pois + other_nuisances)
      matched_rows = self._CheckIfInDataFrame(values_to_match, self.yield_dataframe, return_matching=True)

      # Make interpolation function
      df_to_interp = self.yield_dataframe.loc[matched_rows, [col,self.column_name]].sort_values(by=col)
      f = interp1d(df_to_intep.loc[:,col], df_to_intep.loc[:,self.column_name])

      # Interpolate and adjust total_yield
      total_yield += (f(Y.loc[:,col].iloc[0]) - nominal)  

    return total_yield
