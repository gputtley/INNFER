import copy
import time

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

class Yields():

  def __init__(
      self, 
      yield_dataframe, 
      pois, 
      nuisances, 
      file_name, 
      rate_params = [], 
      method = "default", 
      column_name = "yields"
    ):
    """
    A class used to compute and interpolate yield values based on parameters of interest (POIs) and nuisances.

    Parameters
    ----------
    yield_dataframe : pandas.DataFrame
        DataFrame containing yield data.
    pois : list
        List of parameters of interest.
    nuisances : list
        List of nuisance parameters.
    file_name : str
        Name of the file.
    rate_params : list, optional
        List of rate parameters (default is []).
    method : str, optional
        Method for interpolation (default is "default").
    column_name : str, optional
        Name of the column containing yield values (default is "yields").
    """

    #self.yield_dataframe = yield_dataframe
    self.shape_pois = [poi for poi in pois if poi in yield_dataframe.columns]
    self.nuisances = [nuisance for nuisance in nuisances if nuisance in yield_dataframe.columns]
    self.yield_dataframe = yield_dataframe.loc[(yield_dataframe.loc[:,column_name] != 0) ,self.shape_pois + self.nuisances + [column_name]]
    self.method = method
    self.column_name = column_name
    self.file_name = file_name

    self.rate_scale = 1.0

  def _GetClosestValues(self, Y, column):

    column_values = sorted(list(set(self.yield_dataframe.loc[:,column].to_numpy().flatten())))
    closest_up = pd.DataFrame({column:np.zeros(len(Y))})
    closest_down = pd.DataFrame({column:np.zeros(len(Y))})

    # set below thresholds
    closest_down.loc[(Y.loc[:,column]<column_values[0]),column] = column_values[0]
    closest_up.loc[(Y.loc[:,column]<column_values[0]),column] = column_values[1]

    # set within the thresholds
    for column_ind in range(len(column_values)-1):
      closest_down.loc[((Y.loc[:,column]>=column_values[column_ind]) & (Y.loc[:,column]<column_values[column_ind+1])),column] = column_values[column_ind]
      closest_up.loc[((Y.loc[:,column]>=column_values[column_ind]) & (Y.loc[:,column]<column_values[column_ind+1])),column] = column_values[column_ind+1]

    # set above the thresholds
    closest_down.loc[(Y.loc[:,column]>=column_values[-1]),column] = column_values[-2]
    closest_up.loc[(Y.loc[:,column]>=column_values[-1]),column] = column_values[-1]    

    return closest_down, closest_up

  def Default(self, Y, column):
    """
    Default method for interpolation when there is one shape POI.

    Parameters
    ----------
    Y : pandas.DataFrame
        DataFrame containing the POI and nuisance values.

    Returns
    -------
    float or None
        Interpolated yield value or None if interpolation is not possible.
    """

    # get closest values
    closest_down, closest_up = self._GetClosestValues(Y, column)

    # Make into full Y inputs  
    for col in Y.columns:
      if col != column:
        closest_down.loc[:,col] = Y.loc[:,col]
        closest_up.loc[:,col] = Y.loc[:,col]

    # get closest down yield value
    df_m = copy.deepcopy(self.yield_dataframe)
    df_m['original_index'] = df_m.index
    merged_df_for_yield = pd.merge(closest_down, df_m, on=list(closest_down.columns), how='inner')
    yield_dataframe_indices = merged_df_for_yield['original_index'].to_list()
    closest_down_value = self.yield_dataframe.loc[yield_dataframe_indices, self.column_name].reset_index(drop=True)

    # get closest up yield value
    df_m = copy.deepcopy(self.yield_dataframe)
    df_m['original_index'] = df_m.index
    merged_df_for_yield = pd.merge(closest_up, df_m, on=list(closest_down.columns), how='inner')
    yield_dataframe_indices = merged_df_for_yield['original_index'].to_list()
    closest_up_value = self.yield_dataframe.loc[yield_dataframe_indices, self.column_name].reset_index(drop=True)

    # Get poi interpolation
    grads = (closest_up_value-closest_down_value)/(closest_up.loc[:,column] - closest_down.loc[:,column])
    yields = pd.DataFrame({"yield":np.zeros(len(Y))})
    yields.loc[:,"yield"] = (closest_down_value + (grads*(Y.loc[:,column]-closest_down.loc[:,column])))

    return yields

  def GetYield(self, Y, ignore_rate=False):
    """
    Computes the yield based on the given POI and nuisance values.

    Parameters
    ----------
    Y : pandas.DataFrame
        DataFrame containing the POI and nuisance values.
    ignore_rate : bool, optional
        If True, the rate scaling is ignored (default is False).

    Returns
    -------
    float
        Computed yield value.
    """

    # If yield dataframe is empty return 0
    if len(self.yield_dataframe) == 0:
      return 0.0

    # Reindex Y
    Y = Y.reset_index(drop=True)

    # Set up dataframes to return
    df = pd.DataFrame({"yield":np.ones(len(Y))})

    # Separate shape and rate Y terms
    if f"mu_{self.file_name}" in Y.columns and not ignore_rate:
      df *= Y.loc[:,f"mu_{self.file_name}"].to_numpy()

    # Get shape varying Y parameters
    Y = Y.loc[:, [col for col in Y.columns if not col.startswith("mu_")]]

    # Get Y in the yield datasets
    Y = Y.loc[:, [col for col in Y.columns if col in list(self.yield_dataframe.columns)]]

    ### Get nominal yield ##
    if len(self.shape_pois) > 0:
      nominal_Y = Y.loc[:,self.shape_pois]
      for nui in self.nuisances:
        nominal_Y.loc[:,nui] = np.zeros(len(nominal_Y))
      nominal_yields = self.Default(nominal_Y, self.shape_pois[0]) * df
    elif len(self.nuisances) > 0:
      nominal_Y = Y.loc[:,self.nuisances]
      for nui in self.nuisances:
        nominal_Y.loc[:,nui] = np.zeros(len(nominal_Y))
      nominal_yields = self.Default(nominal_Y, self.nuisances[0]) * df
    else:
      constant_yield = self.yield_dataframe.loc[:,self.column_name].iloc[0]
      nominal_yields = pd.DataFrame({"yield": np.full(len(Y), constant_yield)}) * df

    # loop through nuisances
    yields = copy.deepcopy(nominal_yields)
    for nui in self.nuisances:

      # Find the closest nuisance shifts
      closest_down, closest_up = self._GetClosestValues(Y, nui)

      if len(self.shape_pois) > 0:

        # Get down shift
        shifted_down_Y = copy.deepcopy(nominal_Y)
        shifted_down_Y.loc[:,nui] = closest_down.loc[:,nui]
        shifted_down_yields = self.Default(shifted_down_Y, self.shape_pois[0]) * df

        # Get up shift
        shifted_up_Y = copy.deepcopy(nominal_Y)
        shifted_up_Y.loc[:,nui] = closest_up.loc[:,nui]
        shifted_up_yields = self.Default(shifted_up_Y, self.shape_pois[0]) * df

      elif len(self.nuisances) > 0:

        # Get down shift
        shifted_down_Y = copy.deepcopy(nominal_Y)
        shifted_down_Y.loc[:,nui] = closest_down.loc[:,nui]
        shifted_down_yields = self.Default(shifted_down_Y, self.nuisances[0]) * df

        # Get up shift
        shifted_up_Y = copy.deepcopy(nominal_Y)
        shifted_up_Y.loc[:,nui] = closest_up.loc[:,nui]
        shifted_up_yields = self.Default(shifted_up_Y, self.nuisances[0]) * df

      # Get nuisance interpolation
      grads = (shifted_up_yields.loc[:,"yield"]-shifted_down_yields.loc[:,"yield"])/(shifted_up_Y.loc[:,nui] - shifted_down_Y.loc[:,nui])
      shifted_yields = pd.DataFrame({"yield":np.zeros(len(Y))})
      shifted_yields.loc[:,"yield"] = (shifted_down_yields.loc[:,"yield"] + (grads*(Y.loc[:,nui]-shifted_down_Y.loc[:,nui])))

      # Put into yields
      yields += (shifted_yields - nominal_yields)

    if len(yields) > 1:
      return yields.loc[:,"yield"]
    else:
      return float(yields.loc[0,"yield"])
