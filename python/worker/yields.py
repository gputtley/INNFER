import copy

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

    self.yield_dataframe = yield_dataframe
    self.shape_pois = [poi for poi in pois if poi in self.yield_dataframe.columns]
    self.nuisances = [nuisance for nuisance in nuisances if nuisance in self.yield_dataframe.columns]
    self.method = method
    self.column_name = column_name
    self.file_name = file_name

    self.rate_scale = 1.0

  def _CheckIfInDataFrame(self, Y, df, return_matching=False):
    """
    Checks if the values in DataFrame Y are present in df.

    Parameters
    ----------
    Y : pandas.DataFrame
        DataFrame containing values to check.
    df : pandas.DataFrame
        DataFrame to check against.
    return_matching : bool, optional
        If True, return the indices of matching rows (default is False).

    Returns
    -------
    bool or list
        Boolean indicating if values are present, or list of matching indices if return_matching is True.
    """

    sorted_columns = sorted(list(Y.columns))
    Y = Y.loc[:, sorted_columns]
    df = df.loc[:, sorted_columns]
    matching = (df == Y.iloc[0])
    if not return_matching:
      return matching.all(axis=1).any(axis=0)
    else:
      return matching[matching.all(axis=1)].index.to_list()
      #if len(list(matching.columns)) == 0:
      #  return matching.index.to_list()
      #else:
      #  return matching[matching[matching.columns[0]]].index.to_list()

  def Default(self, Y):
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

    # Get Y with nuisances all equal to 0
    nominal_values_to_match = pd.DataFrame([[0.0]*len(self.nuisances)], columns = self.nuisances, dtype=np.float64)
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

  def ExtendRanges(self, X, Y, extend_range_by=5):
    """
    Performs an extension of the parameter ranges.

    Parameters
    ----------
    X : pandas.Series
        Series containing the axis to extend.
    Y : pandas.Series
        Series containing the values.

    Returns
    -------
    pandas.Series
        Extended Series.
    """

    # Find minimum and maxixum  X values
    min_index = X.nsmallest(1).index.tolist()[0]
    min_indices = X.nsmallest(2).index.tolist()
    min_2_index = [i for i in min_indices if i != min_index][0]
    max_index = X.nlargest(1).index.tolist()[0]
    max_indices = X.nlargest(2).index.tolist()
    max_2_index = [i for i in max_indices if i != max_index][0]

    # Find new min and max
    current_range = X[max_index] - X[min_index]
    new_range = extend_range_by * current_range
    new_max = X[max_index] + ((new_range - current_range)/2)
    new_min = X[min_index] - ((new_range - current_range)/2)

    # Build linear function
    m_top = (Y[max_index] - Y[max_2_index]) / (X[max_index] - X[max_2_index])
    c_top = Y[max_2_index] - m_top * X[max_2_index]    
    m_bottom = (Y[min_2_index] - Y[min_index]) / (X[min_2_index] - X[min_index])
    c_bottom = Y[min_index] - m_bottom * X[min_index]    

    # Extrapolate and tidy
    X_list = list(X.to_numpy()) + [new_min, new_max]
    Y_list = list(Y.to_numpy()) + [(m_bottom*new_min) + c_bottom, (m_top*new_max) + c_top]
    zipped_lists = list(zip(X_list, Y_list))
    sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[0])
    X_list, Y_list = zip(*sorted_zipped_lists)
    X = pd.Series(X_list, name=X.name)
    Y = pd.Series(Y_list, name=Y.name)

    return X, Y

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

    # Separate shape and rate Y terms
    if f"mu_{self.file_name}" in Y.columns and not ignore_rate:
      self.rate_scale = Y.loc[:,f"mu_{self.file_name}"].iloc[0]

    # Get shape varying Y parameters
    Y = Y.loc[:, [col for col in Y.columns if not col.startswith("mu_")]]

    # Get Y in the yield datasets
    Y = Y.loc[:, [col for col in Y.columns if col in list(self.yield_dataframe.columns)]]

    # If no POIs or nuisances just return the total scale
    if len(self.shape_pois) + len(self.nuisances) == 0:
      return self.rate_scale * float(self.yield_dataframe.loc[:,self.column_name].iloc[0])

    # If Y already in the unique values list then just use that
    if self._CheckIfInDataFrame(Y, self.yield_dataframe):
      matched_rows = self._CheckIfInDataFrame(Y, self.yield_dataframe, return_matching=True)
      return self.rate_scale * float(self.yield_dataframe.loc[matched_rows,self.column_name].iloc[0])

    # If no POI or POI in unique values list, and nuisance is not, do nuisance interpolation
    if (len(self.shape_pois) == 0) or self._CheckIfInDataFrame(Y.loc[:,self.shape_pois], self.yield_dataframe):
      return self.rate_scale * float(self.NuisanceInterpolation(Y))

    # Do POI interpolation
    if self.method == "default":
      if len(self.shape_pois) > 1:
        raise ValueError('Default model only works for 0 or 1 shape POIs. Please write your own yield function for your case.')
      yield_from_interp = self.Default(Y)
      if yield_from_interp is not None:  
        return self.rate_scale * yield_from_interp
      else:
        return np.nan

  def NuisanceInterpolation(self, Y):
    """
    Performs interpolation for nuisance parameters.

    Parameters
    ----------
    Y : pandas.DataFrame
        DataFrame containing the POI and nuisance values.

    Returns
    -------
    float
        Interpolated yield value.
    """

    # Get nominal yield (where nuisances parameters are zero)
    nominal_values_to_match = pd.DataFrame([[Y.loc[:,poi].iloc[0] for poi in self.shape_pois] + [0.0]*len(self.nuisances)], columns = self.shape_pois + self.nuisances, dtype=np.float64)
    matched_rows = self._CheckIfInDataFrame(nominal_values_to_match, self.yield_dataframe, return_matching=True)
    nominal = self.yield_dataframe.loc[matched_rows, self.column_name].iloc[0]

    # Set this to the initial total yield
    total_yield = copy.deepcopy(nominal)

    # Loop thrpugh nuisance parameters
    for col in self.nuisances:

      # Get yields when only this nuisance is varied
      other_nuisances = [k for k in self.nuisances if k != col]
      values_to_match = pd.DataFrame([[Y.loc[:,poi].iloc[0] for poi in self.shape_pois] + [0.0]*len(other_nuisances)], columns = self.shape_pois + other_nuisances, dtype=np.float64)
      matched_rows = self._CheckIfInDataFrame(values_to_match, self.yield_dataframe, return_matching=True)

      # Make interpolation function
      df_to_interp = self.yield_dataframe.loc[matched_rows, [col,self.column_name]].sort_values(by=col)

      # Extend nuisance ranges
      nui_val, yield_val = self.ExtendRanges(df_to_interp.loc[:,col], df_to_interp.loc[:,self.column_name])

      # Make interpolator
      f = interp1d(nui_val, yield_val, bounds_error=False, fill_value=np.nan)

      # Interpolate and adjust total_yield
      total_yield += (f(Y.loc[:,col].iloc[0]) - nominal)

    return total_yield