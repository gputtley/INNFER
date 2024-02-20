from preprocess import PreProcess
from likelihood import Likelihood
from data_loader import DataLoader
from plotting import plot_histograms, plot_histogram_with_ratio, plot_likelihood, plot_correlation_matrix, plot_stacked_histogram_with_ratio, plot_stacked_unrolled_2d_histogram_with_ratio, plot_validation_summary, plot_heatmap
from other_functions import GetYName, MakeYieldFunction, MakeBinYields
from scipy.integrate import simpson
from functools import partial
import numpy as np
import pandas as pd
import copy
import yaml

class Validation():
  """
  Validation class for evaluating and visualizing the performance of conditional invertible neural network models. 
  """
  def __init__(self, model, options={}):
    """
    Initialize the Validation class.

    Args:
        model: Trained Bayesian neural network model.
        options (dict): Dictionary of options for the validation process.
    """

    # Set up information
    self.model = model
    self.infer = None
    self.data_parameters = {}
    self.pois = []
    self.nuisances = []
    self.data_key = "val"
    self.tolerance = 1e-6
    self.lower_validation_stats = None
    self.do_binned_fit = None
    self.var_and_bins = None
    self.out_dir = "./data/"
    self.plot_dir = "./plots/"
    self.validation_options = {}
    self.calculate_columns_for_plotting = {}
    self._SetOptions(options)
    self.X_columns = self.data_parameters[list(self.data_parameters.keys())[0]]["X_columns"]
    self.X_plot_columns = copy.deepcopy(self.X_columns)


    # Data storage
    self.synth = None
    self.synth_wt = None
    self.synth_row = None
    self.synth_type = None
    self.lkld = None

  def _SetOptions(self, options):
    """
    Set options for the validation process.

    Args:
        options (dict): Dictionary of options for the validation process.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def _GetXAndWts(self, row, columns=None, use_nominal_wt=False, return_full=False, specific_file=None, return_y=False, ignore_infer=False, transform=False, add_columns=False):
    """
    Extract data and weights for a given row.

    Args:
        row (array-like): The row of data for which to extract information.
        columns (list): List of column names for extraction. If None, uses Y_columns from data_parameters.
        use_nominal_wt (bool): Whether to use nominal weights (default is False).

    Returns:
        X (numpy.ndarray): Extracted data.
        wt (numpy.ndarray): Extracted weights.
    """

    # Do inference on data
    if self.infer is not None and not ignore_infer:
        
        dl = DataLoader(self.infer)
        data = dl.LoadFullDataset()
        total_X = data.loc[:,self.X_columns].to_numpy()
        total_wt = data.loc[:,"wt"].to_numpy().reshape(-1,1)

    # Do inference on simulated data
    else:

      first_loop = True
      for key, val in self.data_parameters.items():

        if specific_file is not None:
          if specific_file != key:
            continue

        # Set up preprocess code to load the data in
        pp = PreProcess()
        pp.parameters = val
        pp.output_dir = val["file_location"]

        # Load and reformat data
        X, Y, wt = pp.LoadSplitData(dataset=self.data_key, get=["X","Y","wt"], use_nominal_wt=use_nominal_wt, untransformX=(not transform))

        if columns is None:
          columns = val["Y_columns"]
        X = X.to_numpy()
        Y = Y.to_numpy()
        wt = wt.to_numpy()

        if not return_full:

          # Choose the matching Y rows
          sel_row = np.array([row[columns.index(y)] for y in val["Y_columns"]])
          if Y.shape[1] > 0:
            matching_rows = np.all(np.isclose(Y, sel_row, rtol=self.tolerance, atol=self.tolerance), axis=1)
            X = X[matching_rows]
            wt = wt[matching_rows]

          # Scale weights to the correct yield
          if "yield" in val.keys():
            if "all" in val["yield"].keys():
              sum_wt = val["yield"]["all"]
            else:
              sum_wt = val["yield"][GetYName(sel_row,purpose="file")]
            old_sum_wt = np.sum(wt, dtype=np.float128)
            wt *= sum_wt/old_sum_wt
        
          # Scale wt by the rate parameter value
          if "mu_"+key in columns:
            rp = row[columns.index("mu_"+key)]
            if rp == 0.0: continue
            wt *= rp

        # Add column
        if add_columns:
          X = self.AddColumns(X)
        else:
          self.X_plot_columns = copy.deepcopy(self.X_columns)

        # Concatenate datasets
        if first_loop:
          first_loop = False
          total_X = copy.deepcopy(X)
          total_wt = copy.deepcopy(wt)
          if return_y:
            total_Y = copy.deepcopy(Y)
        else:
          total_X = np.vstack((total_X, X))
          total_wt = np.vstack((total_wt, wt))
          if return_y:
            total_Y = np.vstack((total_Y, Y))

      # Lower validation stats
      if self.lower_validation_stats is not None:
        if len(total_X) > self.lower_validation_stats:
          rng = np.random.RandomState(seed=42)
          random_indices = rng.choice(total_X.shape[0], self.lower_validation_stats, replace=False)
          total_X = total_X[random_indices,:]
          sum_wt = np.sum(total_wt)
          total_wt = total_wt[random_indices,:]
          new_sum_wt = np.sum(wt)
          total_wt *= sum_wt/new_sum_wt

    if not return_y:
      return total_X, total_wt
    else:
      return total_X, total_Y, total_wt

  def _TrimQuantile(self, column, wt, ignore_quantile=0.01, return_indices=False):
    """
    Trim the data and weights based on quantiles.

    Args:
        column (numpy.ndarray): The column data.
        wt (numpy.ndarray): The corresponding weights.
        ignore_quantile (float): Fraction of data to ignore from both ends during trimming (default is 0.01).

    Returns:
        trimmed_column (numpy.ndarray): Trimmed column data.
        trimmed_wt (numpy.ndarray): Trimmed weights.
    """
    lower_value = np.quantile(column, ignore_quantile)
    upper_value = np.quantile(column, 1-ignore_quantile)
    trimmed_indices = ((column >= lower_value) & (column <= upper_value))
    column = column[trimmed_indices]
    wt = wt[trimmed_indices].flatten()
    if not return_indices:
      return column, wt
    else: return column, wt, trimmed_indices

  def _CustomHistogram(self, data, weights=None, bins=20):
    """
    Compute a custom histogram for the given data.

    Args:
        data (numpy.ndarray): Input data.
        weights (numpy.ndarray, optional): Weights associated with the data. Defaults to None.
        bins (int or array_like, optional): If bins is an integer, it defines the number of equal-width
            bins in the range. If bins is an array, it defines the bin edges. Defaults to 20.

    Returns:
        numpy.ndarray: Histogram of data.
        numpy.ndarray: Bin edges.
    """
    unique_vals = pd.DataFrame(data).drop_duplicates()
    if isinstance(bins, int):
      if len(unique_vals) < bins:
        bins = np.sort(unique_vals.to_numpy().flatten())
        bins = np.append(bins, [(2*bins[-1]) - bins[-2]])
    hist, bins = np.histogram(data, weights=weights, bins=bins)
    return hist, bins

  def _MakeUnrolledBins(self, data, bins=5, sf_diff=2):
    """
    Make unrolled bins for the given data.

    Args:
        data (numpy.ndarray): Input data.
        bins (int, optional): Number of bins. Defaults to 5.
        sf_diff (int, optional): Significant figure difference. Defaults to 2.

    Returns:
        list: Unrolled bins.
    """
    diff = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    significant_figures = sf_diff - int(np.floor(np.log10(abs(diff)))) - 1
    rounded_number = round(diff, significant_figures)
    decimal_places = len(str(rounded_number).rstrip('0').split(".")[1])
    
    unique_vals = pd.DataFrame(data).drop_duplicates()

    # Discrete and less than bins
    if len(unique_vals) < bins:
      unrolled_bins = list(np.sort(unique_vals.to_numpy().flatten())[1:])
    # Discrete
    elif len(unique_vals) < 20:
      sorted_bins = np.sort(unique_vals.to_numpy().flatten())
      unrolled_bins = [round(np.quantile(data, i/bins), min(decimal_places,significant_figures)) for i in range(1,bins)]
      unrolled_bins = list(set(unrolled_bins))
      if unrolled_bins[0] == sorted_bins[0]:
        unrolled_bins = unrolled_bins[1:]
    # Continuous
    else:
      unrolled_bins = [round(np.quantile(data, i/bins), min(decimal_places,significant_figures)) for i in range(1,bins)]

    unrolled_bins = [-np.inf] + unrolled_bins + [np.inf]

    return unrolled_bins

  def Sample(self, row, columns=None, events_per_file=10**6, separate=False, transform=False, add_columns=False):
    """
    Sample synthetic data.

    Args:
        row (array-like): The row of data for which to sample synthetic data.
        columns (list): List of column names for sampling. If None, uses Y_columns from data_parameters.
        events_per_file (int): Number of events per file for sampling (default is 10**6).
        separate (bool): Whether to separate the sampled data by source (default is False).
        transform (bool): Whether to apply transformation to the sampled data (default is False).

    Returns:
        total_X (numpy.ndarray or dict): Sampled data.
        total_wt (numpy.ndarray or dict): Corresponding weights.
    """
    # Check if rows match even if empty
    if self.synth_row is None:
      matching_rows = False
    else:
      if len(self.synth_row) == 0 and len(row) == 0:
        matching_rows = True
      elif self.synth_row == row:
        matching_rows = True
      else:
        matching_rows = False


    if (self.synth is None) or (not matching_rows) or (self.synth_type == False and separate == True):
      first_loop = True
      for key, pdf in self.model.items():

        # Set up useful information
        if columns is None:
          columns = self.data_parameters[key]["Y_columns"]
        sel_row = np.array([row[columns.index(y)] for y in self.data_parameters[key]["Y_columns"]])
        wt = np.ones(events_per_file)/events_per_file

        # Sample through dataset
        X = pdf.Sample(sel_row, n_events=events_per_file, transform=transform)

        # Scale to yield
        if "yield" in self.data_parameters[key].keys():
          func = MakeYieldFunction(self.pois, self.nuisances, self.data_parameters[key])
          wt *= func(sel_row)

        # Scale by rate parameters
        if "mu_"+key in columns:
          rp = row[columns.index("mu_"+key)]
          if rp == 0.0: continue
          wt *= rp

        if add_columns:
          X = self.AddColumns(X)
        else:
          self.X_plot_columns = copy.deepcopy(self.X_columns)

        if not separate:
          # Concatenate datasets
          if first_loop:
            first_loop = False
            total_X = copy.deepcopy(X)
            total_wt = copy.deepcopy(wt.reshape(-1,1))
          else:
            total_X = np.vstack((total_X, X))
            total_wt = np.vstack((total_wt, wt.reshape(-1,1)))
        else:
          # Put datasets in a dictionary
          if first_loop:
            first_loop = False
            total_X = {key : X}
            total_wt = {key : wt.flatten()}
          else:
            total_X[key] = X
            total_wt[key] = wt.flatten() 

      
      self.synth_row = row
      self.synth = total_X
      if not separate: total_wt = total_wt.flatten()
      self.synth_wt = total_wt
      self.synth_type = separate

    # Combine saved dictionary
    elif (matching_rows and self.synth_type == True and separate == False):

      first_loop = True
      for key, val in self.synth.items():
        if first_loop:
          first_loop = False
          total_X = copy.deepcopy(val)
          total_wt = copy.deepcopy(self.synth_wt[key].reshape(-1,1))
        else:
          total_X = np.vstack((total_X, val))
          total_wt = np.vstack((total_wt, self.synth_wt[key].reshape(-1,1)))
      total_wt = total_wt.flatten()
      self.synth = total_X
      self.synth_wt = total_wt
      self.synth_type = separate

    # Load saved samples
    else:
      total_X = self.synth
      total_wt = self.synth_wt

    return total_X, total_wt

  def BuildLikelihood(self):
    """
    Build the likelihood object
    """
    if self.do_binned_fit:
      bin_yields_dict = {}
      for key, val in self.data_parameters.items():
        X, Y, wt = self._GetXAndWts(None, return_full=True, specific_file=key, ignore_infer=True, return_y=True)
        bins = [float(i) for i in self.var_and_bins.split("[")[1].split("]")[0].split(",")]
        column = self.X_columns.index(self.var_and_bins.split("[")[0])

        bin_yields, bin_edges = MakeBinYields(pd.DataFrame(X,columns=val["X_columns"]), pd.DataFrame(Y,columns=val["Y_columns"]), val, self.pois, self.nuisances, wt=pd.DataFrame(wt), column=column, bins=bins)
        bin_yields_dict[key] = copy.deepcopy(bin_yields)

      self.lkld = Likelihood(
        {
          "bin_yields": bin_yields_dict,
          "bin_edges": bin_edges,
        }, 
        type="binned_extended", 
        data_parameters=self.data_parameters,
        parameters=self.validation_options,
      )

    else:

      self.lkld = Likelihood(
        {
          "pdfs":self.model,
          "yields":{k:MakeYieldFunction(self.pois, self.nuisances, v) for k, v in self.data_parameters.items()}
        }, 
        type="unbinned_extended", 
        data_parameters=self.data_parameters,
        parameters=self.validation_options,
      )

  def DoDebug(self, row, columns=None):

    # get log probs
    X, wt = self._GetXAndWts(row, columns=columns)
    self.lkld.debug_mode = True
    loop = [171.5,172.0,172.5,173.0,173.5,174.0,174.5] # edit for your scenario
    for i in loop:
      self.lkld.Run(X, np.array([i]), wts=wt, return_ln=True)

    # plot histograms
    for key, val in self.lkld.debug_hists.items():
      plot_histograms(
        self.lkld.debug_bins[key][:-1],
        [v for _,v in val.items()],
        [k for k,_ in val.items()],
        x_label = key,
        name = f"debug/log_probs_hist_{key}", 
        y_label = r"$\sum \ln L$",
      )

    # subtract minimum
    sum_vals = []
    example_key = list(self.lkld.debug_hists.keys())[0]
    for key, val in self.lkld.debug_hists[example_key].items():
      sum_vals.append(np.sum(val))
    max_val = max(sum_vals)
    max_val_name = list(self.lkld.debug_hists[example_key].keys())[sum_vals.index(max_val)]
    for col, val_dict in self.lkld.debug_hists.items():
      max_hist = copy.deepcopy(self.lkld.debug_hists[col][max_val_name])
      for key, val in val_dict.items():
        self.lkld.debug_hists[col][key] = self.lkld.debug_hists[col][key] - max_hist

    # plot histograms
    for key, val in self.lkld.debug_hists.items():
      plot_histograms(
        self.lkld.debug_bins[key][:-1],
        [v for _,v in val.items()],
        [k for k,_ in val.items()],
        x_label = key,
        name = f"debug/log_probs_hist_{key}_ratio", 
        y_label = r"$\sum \ln L - (\sum \ln L)_{max}$",
      )

  def GetAndDumpBestFit(self, row, initial_guess, columns=None, ind=0):
    """
    Get and dump the best-fit parameters to a YAML file.

    Args:
        row (array-like): The row of data for which to find the best-fit parameters.
        initial_guess (array-like): Initial guess for the parameters during the best-fit search.
        columns (list): List of column names for best-fit search. If None, uses Y_columns from data_parameters.
        ind (int): Index for file naming (default is 0).
    """
    X, wt = self._GetXAndWts(row, columns=columns)
    self.lkld.GetAndWriteBestFitToYaml(X, row, initial_guess, wt=wt, filename=f"{self.out_dir}/best_fit_{ind}.yaml")

  def GetAndDumpScanRanges(self, row, col, columns=None, ind=0):
    """
    Get and dump scan ranges to a YAML file.

    Args:
        row (array-like): The row of data for which to find scan ranges.
        col (str): The column for which to perform the scan.
        columns (list): List of column names for scan. If None, uses Y_columns from data_parameters.
        ind (int): Index for file naming (default is 0).
    """
    X, wt = self._GetXAndWts(row, columns=columns)
    self.lkld.GetAndWriteScanRangesToYaml(X, row, col, wt=wt, filename=f"{self.out_dir}/scan_values_{col}_{ind}.yaml")

  def GetAndDumpScan(self, row, col, col_val, columns=None, ind1=0, ind2=0):
    """
    Get and dump negative log-likelihood to a YAML file.

    Args:
        row (array-like): The row of data for which to find negative log-likelihood.
        col (str): The column for which to compute the likelihood.
        col_val (float): The value of the column for which to compute the likelihood.
        columns (list): List of column names for likelihood computation. If None, uses Y_columns from data_parameters.
        ind1 (int): Index for file naming (default is 0).
        ind2 (int): Index for file naming (default is 0).
    """
    X, wt = self._GetXAndWts(row, columns=columns)
    self.lkld.GetAndWriteScanToYaml(X, row, col, col_val, wt=wt, filename=f"{self.out_dir}/scan_results_{col}_{ind1}_{ind2}.yaml")

  def PlotCorrelationMatrix(self, row, columns=None, extra_dir=""):
    """
    Plot correlation matrix.

    Args:
        row (array-like): The row of data for which to plot the correlation matrix.
        columns (list): List of column names for plotting. If None, uses Y_columns from data_parameters.
        extra_dir (str): Extra directory to save the plot (default is "").
    """
    print(">> Producing correlation matrix plots")
    X, wt = self._GetXAndWts(row, columns=columns)
    synth, synth_wt = self.Sample(row, columns=columns)

    true_cov_matrix = np.cov(X, aweights=wt.flatten(), rowvar=False)
    true_corr_matrix = true_cov_matrix / np.sqrt(np.outer(np.diag(true_cov_matrix), np.diag(true_cov_matrix)))
    
    synth_cov_matrix = np.cov(synth, aweights=synth_wt, rowvar=False)
    synth_corr_matrix = synth_cov_matrix / np.sqrt(np.outer(np.diag(synth_cov_matrix), np.diag(synth_cov_matrix)))

    file_extra_name = GetYName(row, purpose="file", prefix="_y_")
    plot_extra_name = GetYName(row, purpose="plot", prefix="y=")

    if extra_dir != "": 
      add_extra_dir = f"/{extra_dir}"
    else:
      add_extra_dir = ""

    plot_correlation_matrix(
      true_corr_matrix, 
      self.X_columns, 
      name=f"{self.plot_dir}{add_extra_dir}/correlation_matrix_true{file_extra_name}",
      title_right=plot_extra_name
    )

    plot_correlation_matrix(
      synth_corr_matrix, 
      self.X_columns, 
      name=f"{self.plot_dir}{add_extra_dir}/correlation_matrix_synth{file_extra_name}",
      title_right=plot_extra_name
    )

    plot_correlation_matrix(
      true_corr_matrix-synth_corr_matrix, 
      self.X_columns, 
      name=f"{self.plot_dir}{add_extra_dir}/correlation_matrix_subtracted{file_extra_name}",
      title_right=plot_extra_name
    )


  def PlotGeneration(self, row, columns=None, n_bins=40, ignore_quantile=0.01, sample_row=None, extra_dir="", transform=False):
    """
    Plot generation comparison between simulated and synthetic data for a given row.

    Args:
        row (list): List representing the unique row for comparison.
        columns (list): List of column names for plotting. If None, uses Y_columns from data_parameters.
        data_key (str): Key specifying the dataset (e.g., "val" for validation data).
        n_bins (int): Number of bins for histogram plotting.
        ignore_quantile (float): Fraction of data to ignore from both ends during histogram plotting.
    """
    print(">> Producing generation plots")
    if sample_row is None: sample_row = row

    sim_file_extra_name = GetYName(row, purpose="file", prefix="_sim_y_")
    sample_file_extra_name = GetYName(sample_row, purpose="file", prefix="_synth_y_")
    sim_plot_extra_name = GetYName(row, purpose="plot", prefix="y=")
    sample_plot_extra_name = GetYName(sample_row, purpose="plot", prefix="y=")

    X, wt = self._GetXAndWts(row, columns=columns, transform=transform, add_columns=(False if transform else True))
    synth, synth_wt = self.Sample(sample_row, columns=columns, separate=True, transform=transform, add_columns=(False if transform else True))
    synth_comb, synth_comb_wt = self.Sample(sample_row, columns=columns, separate=False, transform=transform, add_columns=(False if transform else True))

    for col in range(X.shape[1]):

      trimmed_X, trimmed_wt = self._TrimQuantile(X[:,col], wt, ignore_quantile=ignore_quantile)

      sim_hist, bins =  self._CustomHistogram(trimmed_X, weights=trimmed_wt, bins=n_bins)
      sim_hist_err_sq, _  = self._CustomHistogram(trimmed_X, weights=trimmed_wt**2, bins=bins)
      sim_hist_err = np.sqrt(sim_hist_err_sq)

      synth_hists = {}
      for key in synth.keys():
        synth_hist, _  = self._CustomHistogram(synth[key][:,col], weights=synth_wt[key], bins=bins)
        synth_hists[f"Synthetic {key} {sample_plot_extra_name}"] = synth_hist
      synth_hist_err_sq, _  = self._CustomHistogram(synth_comb[:,col], weights=synth_comb_wt**2, bins=bins)
      synth_hist_err = np.sqrt(synth_hist_err_sq)
      
      if extra_dir != "": 
        add_extra_dir = f"/{extra_dir}"
      else:
        add_extra_dir = ""

      plot_stacked_histogram_with_ratio(
        sim_hist, 
        synth_hists, 
        bins, 
        data_name=f'Simulated {sim_plot_extra_name}', 
        xlabel=self.X_plot_columns[col],
        ylabel="Events",
        name=f"{self.plot_dir}{add_extra_dir}/generation_{self.X_plot_columns[col]}{sim_file_extra_name}{sample_file_extra_name}", 
        data_errors=sim_hist_err, 
        stack_hist_errors=synth_hist_err, 
        title_right="",
        use_stat_err=False,
        axis_text="",
        )

  def Plot2DPulls(self, row, columns=None, n_bins=100, ignore_quantile=0.01, sample_row=None, extra_dir="", transform=False):
    print(">> Producing 2d pulls")

    if sample_row is None: sample_row = row
    X, wt = self._GetXAndWts(row, columns=columns, transform=transform, add_columns=(False if transform else True))
    synth, synth_wt = self.Sample(sample_row, columns=columns, separate=False, transform=transform, add_columns=(False if transform else True))

    sim_file_extra_name = GetYName(row, purpose="file", prefix="_sim_y_")
    sample_file_extra_name = GetYName(sample_row, purpose="file", prefix="_synth_y_")
    sim_plot_extra_name = GetYName(row, purpose="plot", prefix="y=")
    sample_plot_extra_name = GetYName(sample_row, purpose="plot", prefix="y=")

    for col_1 in range(X.shape[1]):
      for col_2 in range(synth.shape[1]):

        if col_1 >= col_2:
          continue

        _, plot_wt, inds = self._TrimQuantile(X[:,col_1], wt, ignore_quantile=ignore_quantile, return_indices=True)
        plot_X = X[inds, :]
        _, plot_wt, inds = self._TrimQuantile(plot_X[:,col_2], plot_wt, ignore_quantile=ignore_quantile, return_indices=True)
        plot_X = plot_X[inds, :]
      
        sim_hist, xedges, yedges = np.histogram2d(plot_X[:,col_1], plot_X[:,col_2], weights=plot_wt.flatten(), bins=n_bins)
        sim_hist_err_sq, _, _ = np.histogram2d(plot_X[:,col_1], plot_X[:,col_2], weights=plot_wt.flatten()**2, bins=[xedges, yedges])
        synth_hist, _, _ = np.histogram2d(synth[:,col_1], synth[:,col_2], weights=synth_wt.flatten(), bins=[xedges, yedges]) 
        synth_his_err_sq, _, _ = np.histogram2d(synth[:,col_1], synth[:,col_2], weights=synth_wt.flatten()**2, bins=[xedges, yedges]) 

        total_error = np.sqrt(sim_hist_err_sq + synth_his_err_sq)
        diff = np.abs(sim_hist-synth_hist)

        pull = np.full_like(diff, np.nan, dtype=float)
        mask = (total_error != 0)
        pull[mask] = np.divide(diff[mask], total_error[mask])

        if extra_dir != "": 
          add_extra_dir = f"/{extra_dir}"
        else:
          add_extra_dir = ""

        plot_heatmap(
          pull, 
          xedges, 
          yedges, 
          x_title=self.X_plot_columns[col_1], 
          y_title=self.X_plot_columns[col_2], 
          z_title=f"Simulated {sim_plot_extra_name} vs Synthetic {sample_plot_extra_name} Pulls", 
          name=f"{self.plot_dir}{add_extra_dir}/generation_pulls_2d_{self.X_plot_columns[col_1]}_{self.X_plot_columns[col_2]}{sim_file_extra_name}{sample_file_extra_name}", 
          )

  def Plot2DUnrolledGeneration(self, row, columns=None, n_unrolled_bins=5, n_bins=10, ignore_quantile=0.01, sf_diff=2, sample_row=None, extra_dir="", transform=False):
    """
    Plot unrolled generation comparison between simulated and synthetic data.

    Args:
        row (list): List representing the unique row for comparison.
        columns (list): List of column names for plotting. If None, uses Y_columns from data_parameters.
        n_bins (int): Number of bins for plotting histograms (default is 40).
        ignore_quantile (float): Fraction of data to ignore from both ends during plotting (default is 0.01).
        sample_row (list): List representing the unique row for synthetic data, if None will use random.
        extra_dir (str): Extra directory to save the plot (default is "").
        transform (bool): Whether to apply transformation to the data (default is False).
    """
    print(">> Producing 2d unrolled generation plots")

    if sample_row is None: sample_row = row
    X, wt = self._GetXAndWts(row, columns=columns, transform=transform, add_columns=(False if transform else True))
    synth, synth_wt = self.Sample(sample_row, columns=columns, separate=True, transform=transform, add_columns=(False if transform else True))
    synth_comb, synth_comb_wt = self.Sample(sample_row, columns=columns, separate=False, transform=transform, add_columns=(False if transform else True))

    sim_file_extra_name = GetYName(row, purpose="file", prefix="_sim_y_")
    sample_file_extra_name = GetYName(sample_row, purpose="file", prefix="_synth_y_")
    sim_plot_extra_name = GetYName(row, purpose="plot", prefix="y=")
    sample_plot_extra_name = GetYName(sample_row, purpose="plot", prefix="y=")

    unrolled_bins = []
    plot_bins = []
    for col in range(X.shape[1]):
      synth_column = synth_comb[:,col]
      nan_rows = np.isnan(synth_column.reshape(-1,1)).any(axis=1)
      synth_column = synth_column[~nan_rows]
      synth_comb_column_wt = synth_comb_wt[~nan_rows]

      # Find unrolled equal stat rounded bins
      unrolled_bins.append(self._MakeUnrolledBins(synth_column, sf_diff=sf_diff, bins=n_unrolled_bins))

      # Find equally spaced plotted bins after ignoring quantile
      lower_value = np.quantile(synth_column, ignore_quantile)
      upper_value = np.quantile(synth_column, 1-ignore_quantile)
      trimmed_indices = ((synth_column >= lower_value) & (synth_column <= upper_value))
      trimmed_X = synth_column[trimmed_indices]
      trimmed_wt = synth_comb_column_wt.flatten()[trimmed_indices]

      _, bins  = self._CustomHistogram(trimmed_X, weights=trimmed_wt, bins=n_bins)
      plot_bins.append(list(bins))

    for plot_col in range(X.shape[1]):
      for unrolled_col in range(X.shape[1]):
        if plot_col == unrolled_col: continue

        synth_hists_full = {}
        for key in synth.keys():

          synth_hists = []
          synth_err_hists = []
          X_hists = []
          X_err_hists = []
          for unrolled_bin_ind in range(len(unrolled_bins[unrolled_col])-1):

            unrolled_bin = [unrolled_bins[unrolled_col][unrolled_bin_ind], unrolled_bins[unrolled_col][unrolled_bin_ind+1]]

            synth_unrolled_bin_indices = ((synth[key][:,unrolled_col] >= unrolled_bin[0]) & (synth[key][:,unrolled_col] < unrolled_bin[1]))
            synth_unrolled_bin = synth[key][synth_unrolled_bin_indices]
            synth_unrolled_bin_wt = synth_wt[key][synth_unrolled_bin_indices]

            synth_hist, _  = self._CustomHistogram(synth_unrolled_bin[:,plot_col], weights=synth_unrolled_bin_wt, bins=plot_bins[plot_col])
            synth_hists.append(synth_hist)

          synth_hists_full[f"Synthetic {key} {sample_plot_extra_name}"] = synth_hists


        X_hists = []
        X_err_hists = []
        synth_err_hists = []
        for unrolled_bin_ind in range(len(unrolled_bins[unrolled_col])-1):

          unrolled_bin = [unrolled_bins[unrolled_col][unrolled_bin_ind], unrolled_bins[unrolled_col][unrolled_bin_ind+1]]

          synth_unrolled_bin_indices = ((synth_comb[:,unrolled_col] >= unrolled_bin[0]) & (synth_comb[:,unrolled_col] < unrolled_bin[1]))
          synth_unrolled_bin = synth_comb[synth_unrolled_bin_indices]
          synth_unrolled_bin_wt = synth_comb_wt[synth_unrolled_bin_indices]

          synth_hist_err_sq, _  = self._CustomHistogram(synth_unrolled_bin[:,plot_col], weights=synth_unrolled_bin_wt**2, bins=plot_bins[plot_col])
          synth_hist_err = np.sqrt(synth_hist_err_sq)
          synth_err_hists.append(synth_hist_err)

          X_unrolled_bin_indices = ((X[:,unrolled_col] >= unrolled_bin[0]) & (X[:,unrolled_col] < unrolled_bin[1]))
          X_unrolled_bin = X[X_unrolled_bin_indices]
          wt_unrolled_bin = wt[X_unrolled_bin_indices].flatten()
          X_hist, _  = self._CustomHistogram(X_unrolled_bin[:,plot_col], weights=wt_unrolled_bin, bins=plot_bins[plot_col])
          X_hist_err_sq, _  = self._CustomHistogram(X_unrolled_bin[:,plot_col], weights=wt_unrolled_bin**2, bins=plot_bins[plot_col])
          X_hist_err = np.sqrt(X_hist_err_sq)

          X_hists.append(X_hist)
          X_err_hists.append(X_hist_err)

        if extra_dir != "": 
          add_extra_dir = f"/{extra_dir}"
        else:
          add_extra_dir = ""

        plot_stacked_unrolled_2d_histogram_with_ratio(
          X_hists, 
          synth_hists_full, 
          plot_bins[plot_col],
          unrolled_bins[unrolled_col],
          self.X_plot_columns[unrolled_col],
          data_name=f'Simulated {sim_plot_extra_name}', 
          xlabel=self.X_plot_columns[plot_col],
          name=f"{self.plot_dir}{add_extra_dir}/generation_unrolled_2d_{self.X_plot_columns[plot_col]}_{self.X_plot_columns[unrolled_col]}{sim_file_extra_name}{sample_file_extra_name}", 
          data_hists_errors=X_err_hists, 
          stack_hists_errors=synth_err_hists, 
          title_right="",
          use_stat_err=False,
          ylabel = "Events",
        )
  
  def PlotLikelihood(self, x, y, row, col, crossings, best_fit, true_pdf=None, columns=None, extra_dir=""):
    """
    Plot likelihood scans.

    Args:
        x (array-like): Values of the parameter for which to plot the scan.
        y (array-like): Negative log-likelihood values corresponding to the parameter scan.
        row (array-like): The row of data for which the scan is performed.
        col (str): The column for which the scan is performed.
        crossings (list): List of crossing points in the scan.
        best_fit (array-like): Best-fit parameters obtained from the scan.
        true_pdf (function): True probability density function used for comparison plots (default is None).
        columns (list): List of column names for likelihood computation. If None, uses Y_columns from data_parameters.
    """
    print(">> Producing likelihood scans")
    if 0 in crossings.keys() and 1 in crossings.keys() and -1 in crossings.keys():
      print(f"{col}: {crossings[0]} + {crossings[1]-crossings[0]} - {crossings[0]-crossings[-1]}")
    if columns == None: columns = self.data_parameters["Y_columns"]
    ind = columns.index(col)
    file_extra_name = GetYName(row, purpose="file")
    plot_extra_name = GetYName(row, purpose="plot")

    other_lkld = {}

    X, wt = self._GetXAndWts(row, columns=columns)
    eff_events = float((np.sum(wt.flatten())**2)/np.sum(wt.flatten()**2))
    total_weight = float(np.sum(wt.flatten()))
    if not eff_events == total_weight:
      other_lkld[r"Inferred N=$N_{eff}$"] = (eff_events/total_weight)*np.array(y)

    # make true likelihood
    if true_pdf is not None:

      def Probability(X, Y, y_columns=None, k=None, **kwargs):
        Y = np.array(Y)
        if y_columns is not None:
          column_indices = [y_columns.index(col) for col in self.data_parameters[k]["Y_columns"]]
          Y = Y[:,column_indices]
        if len(Y) == 1: Y = np.tile(Y, (len(X), 1))
        prob = np.zeros(len(X))
        for i in range(len(X)):
          prob[i] = true_pdf[k](X[i],Y[i])
        return np.log(prob)

      for k in self.lkld.models["pdfs"].keys():
        self.lkld.models["pdfs"][k].Probability = partial(Probability, k=k)

      nlls = []
      for x_val in x:
        test_row = copy.deepcopy(row)
        test_row[ind] = x_val
        nlls.append(-2*self.lkld.Run(X, test_row, wts=wt, return_ln=True))

      min_nll = min(nlls)
      nlls = [nll - min_nll for nll in nlls]
      other_lkld["True"] = nlls

    if extra_dir != "": 
      add_extra_dir = f"/{extra_dir}"
    else:
      add_extra_dir = ""

    plot_likelihood(
      x, 
      y, 
      crossings, 
      name=f"{self.plot_dir}{add_extra_dir}/likelihood_{col}_y_{file_extra_name}", 
      xlabel=col, 
      true_value=row[ind],
      title_right=f"y={plot_extra_name}" if plot_extra_name != "" else "",
      cap_at=9,
      label="Inferred N="+str(int(round(total_weight))),
      other_lklds=other_lkld,
    )

  def PlotComparisons(self, row, best_fit, ignore_quantile=0.001, n_bins=40, columns=None, extra_dir=""):
    """
    Plot comparisons between data, true, and best-fit distributions.

    Args:
        row (array-like): The row of data for which to plot the comparisons.
        best_fit (array-like): Best-fit parameters used for synthetic data generation.
        ignore_quantile (float): Fraction of data to ignore from both ends during histogram plotting (default is 0.001).
        n_bins (int): Number of bins for histogram plotting (default is 40).
        columns (list): List of column names for plotting. If None, uses Y_columns from data_parameters.
    """
    print(">> Producing comparison plots")
    if columns == None: columns = self.data_parameters["Y_columns"]
    X, wt = self._GetXAndWts(row, columns=columns, add_columns=True)
    synth_true, synth_true_wt = self.Sample(row, columns=columns, separate=False, add_columns=True)
    synth_best_fit, synth_best_fit_wt = self.Sample(best_fit, columns=columns, separate=False, add_columns=True)

    for col in range(X.shape[1]):

      trimmed_X, trimmed_wt = self._TrimQuantile(X[:,col], wt, ignore_quantile=ignore_quantile)

      sim_hist, bins  = np.histogram(trimmed_X, weights=trimmed_wt,bins=n_bins)
      sim_hist_err_sq, _  = np.histogram(trimmed_X, weights=trimmed_wt**2, bins=bins)
      synth_true_hist, _  = np.histogram(synth_true[:,col], weights=synth_true_wt, bins=bins)
      synth_best_fit_hist, _  = np.histogram(synth_best_fit[:,col], weights=synth_best_fit_wt, bins=bins)

      file_extra_name = GetYName(row, purpose="file")
      plot_extra_name_true = GetYName(row, purpose="plot")
      plot_extra_name_bf = GetYName(best_fit, purpose="plot")

      if extra_dir != "": 
        add_extra_dir = f"/{extra_dir}"
      else:
        add_extra_dir = ""

      plot_histograms(
        bins[:-1],
        [synth_best_fit_hist,synth_true_hist],
        [f"Synthetic y={plot_extra_name_bf} (Best Fit)",f"Synthetic y={plot_extra_name_true} (True)"],
        colors = ["blue","red",],
        linestyles = ["-","-"],
        title_right = f"y={plot_extra_name_true}",
        x_label=self.X_plot_columns[col],
        name=f"{self.plot_dir}{add_extra_dir}/comparison_{self.X_plot_columns[col]}_y_{file_extra_name}", 
        y_label = "Events",
        error_bar_hists = [sim_hist],
        error_bar_hist_errs = [np.sqrt(sim_hist_err_sq)],
        error_bar_names = [f"Simulated y={plot_extra_name_true}"],
      )

  def DrawProbability(self, y_vals, n_bins=100, ignore_quantile=0.001, extra_dir=""):
    """
    Draw probability distributions for given Y values.

    Args:
        y_vals (list): List of Y values for which to draw probability distributions.
        n_bins (int): Number of bins for the histogram plots (default is 100).
        ignore_quantile (float): Fraction of data to ignore from both ends during histogram plotting (default is 0.001).
    """
    if len(self.data_parameters["X_columns"]) != 1:
      print("DrawProbability only works for X dimension of 1.")
      return None
    
    # Sample to get ranges
    first_loop = True
    for y_val in y_vals:
      if first_loop:
        synth = self.model.Sample(np.array([y_val]), n_events=10**4)
        first_loop = False
      else:
        synth = np.vstack((synth, self.model.Sample(np.array([y_val]), n_events=10**4)))

    synth = synth.flatten()
    lower_value = np.quantile(synth, ignore_quantile)
    upper_value = np.quantile(synth, 1-ignore_quantile)
    trimmed_indices = ((synth >= lower_value) & (synth <= upper_value))
    synth = synth[trimmed_indices]

    _, bins = np.histogram(synth, bins=n_bins)

    # Calculate probabilities
    hists = []
    for y_val in y_vals:

      probs = self.model.Probability(np.array(bins[:-1]).reshape(-1, 1), np.array([y_val]), change_zero_prob=False)
      integral = simpson(probs, dx=bins[1]-bins[0])
      print(f"Integral for Y is {integral}")
      hists.append(probs)

    if extra_dir != "": 
      add_extra_dir = f"/{extra_dir}"
    else:
      add_extra_dir = ""

    plot_histograms(
      bins[:-1],
      hists,
      [f"y={y_val}" for y_val in y_vals],
      title_right = "",
      x_label=self.data_parameters["X_columns"][0],
      name=f"{self.plot_dir}{add_extra_dir}/probability", 
      y_label = "p(x|y)",
      anchor_y_at_0 = True,
    )

  def PlotBinned(self, row, columns=None, sample_row=None, extra_dir=""):
    """
    Plot binned generation comparison between simulated and synthetic data.

    Args:
        row (list): List representing the unique row for comparison.
        columns (list): List of column names for plotting. If None, uses Y_columns from data_parameters.
        n_bins (int): Number of bins for plotting histograms (default is 40).
        ignore_quantile (float): Fraction of data to ignore from both ends during plotting (default is 0.01).
        sample_row (list): List representing the unique row for synthetic data, if None will use random.
        extra_dir (str): Extra directory to save the plot (default is "").
        transform (bool): Whether to apply transformation to the data (default is False).
    """
    sim_file_extra_name = GetYName(row, purpose="file", prefix="_sim_y_")
    sample_file_extra_name = GetYName(sample_row, purpose="file", prefix="_synth_y_")
    sim_plot_extra_name = GetYName(row, purpose="plot", prefix="y=")
    sample_plot_extra_name = GetYName(sample_row, purpose="plot", prefix="y=")

    if sample_row is None: sample_row = row

    bins = [float(i) for i in self.var_and_bins.split("[")[1].split("]")[0].split(",")]
    column = self.X_columns.index(self.var_and_bins.split("[")[0])
    col_name = self.var_and_bins.split("[")[0]

    X, wt = self._GetXAndWts(row, columns=columns)
    data_hist, _ = np.histogram(X[:,column], weights=wt.flatten(), bins=bins)
    data_hist_err_sq, _ = np.histogram(X[:,column], weights=wt.flatten()**2, bins=bins)
    data_hist_err = np.sqrt(data_hist_err_sq)
    
    # Make estimators and hists
    procs = {}
    proc_err_sq = np.zeros(len(bins)-1)
    for key, val in self.data_parameters.items():
      sample_row = np.array(sample_row)
      if columns is not None:
        column_indices = [columns.index(col) for col in val["Y_columns"]]
        sample_row_for_yield = sample_row[column_indices]

      X, Y, wt = self._GetXAndWts(None, return_full=True, specific_file=key, ignore_infer=True, return_y=True)

      rp = 1.0
      if "mu_"+key in columns:
        rp = sample_row[columns.index("mu_"+key)]
        if rp == 0.0: continue

      bin_yields, _ = MakeBinYields(pd.DataFrame(X,columns=val["X_columns"]), pd.DataFrame(Y,columns=val["Y_columns"]), val, self.pois, self.nuisances, wt=pd.DataFrame(wt), column=column, bins=bins, inf_edges=False)
      procs[f"{key} {sample_plot_extra_name}"] = np.array([rp * bin_yields[i](sample_row_for_yield) for i in range(len(bin_yields))])

      bin_yields, _ = MakeBinYields(pd.DataFrame(X,columns=val["X_columns"]), pd.DataFrame(Y,columns=val["Y_columns"]), val, self.pois, self.nuisances, wt=pd.DataFrame(wt), column=column, bins=bins, do_err=True, inf_edges=False)
      proc_err_sq += np.array([(rp * bin_yields[i](sample_row_for_yield))**2 for i in range(len(bin_yields))])

    proc_err = np.sqrt(proc_err_sq)

    if extra_dir != "": 
      add_extra_dir = f"/{extra_dir}"
    else:
      add_extra_dir = ""

    plot_stacked_histogram_with_ratio(
      data_hist, 
      procs, 
      bins, 
      data_name=f'Data' if self.infer is not None else f"Simulated {sim_plot_extra_name}", 
      xlabel=col_name,
      ylabel="Events",
      name=f"{self.plot_dir}{add_extra_dir}/binned_{col_name}{sim_file_extra_name}{sample_file_extra_name}", 
      data_errors=data_hist_err, 
      stack_hist_errors=proc_err, 
      title_right="",
      use_stat_err=False,
      axis_text="",
      )

  def PlotValidationSummary(self, results, true_pdf=None, extra_dir=""):
    """
    Plot a validation summary.

    Args:
        results (dict): Dictionary containing the validation results.
        true_pdf (dict): Dictionary containing the true PDFs (default is None).
        extra_dir (str): Extra directory to save the plot (default is "").
    """
    plot_dict = {}
    other_summaries = {}
    if true_pdf is not None:
      other_summaries["True PDF"] = {}

    for col, col_vals in results.items():
      plot_dict[col] = {}
      if true_pdf is not None:
        other_summaries["True PDF"][col] = {}      

      for _, info in col_vals.items():
        norm_crossings = {k: v/info["row"][info["columns"].index(info["varied_column"])] for k,v in info['crossings'].items()}
        plot_dict[col][GetYName(info["row"],purpose="plot",prefix="y=")] = norm_crossings
    
        # make true likelihood
        if true_pdf is not None:
          def Probability(X, Y, y_columns=None, k=None, **kwargs):
            Y = np.array(Y)
            if y_columns is not None:
              column_indices = [y_columns.index(col) for col in self.data_parameters[k]["Y_columns"]]
              Y = Y[:,column_indices]
            if len(Y) == 1: Y = np.tile(Y, (len(X), 1))
            prob = np.zeros(len(X))
            for i in range(len(X)):
              prob[i] = true_pdf[k](X[i],Y[i])
            return np.log(prob)
          
          for k in self.lkld.models["pdfs"].keys():
            self.lkld.models["pdfs"][k].Probability = partial(Probability, k=k)

          X, wt = self._GetXAndWts(info["row"], columns=info["columns"])
          nlls = []
          for x_val in info["scan_values"]:
            test_row = copy.deepcopy(info["row"])
            test_row[info["columns"].index(info["varied_column"])] = x_val
            nlls.append(-2*self.lkld.Run(X, test_row, wts=wt, return_ln=True))

          min_nll = min(nlls)
          nlls = [nll - min_nll for nll in nlls]

          true_crossings = self.lkld.FindCrossings(info["scan_values"], nlls, crossings=[1, 2])
          true_norm_crossings = {k: v/info["row"][info["columns"].index(info["varied_column"])] for k,v in true_crossings.items()}
          other_summaries["True PDF"][col][GetYName(info["row"],purpose="plot",prefix="y=")] = true_norm_crossings

    if extra_dir != "": 
      add_extra_dir = f"/{extra_dir}"
    else:
      add_extra_dir = ""

    plot_validation_summary(
      plot_dict, 
      name=f"{self.plot_dir}{add_extra_dir}/validation_summary",
      nominal_name="Learned",
      other_summaries=other_summaries,
      )
    
  def AddColumns(self, dataset):
    df = pd.DataFrame(dataset, columns=self.X_columns)
    for k, v in self.calculate_columns_for_plotting.items():
      df.eval(f"{k} = {v}", inplace=True)
    self.X_plot_columns = list(df.columns)
    dataset = df.to_numpy()
    return dataset