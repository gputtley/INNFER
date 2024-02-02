from preprocess import PreProcess
from likelihood import Likelihood
from data_loader import DataLoader
from plotting import plot_histograms, plot_histogram_with_ratio, plot_likelihood, plot_correlation_matrix, plot_stacked_histogram_with_ratio, plot_stacked_unrolled_2d_histogram_with_ratio
from other_functions import GetYName, MakeYieldFunction, MakeBinYields
from scipy.integrate import simpson
from functools import partial
import numpy as np
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
    self._SetOptions(options)
    self.X_columns = self.data_parameters[list(self.data_parameters.keys())[0]]["X_columns"]

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

  def _GetXAndWts(self, row, columns=None, use_nominal_wt=False):
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
    if self.infer is not None:
        
        dl = DataLoader(self.infer)
        data = dl.LoadFullDataset()
        total_X = data.loc[:,self.X_columns].to_numpy()
        total_wt = data.loc[:,"wt"].to_numpy().reshape(-1,1)

    # Do inference on simulated data
    else:

      first_loop = True
      for key, val in self.data_parameters.items():

        # Set up preprocess code to load the data in
        pp = PreProcess()
        pp.parameters = val
        pp.output_dir = val["file_location"]

        # Load and reformat data
        X, Y, wt = pp.LoadSplitData(dataset=self.data_key, get=["X","Y","wt"], use_nominal_wt=use_nominal_wt)
        if columns is None:
          columns = val["Y_columns"]
        X = X.to_numpy()
        Y = Y.to_numpy()
        wt = wt.to_numpy()

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

        # Concatenate datasets
        if first_loop:
          first_loop = False
          total_X = copy.deepcopy(X)
          total_wt = copy.deepcopy(wt)
        else:
          total_X = np.vstack((total_X, X))
          total_wt = np.vstack((total_wt, wt))

      # Lower validation stats
      if self.lower_validation_stats is not None:
        if len(total_X) > self.lower_validation_stats:
          random_indices = np.random.choice(total_X.shape[0], self.lower_validation_stats, replace=False)
          total_X = total_X[random_indices,:]
          sum_wt = np.sum(total_wt)
          total_wt = total_wt[random_indices,:]
          new_sum_wt = np.sum(wt)
          total_wt *= sum_wt/new_sum_wt

    return total_X, total_wt

  def _TrimQuantile(self, column, wt, ignore_quantile=0.01):

    lower_value = np.quantile(column, ignore_quantile)
    upper_value = np.quantile(column, 1-ignore_quantile)
    trimmed_indices = ((column >= lower_value) & (column <= upper_value))
    column = column[trimmed_indices]
    wt = wt[trimmed_indices].flatten()
    return column, wt

  def Sample(self, row, columns=None, events_per_file=10**6, separate=False):

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
        X = pdf.Sample(sel_row, n_events=events_per_file)

        # Scale to yield
        if "yield" in self.data_parameters[key].keys():
          func = MakeYieldFunction(self.pois, self.nuisances, self.data_parameters[key])
          wt *= func(sel_row)

        # Scale by rate parameters
        if "mu_"+key in columns:
          rp = row[columns.index("mu_"+key)]
          if rp == 0.0: continue
          wt *= rp

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
      X, Y, wt = self.pp.LoadSplitData(dataset="full", get=["X","Y","wt"], use_nominal_wt=False)
      if self.var_and_bins is not None:
        bins = [float(i) for i in self.var_and_bins.split("[")[1].split("]")[0].split(",")]
        column = self.data_parameters["X_columns"].index(self.var_and_bins.split("[")[0])
      else:
        bins = None
        column = 0
      bin_yields, bin_edges = MakeBinYields(X, Y, self.data_parameters, self.pois, self.nuisances, wt=wt, column=column, bins=bins)


      self.lkld = Likelihood(
        {
          "bin_yields": {self.model_name:bin_yields},
          "bin_edges": bin_edges,
        }, 
        type="binned_extended", 
        data_parameters=self.data_parameters,

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
    print(X)
    print(wt)
    #loop = [166.5,169.5,171.5,172.5,173.5,175.5,178.5]
    #for i in loop:
    #  print(np.array([i]), len(X), float(np.sum(wt)))
    #  self.lkld.Run(X, np.array([i]), wts=wt, return_ln=True)
    #exit()
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


  def PlotGeneration(self, row, columns=None, n_bins=40, ignore_quantile=0.01, sample_row=None, extra_dir=""):
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

    X, wt = self._GetXAndWts(row, columns=columns)
    synth, synth_wt = self.Sample(sample_row, columns=columns, separate=True)
    synth_comb, synth_comb_wt = self.Sample(sample_row, columns=columns, separate=False)

    for col in range(X.shape[1]):

      trimmed_X, trimmed_wt = self._TrimQuantile(X[:,col], wt, ignore_quantile=ignore_quantile)

      sim_hist, bins  = np.histogram(trimmed_X, weights=trimmed_wt,bins=n_bins)
      sim_hist_err_sq, _  = np.histogram(trimmed_X, weights=trimmed_wt**2, bins=bins)
      sim_hist_err = np.sqrt(sim_hist_err_sq)

      synth_hists = {}
      for key in synth.keys():
        synth_hist, _  = np.histogram(synth[key][:,col], weights=synth_wt[key], bins=bins)
        synth_hists[f"Synthetic {key} {sample_plot_extra_name}"] = synth_hist
      synth_hist_err_sq, _  = np.histogram(synth_comb[:,col], weights=synth_comb_wt**2, bins=bins)
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
        xlabel=self.X_columns[col],
        ylabel="Events",
        name=f"{self.plot_dir}{add_extra_dir}/generation_{self.X_columns[col]}{sim_file_extra_name}{sample_file_extra_name}", 
        data_errors=sim_hist_err, 
        stack_hist_errors=synth_hist_err, 
        title_right="",
        use_stat_err=False,
        axis_text="",
        )

  def Plot2DUnrolledGeneration(self, row, columns=None, n_unrolled_bins=5, n_bins=10, ignore_quantile=0.01, sf_diff=2, sample_row=None, extra_dir=""):
    print(">> Producing 2d unrolled generation plots")

    if sample_row is None: sample_row = row
    X, wt = self._GetXAndWts(row, columns=columns)
    synth, synth_wt = self.Sample(sample_row, columns=columns, separate=True)
    synth_comb, synth_comb_wt = self.Sample(sample_row, columns=columns, separate=False)

    sim_file_extra_name = GetYName(row, purpose="file", prefix="_sim_y_")
    sample_file_extra_name = GetYName(sample_row, purpose="file", prefix="_synth_y_")
    sim_plot_extra_name = GetYName(row, purpose="plot", prefix="y=")
    sample_plot_extra_name = GetYName(sample_row, purpose="plot", prefix="y=")

    unrolled_bins = []
    plot_bins = []
    for col in range(X.shape[1]):
      synth_column = synth_comb[:,col]

      # Find unrolled equal stat rounded bins
      diff = np.quantile(synth_column, 0.75) - np.quantile(synth_column, 0.25)
      significant_figures = sf_diff - int(np.floor(np.log10(abs(diff)))) - 1
      rounded_number = round(diff, significant_figures)
      decimal_places = len(str(rounded_number).rstrip('0').split(".")[1])
      unrolled_bins.append([round(np.quantile(synth_column, i/n_unrolled_bins), min(decimal_places,significant_figures)) for i in range(1,n_unrolled_bins)])
      unrolled_bins[col] = [-np.inf] + unrolled_bins[col] + [np.inf]

      # Find equally spaced plotted bins after ignoring quantile
      lower_value = np.quantile(synth_column, ignore_quantile)
      upper_value = np.quantile(synth_column, 1-ignore_quantile)
      trimmed_indices = ((synth_column >= lower_value) & (synth_column <= upper_value))
      trimmed_X = synth_column[trimmed_indices]
      _, bins  = np.histogram(trimmed_X, bins=n_bins)
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
          for unrolled_bin_ind in range(n_unrolled_bins):
            unrolled_bin = [unrolled_bins[unrolled_col][unrolled_bin_ind], unrolled_bins[unrolled_col][unrolled_bin_ind+1]]

            synth_unrolled_bin_indices = ((synth[key][:,unrolled_col] >= unrolled_bin[0]) & (synth[key][:,unrolled_col] < unrolled_bin[1]))
            synth_unrolled_bin = synth[key][synth_unrolled_bin_indices]
            synth_unrolled_bin_wt = synth_wt[key][synth_unrolled_bin_indices]

            synth_hist, _  = np.histogram(synth_unrolled_bin[:,plot_col], weights=synth_unrolled_bin_wt, bins=plot_bins[plot_col])
            synth_hists.append(synth_hist)

          synth_hists_full[f"Synthetic {key} {sample_plot_extra_name}"] = synth_hists


        X_hists = []
        X_err_hists = []
        synth_err_hists = []
        for unrolled_bin_ind in range(n_unrolled_bins):

          unrolled_bin = [unrolled_bins[unrolled_col][unrolled_bin_ind], unrolled_bins[unrolled_col][unrolled_bin_ind+1]]

          synth_unrolled_bin_indices = ((synth_comb[:,unrolled_col] >= unrolled_bin[0]) & (synth_comb[:,unrolled_col] < unrolled_bin[1]))
          synth_unrolled_bin = synth_comb[synth_unrolled_bin_indices]
          synth_unrolled_bin_wt = synth_comb_wt[synth_unrolled_bin_indices]

          synth_hist_err_sq, _  = np.histogram(synth_unrolled_bin[:,plot_col], weights=synth_unrolled_bin_wt**2, bins=plot_bins[plot_col])
          synth_hist_err = np.sqrt(synth_hist_err_sq)
          synth_err_hists.append(synth_hist_err)

          X_unrolled_bin_indices = ((X[:,unrolled_col] >= unrolled_bin[0]) & (X[:,unrolled_col] < unrolled_bin[1]))
          X_unrolled_bin = X[X_unrolled_bin_indices]
          wt_unrolled_bin = wt[X_unrolled_bin_indices].flatten()
          X_hist, _  = np.histogram(X_unrolled_bin[:,plot_col], weights=wt_unrolled_bin, bins=plot_bins[plot_col])
          X_hist_err_sq, _  = np.histogram(X_unrolled_bin[:,plot_col], weights=wt_unrolled_bin**2, bins=bins)
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
          self.X_columns[unrolled_col],
          data_name=f'Simulated {sim_plot_extra_name}', 
          xlabel=self.X_columns[plot_col],
          name=f"{self.plot_dir}{add_extra_dir}/generation_unrolled_2d_{self.X_columns[plot_col]}_{self.X_columns[unrolled_col]}{sim_file_extra_name}{sample_file_extra_name}", 
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
        test_row = copy.deepcopy(best_fit)
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
    X, wt = self._GetXAndWts(row, columns=columns)
    synth_true, synth_true_wt = self.Sample(row, columns=columns, separate=False)
    synth_best_fit, synth_best_fit_wt = self.Sample(best_fit, columns=columns, separate=False)

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
        x_label=self.X_columns[col],
        name=f"{self.plot_dir}{add_extra_dir}/comparison_{self.X_columns[col]}_y_{file_extra_name}", 
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