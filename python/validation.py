from preprocess import PreProcess
from likelihood import Likelihood
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
    self.model = model
    self.model_name = "model"
    self.data_parameters = {}
    self.lkld = None
    self.pois = []
    self.nuisances = []
    self.data_key = "val"
    self.tolerance = 1e-6
    self.lower_validation_stats = None
    self.do_binned_fit = None
    self.var_and_bins = None
    self.data_dir = "./data/"
    self.out_dir = "./data/"
    self.plot_dir = "./plots/"
    self._SetOptions(options)

    self.pp = PreProcess()
    self.pp.parameters = self.data_parameters
    self.pp.output_dir = self.data_dir

    self.synth = None
    self.synth_row = None

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
    X, Y, wt = self.pp.LoadSplitData(dataset=self.data_key, get=["X","Y","wt"], use_nominal_wt=use_nominal_wt)
    if columns is None:
      columns = self.data_parameters["Y_columns"]
    X = X.to_numpy()
    Y = Y.to_numpy()
    wt = wt.to_numpy()
    if Y.shape[1] > 0:
      row = np.array(list(row))
      matching_rows = np.all(np.isclose(Y, row, rtol=self.tolerance, atol=self.tolerance), axis=1)
      X = X[matching_rows]
      wt = wt[matching_rows]

    if "yield" in self.data_parameters.keys():
      if "all" in self.data_parameters["yield"].keys():
        sum_wt = self.data_parameters["yield"]["all"]
      else:
        sum_wt = self.data_parameters["yield"][GetYName(row,purpose="file")]
      old_sum_wt = np.sum(wt, dtype=np.float128)
      wt *= sum_wt/old_sum_wt

    if self.lower_validation_stats is not None:
      if len(X) > self.lower_validation_stats:
        random_indices = np.random.choice(X.shape[0], self.lower_validation_stats, replace=False)
        X = X[random_indices,:]
        sum_wt = np.sum(wt)
        wt = wt[random_indices,:]
        new_sum_wt = np.sum(wt)
        wt *= sum_wt/new_sum_wt

    return X, wt

  def BuildLikelihood(self):
    """
    Build the likelihood object
    """
    if self.do_binned_fit:
      X, Y, wt = self.pp.LoadSplitData(dataset="val", get=["X","Y","wt"], use_nominal_wt=False)
      if self.var_and_bins is not None:
        bins = [float(i) for i in self.var_and_bins.split("[")[1].split("]")[0].split(",")]
        column = self.data_parameters["X_columns"].index(self.var_and_bins.split("[")[0])
      else:
        bins = 100
        column = 0
      bin_yields, bin_edges = MakeBinYields(X, Y, self.data_parameters, self.pois, self.nuisances, wt=wt, column=column, bins=bins)
      self.lkld = Likelihood(
        {
          "bin_yields": {self.model_name:bin_yields},
          "bin_edges": bin_edges,
        }, 
        type="binned_extended", 
        data_parameters={
          self.model_name:{
            "X_columns" : self.data_parameters["X_columns"],
            "Y_columns" : self.data_parameters["Y_columns"],
          },
        }
      )

    else:

      self.lkld = Likelihood(
        {
          "pdfs":{self.model_name:self.model},
          "yields":{self.model_name:MakeYieldFunction(self.pois, self.nuisances, self.data_parameters)}
        }, 
        type="unbinned_extended", 
        data_parameters={
          self.model_name:{
            "X_columns" : self.data_parameters["X_columns"],
            "Y_columns" : self.data_parameters["Y_columns"],
          },
        }
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
    if columns == None: columns = self.data_parameters["Y_columns"]
    X, wt = self._GetXAndWts(row, columns=columns)
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
    if columns == None: columns = self.data_parameters["Y_columns"]
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
    if columns == None: columns = self.data_parameters["Y_columns"]
    X, wt = self._GetXAndWts(row, columns=columns)
    self.lkld.GetAndWriteScanToYaml(X, row, col, col_val, wt=wt, filename=f"{self.out_dir}/scan_results_{col}_{ind1}_{ind2}.yaml")

  def PlotCorrelationMatrix(self, row, columns=None):

    print(">> Producing correlation matrix plots")
    X, wt = self._GetXAndWts(row, columns=columns)
    if self.synth is None or self.synth_row != row:
      synth = self.model.Sample(np.array([list(row)]), columns=columns)
      self.synth_row = row
      self.synth = synth
    else:
      synth = self.synth

    true_cov_matrix = np.cov(X, aweights=wt.flatten(), rowvar=False)
    true_corr_matrix = true_cov_matrix / np.sqrt(np.outer(np.diag(true_cov_matrix), np.diag(true_cov_matrix)))
    
    synth_cov_matrix = np.cov(synth, rowvar=False)
    synth_corr_matrix = synth_cov_matrix / np.sqrt(np.outer(np.diag(synth_cov_matrix), np.diag(synth_cov_matrix)))

    file_extra_name = GetYName(row, purpose="file")
    plot_extra_name = GetYName(row, purpose="plot")

    plot_correlation_matrix(
      true_corr_matrix, 
      self.data_parameters["X_columns"], 
      name=f"{self.plot_dir}/correlation_matrix_true_y_{file_extra_name}",
      title_right=f"y={plot_extra_name}" if plot_extra_name != "" else ""
    )

    plot_correlation_matrix(
      synth_corr_matrix, 
      self.data_parameters["X_columns"], 
      name=f"{self.plot_dir}/correlation_matrix_synth_y_{file_extra_name}",
      title_right=f"y={plot_extra_name}" if plot_extra_name != "" else ""
    )

    plot_correlation_matrix(
      true_corr_matrix-synth_corr_matrix, 
      self.data_parameters["X_columns"], 
      name=f"{self.plot_dir}/correlation_matrix_subtracted_y_{file_extra_name}",
      title_right=f"y={plot_extra_name}" if plot_extra_name != "" else ""
    )


  def PlotGeneration(self, row, columns=None, n_bins=40, ignore_quantile=0.01):
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
    X, wt = self._GetXAndWts(row, columns=columns)
    if self.synth is None or self.synth_row != row:
      synth = self.model.Sample(np.array([list(row)]), columns=columns)
      self.synth_row = row
      self.synth = synth
    else:
      synth = self.synth
    
    if "yield" in self.data_parameters.keys():
      yf = MakeYieldFunction(self.pois, self.nuisances, self.data_parameters)
      synth_wt = (yf(row)/len(synth)) * np.ones(len(synth))
    else:
      synth_wt = (np.sum(wt)/len(synth)) * np.ones(len(synth))

    for col in range(X.shape[1]):

      trimmed_X = X[:,col]
      lower_value = np.quantile(trimmed_X, ignore_quantile)
      upper_value = np.quantile(trimmed_X, 1-ignore_quantile)
      trimmed_indices = ((trimmed_X >= lower_value) & (trimmed_X <= upper_value))
      trimmed_X = trimmed_X[trimmed_indices]
      trimmed_wt = wt[trimmed_indices].flatten()

      sim_hist, bins  = np.histogram(trimmed_X, weights=trimmed_wt,bins=n_bins)
      sim_hist_err_sq, _  = np.histogram(trimmed_X, weights=trimmed_wt**2, bins=bins)
      sim_hist_err = np.sqrt(sim_hist_err_sq)
      synth_hist, _  = np.histogram(synth[:,col], weights=synth_wt, bins=bins)
      synth_hist_err_sq, _  = np.histogram(synth[:,col], weights=synth_wt**2, bins=bins)
      synth_hist_err = np.sqrt(synth_hist_err_sq)

      file_extra_name = GetYName(row, purpose="file")
      plot_extra_name = GetYName(row, purpose="plot")
      
      plot_stacked_histogram_with_ratio(
        sim_hist, 
        {"Synthetic" : synth_hist}, 
        bins, 
        data_name='Simulated', 
        xlabel=self.data_parameters["X_columns"][col],
        ylabel="Events",
        name=f"{self.plot_dir}/generation_{self.data_parameters['X_columns'][col]}_y_{file_extra_name}", 
        data_errors=sim_hist_err, 
        stack_hist_errors=synth_hist_err, 
        title_right=f"y={plot_extra_name}",
        use_stat_err=False,
        axis_text="",
        )

  def Plot2DUnrolledGeneration(self, row, columns=None, n_unrolled_bins=5, n_bins=10, ignore_quantile=0.01, sf_diff=2):
    print(">> Producing 2d unrolled generation plots")
    X, wt = self._GetXAndWts(row, columns=columns)
    if self.synth is None or self.synth_row != row:
      synth = self.model.Sample(np.array([list(row)]), columns=columns)
      self.synth_row = row
      self.synth = synth
    else:
      synth = self.synth

    unrolled_bins = []
    plot_bins = []
    for col in range(X.shape[1]):
      synth_column = synth[:,col]

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
        synth_hists = []
        synth_err_hists = []
        X_hists = []
        X_err_hists = []
        for unrolled_bin_ind in range(n_unrolled_bins):
          unrolled_bin = [unrolled_bins[unrolled_col][unrolled_bin_ind], unrolled_bins[unrolled_col][unrolled_bin_ind+1]]

          synth_unrolled_bin_indices = ((synth[:,unrolled_col] >= unrolled_bin[0]) & (synth[:,unrolled_col] < unrolled_bin[1]))
          synth_unrolled_bin = synth[synth_unrolled_bin_indices]

          X_unrolled_bin_indices = ((X[:,unrolled_col] >= unrolled_bin[0]) & (X[:,unrolled_col] < unrolled_bin[1]))
          X_unrolled_bin = X[X_unrolled_bin_indices]
          wt_unrolled_bin = wt[X_unrolled_bin_indices].flatten()
          synth_hist, _  = np.histogram(synth_unrolled_bin[:,plot_col], bins=plot_bins[plot_col])
          synth_hist_err = np.sqrt(synth_hist)
          X_hist, _  = np.histogram(X_unrolled_bin[:,plot_col], weights=wt_unrolled_bin, bins=plot_bins[plot_col])
          X_hist_err_sq, _  = np.histogram(X_unrolled_bin[:,plot_col], weights=wt_unrolled_bin**2, bins=bins)
          X_hist_err = np.sqrt(X_hist_err_sq)

          sum_synth_hist = float(np.sum(synth_hist))
          sum_X_hist = float(np.sum(X_hist))
          synth_hist = synth_hist/sum_synth_hist
          synth_hist_err = synth_hist_err/sum_synth_hist
          X_hist = X_hist/sum_X_hist
          X_hist_err = X_hist_err/sum_X_hist

          synth_hists.append(synth_hist)
          synth_err_hists.append(synth_hist_err)
          X_hists.append(X_hist)
          X_err_hists.append(X_hist_err)

        plot_extra_name = GetYName(row, purpose="plot")
        file_extra_name = GetYName(row, purpose="file")

        plot_stacked_unrolled_2d_histogram_with_ratio(
          X_hists, 
          {"Synthetic": synth_hists}, 
          plot_bins[plot_col],
          unrolled_bins[col],
          self.data_parameters["X_columns"][unrolled_col],
          data_name='Simulated', 
          xlabel=self.data_parameters["X_columns"][plot_col],
          name=f"{self.plot_dir}/generation_unrolled_2d_{self.data_parameters['X_columns'][plot_col]}_{self.data_parameters['X_columns'][unrolled_col]}_y_{file_extra_name}", 
          data_hists_errors=X_err_hists, 
          stack_hists_errors=synth_err_hists, 
          title_right=f"y={plot_extra_name}" if plot_extra_name != "" else "",
          use_stat_err=False,
          ylabel = "Density",
        )

  def _GetYName(self, ur, purpose="plot", round_to=2):
    """
    Get a formatted label for a given unique row.

    Args:
        ur (list): List representing the unique row.
        purpose (str): Purpose of the label, either "plot" or "file
    """
    if len(ur) > 0:
      label_list = [str(round(i,round_to)) for i in ur] 
      if purpose == "file":
        name = "_".join([i.replace(".","p").replace("-","m") for i in label_list])
      elif purpose == "plot":
        if len(label_list) > 1:
          name = "({})".format(",".join(label_list))
        else:
          name = label_list[0]
    else:
      name = ""
    return name
  

  def PlotLikelihood(self, x, y, row, col, crossings, best_fit, true_pdf=None, columns=None):
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
          column_indices = [y_columns.index(col) for col in self.data_parameters["Y_columns"]]
          Y = Y[:,column_indices]
        if len(Y) == 1: Y = np.tile(Y, (len(X), 1))
        prob = np.zeros(len(X))
        for i in range(len(X)):
          prob[i] = true_pdf(X[i],Y[i])
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

    plot_likelihood(
      x, 
      y, 
      crossings, 
      name=f"{self.plot_dir}/likelihood_{col}_y_{file_extra_name}", 
      xlabel=col, 
      true_value=row[ind],
      title_right=f"y={plot_extra_name}" if plot_extra_name != "" else "",
      cap_at=9,
      label="Inferred N="+str(int(round(total_weight))),
      other_lklds=other_lkld,
    )

  def PlotComparisons(self, row, best_fit, ignore_quantile=0.001, n_bins=40, columns=None):
    """
    Plot comparisons between data, true, and best-fit distributions.

    Args:
        row (array-like): The row of data for which to plot the comparisons.
        best_fit (array-like): Best-fit parameters used for synthetic data generation.
        ignore_quantile (float): Fraction of data to ignore from both ends during histogram plotting (default is 0.001).
        n_bins (int): Number of bins for histogram plotting (default is 40).
        columns (list): List of column names for plotting. If None, uses Y_columns from data_parameters.
    """
    if columns == None: columns = self.data_parameters["Y_columns"]
    X, wt = self._GetXAndWts(row, columns=columns)
    synth_true = self.model.Sample(np.array([list(row)]), columns=columns)
    synth_best_fit = self.model.Sample(np.array([list(best_fit)]), columns=columns)
    if "yield" in self.data_parameters.keys():
      yf = MakeYieldFunction(self.pois, self.nuisances, self.data_parameters)
      synth_true_wt = (yf(row)/len(synth_true)) * np.ones(len(synth_true))
      synth_best_fit_wt = (yf(best_fit)/len(synth_best_fit)) * np.ones(len(synth_best_fit))
    else:
      synth_true_wt = (np.sum(wt)/len(synth_true)) * np.ones(len(synth_true))
      synth_best_fit_wt = (np.sum(wt)/len(synth_best_fit)) * np.ones(len(synth_best_fit))

    for col in range(X.shape[1]):
      trimmed_X = X[:,col]
      lower_value = np.quantile(trimmed_X, ignore_quantile)
      upper_value = np.quantile(trimmed_X, 1-ignore_quantile)
      trimmed_indices = ((trimmed_X >= lower_value) & (trimmed_X <= upper_value))
      trimmed_X = trimmed_X[trimmed_indices]
      trimmed_wt = wt[trimmed_indices].flatten()
      sim_hist, bins  = np.histogram(trimmed_X, weights=trimmed_wt,bins=n_bins)
      sim_hist_err_sq, _  = np.histogram(trimmed_X, weights=trimmed_wt**2, bins=bins)
      synth_true_hist, _  = np.histogram(synth_true[:,col], weights=synth_true_wt, bins=bins)
      synth_best_fit_hist, _  = np.histogram(synth_best_fit[:,col], weights=synth_best_fit_wt, bins=bins)

      file_extra_name = GetYName(row, purpose="file")
      plot_extra_name_true = GetYName(row, purpose="plot")
      plot_extra_name_bf = GetYName(best_fit, purpose="plot")

      plot_histograms(
        bins[:-1],
        [synth_best_fit_hist,synth_true_hist],
        [f"Learned y={plot_extra_name_bf} (Best Fit)",f"Learned y={plot_extra_name_true} (True)"],
        colors = ["blue","red",],
        linestyles = ["-","-"],
        title_right = f"y={plot_extra_name_true}",
        x_label=self.data_parameters["X_columns"][col],
        name=f"{self.plot_dir}/comparison_{self.data_parameters['X_columns'][col]}_y_{file_extra_name}", 
        y_label = "Events",
        error_bar_hists = [sim_hist],
        error_bar_hist_errs = [np.sqrt(sim_hist_err_sq)],
        error_bar_names = [f"Data y={plot_extra_name_true}"],
      )

  def DrawProbability(self, y_vals, n_bins=100, ignore_quantile=0.001):
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

    plot_histograms(
      bins[:-1],
      hists,
      [f"y={y_val}" for y_val in y_vals],
      title_right = "",
      x_label=self.data_parameters["X_columns"][0],
      name=f"{self.plot_dir}/probability", 
      y_label = "p(x|y)",
      anchor_y_at_0 = True,
    )