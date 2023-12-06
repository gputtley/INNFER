from preprocess import PreProcess
from likelihood import Likelihood
from plotting import plot_histograms, plot_histogram_with_ratio, plot_likelihood
from other_functions import GetYName
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
    self.n_asimov_events = 1000.0
    self.data_key = "val"
    self.tolerance = 1e-6
    self.data_dir = "./data/"
    self.plot_dir = "./plots/"
    self._SetOptions(options)

    self.pp = PreProcess()
    self.pp.parameters = self.data_parameters
    self.pp.output_dir = self.data_dir

  def _SetOptions(self, options):
    """
    Set options for the validation process.

    Args:
        options (dict): Dictionary of options for the validation process.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def _GetXAndWts(self, row, columns=None, use_nominal_wt=False):

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
    return X, wt

  def BuildLikelihood(self):

    self.lkld = Likelihood(
      {"pdfs":{self.model_name:self.model}}, 
      type="unbinned", 
      data_parameters={
        self.model_name:{
          "X_columns" : self.data_parameters["X_columns"],
          "Y_columns" : self.data_parameters["Y_columns"]
        },
      }
    )

  def GetAndDumpBestFit(self, row, initial_guess, columns=None, ind=0):
    if columns == None: columns = self.data_parameters["Y_columns"]
    X, wt = self._GetXAndWts(row, columns=columns)
    if self.n_asimov_events is not None:
      wt *= (self.n_asimov_events/np.sum(wt))
    self.lkld.GetAndWriteBestFitToYaml(X, row, initial_guess, wt=wt, filename=f"{self.data_dir}/best_fit_{ind}.yaml")

  def GetAndDumpScanRanges(self, row, col, columns=None, ind=0):
    if columns == None: columns = self.data_parameters["Y_columns"]
    X, wt = self._GetXAndWts(row, columns=columns)
    if self.n_asimov_events is not None:
      wt *= (self.n_asimov_events/np.sum(wt))
    self.lkld.GetAndWriteScanRangesToYaml(X, row, col, wt=wt, filename=f"{self.data_dir}/scan_values_{col}_{ind}.yaml")

  def GetAndDumpNLL(self, row, col, col_val, columns=None, ind1=0, ind2=0):
    if columns == None: columns = self.data_parameters["Y_columns"]
    X, wt = self._GetXAndWts(row, columns=columns)
    if self.n_asimov_events is not None:
      wt *= (self.n_asimov_events/np.sum(wt))
    self.lkld.GetAndWriteNLLToYaml(X, row, col, col_val, wt=wt, filename=f"{self.data_dir}/scan_results_{col}_{ind1}_{ind2}.yaml")


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
    synth = self.model.Sample(np.array([list(row)]), columns=columns)

    for col in range(X.shape[1]):

      trimmed_X = X[:,col]
      lower_value = np.quantile(trimmed_X, ignore_quantile)
      upper_value = np.quantile(trimmed_X, 1-ignore_quantile)
      trimmed_indices = ((trimmed_X >= lower_value) & (trimmed_X <= upper_value))
      trimmed_X = trimmed_X[trimmed_indices]
      trimmed_wt = wt[trimmed_indices].flatten()

      sim_hist, bins  = np.histogram(trimmed_X, weights=trimmed_wt,bins=n_bins)
      sim_hist_err_sq, _  = np.histogram(trimmed_X, weights=trimmed_wt**2, bins=bins)
      synth_hist, _  = np.histogram(synth[:,col], bins=bins)
      
      file_extra_name = GetYName(row, purpose="file")
      plot_extra_name = GetYName(row, purpose="plot")

      plot_histogram_with_ratio(
        sim_hist, 
        synth_hist, 
        bins, 
        name_1='Simulated', 
        name_2='Synthetic',
        xlabel=self.data_parameters["X_columns"][col],
        name=f"{self.plot_dir}/generation_{self.data_parameters['X_columns'][col]}_y_{file_extra_name}", 
        title_right = f"y={plot_extra_name}",
        density = True,
        use_stat_err = False,
        errors_1=np.sqrt(sim_hist_err_sq), 
        errors_2=np.sqrt(synth_hist),
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

    print(f"{col}: {crossings[0]} + {crossings[1]-crossings[0]} - {crossings[0]-crossings[-1]}")
    if columns == None: columns = self.data_parameters["Y_columns"]
    ind = columns.index(col)
    file_extra_name = GetYName(row, purpose="file")
    plot_extra_name = GetYName(row, purpose="plot")

    # make true likelihood
    other_lkld = {}
    if true_pdf is not None:
      X, wt = self._GetXAndWts(row, columns=columns)
      if self.n_asimov_events is not None:
        wt *= (self.n_asimov_events/np.sum(wt))
      nlls = []
      for x_val in x:
        nll = 0
        for data_ind, data in enumerate(X):
          test_row = copy.deepcopy(best_fit)
          test_row[ind] = x_val
          pdf = true_pdf(data,test_row)
          nll += -2*np.log(pdf**wt.flatten()[data_ind])
        nlls.append(nll)

      true_nll = 0
      for data_ind, data in enumerate(X):
        true_nll += -2*np.log(true_pdf(data,row)**wt.flatten()[data_ind])

      nlls = [nll - true_nll for nll in nlls]
      other_lkld = {"True":nlls}

    plot_likelihood(
      x, 
      y, 
      crossings, 
      name=f"{self.plot_dir}/likelihood_{col}_y_{file_extra_name}", 
      xlabel=col, 
      true_value=row[ind],
      title_right=f"y={plot_extra_name}",
      cap_at=9,
      label="Inferred",
      other_lklds=other_lkld,
    )

  def PlotComparisons(self, row, best_fit, ignore_quantile=0.001, n_bins=40, columns=None):

    if columns == None: columns = self.data_parameters["Y_columns"]
    X, wt = self._GetXAndWts(row, columns=columns)
    if self.n_asimov_events is not None:
      wt *= (self.n_asimov_events/np.sum(wt))
    synth_true = self.model.Sample(np.array([list(row)]), columns=columns)
    synth_best_fit = self.model.Sample(np.array([list(best_fit)]), columns=columns)

    for col in range(X.shape[1]):
      trimmed_X = X[:,col]
      lower_value = np.quantile(trimmed_X, ignore_quantile)
      upper_value = np.quantile(trimmed_X, 1-ignore_quantile)
      trimmed_indices = ((trimmed_X >= lower_value) & (trimmed_X <= upper_value))
      trimmed_X = trimmed_X[trimmed_indices]
      trimmed_wt = wt[trimmed_indices].flatten()
      sim_hist, bins  = np.histogram(trimmed_X, weights=trimmed_wt,bins=n_bins)
      sim_hist_err_sq, _  = np.histogram(trimmed_X, weights=trimmed_wt**2, bins=bins)
      synth_true_hist, _  = np.histogram(synth_true[:,col], bins=bins)
      synth_best_fit_hist, _  = np.histogram(synth_best_fit[:,col], bins=bins)

      sim_hist_err_sq = sim_hist_err_sq/np.sum(sim_hist)
      sim_hist = sim_hist/np.sum(sim_hist)
      synth_true_hist = synth_true_hist/np.sum(synth_true_hist)
      synth_best_fit_hist = synth_best_fit_hist/np.sum(synth_best_fit_hist)

      file_extra_name = GetYName(row, purpose="file")
      plot_extra_name_true = GetYName(row, purpose="plot")
      plot_extra_name_bf = GetYName(best_fit, purpose="plot")

      plot_histograms(
        bins[:-1],
        [synth_best_fit_hist,synth_true_hist],
        [f"Learned y={plot_extra_name_bf}",f"Learned y={plot_extra_name_true}"],
        colors = ["blue","red",],
        linestyles = ["-","-"],
        title_right = "",
        x_label=self.data_parameters["X_columns"][col],
        name=f"{self.plot_dir}/comparison_{self.data_parameters['X_columns'][col]}_y_{file_extra_name}", 
        y_label = "Density",
        error_bar_hists = [sim_hist],
        error_bar_hist_errs = [sim_hist_err_sq],
        error_bar_names = [f"Data y={plot_extra_name_true}"],
      )

  def PlotUnbinnedLikelihood(self, row, initial_guess, columns=None, data_key="val", n_bins=40, ignore_quantile=0.01, tolerance=1e-6, n_asimov_events=1000.0, true_pdf=None):
    """
    Plots the unbinned likelihood scan for each parameter in the specified row.

    Args:
        row (array-like): The row of data for which to plot the likelihood scan.
        initial_guess (array-like): Initial guess for the parameters during the likelihood scan.
        columns (list, optional): List of column names to consider during the likelihood scan (default is None, which uses all columns).
        data_key (str, optional): The key to access the data in the row (default is "val").
        n_bins (int, optional): Number of bins for histogram plots (default is 40).
        ignore_quantile (float, optional): The fraction of data points to ignore on each side of the distribution when plotting comparisons (default is 0.01).
        tolerance (float, optional): Tolerance parameter for extracting X and weights from the row (default is 1e-6).
        n_asimov_events (float, optional): The number of Asimov events to use for the unbinned likelihood (default is 1000.0).
        true_pdf (function, optional): True probability density function used for comparison plots (default is None).

    Returns:
        None
    """
    X, wt = self._GetXAndWts(row, columns=columns, data_key=data_key, tolerance=tolerance)
    wt *= (n_asimov_events/np.sum(wt))

    lkld = Likelihood(
      {"pdfs":{self.model_name:self.model}}, 
      type="unbinned", 
      data_parameters={
        self.model_name:{
          "X_columns" : self.data_parameters["X_columns"],
          "Y_columns" : self.data_parameters["Y_columns"]
        },
      }
    )

    print(">> Getting best fit")
    lkld.GetBestFit(X, np.array(initial_guess), wts=wt)

    if columns == None: columns = self.data_parameters["Y_columns"]

    # Plot likelihood scan
    print(">> Runnning scan")
    for ind, col in enumerate(columns):
      x, y, crossings = lkld.MakeScanInSeries(X, col, wts=wt)
      print(f"{col}: {lkld.best_fit[ind]} + {crossings[1]-lkld.best_fit[ind]} - {lkld.best_fit[ind]-crossings[-1]}")
      file_extra_name = GetYName(row, purpose="file")
      plot_extra_name = GetYName(row, purpose="plot")

      # make true likelihood
      other_lkld = {}
      if true_pdf is not None:
        nlls = []
        for x_val in x:
          nll = 0
          for data_ind, data in enumerate(X):
            test_row = copy.deepcopy(lkld.best_fit)
            test_row[ind] = x_val
            nll += -2*np.log(true_pdf(data,test_row)**wt.flatten()[data_ind])
          nlls.append(nll)

        true_nll = 0
        for data_ind, data in enumerate(X):
          true_nll += -2*np.log(true_pdf(data,row)**wt.flatten()[data_ind])

        nlls = [nll - true_nll for nll in nlls]
        other_lkld = {"True":nlls}

      plot_likelihood(
        x, 
        y, 
        crossings, 
        name=f"{self.plot_dir}/likelihood_{col}_y_{file_extra_name}", 
        xlabel=col, 
        true_value=row[ind],
        title_right=f"y={plot_extra_name}",
        cap_at=9,
        label="Inferred",
        other_lklds=other_lkld,
      )
    
    # Plot comparison of best fit learned distribution and the true distribution
    print(">> Drawing comparisons")
    synth_true = self.model.Sample(np.array([list(row)]), columns=columns)
    synth_best_fit = self.model.Sample(np.array([list(lkld.best_fit)]), columns=columns)

    for col in range(X.shape[1]):
      trimmed_X = X[:,col]
      lower_value = np.quantile(trimmed_X, ignore_quantile)
      upper_value = np.quantile(trimmed_X, 1-ignore_quantile)
      trimmed_indices = ((trimmed_X >= lower_value) & (trimmed_X <= upper_value))
      trimmed_X = trimmed_X[trimmed_indices]
      trimmed_wt = wt[trimmed_indices].flatten()
      sim_hist, bins  = np.histogram(trimmed_X, weights=trimmed_wt,bins=n_bins)
      sim_hist_err_sq, _  = np.histogram(trimmed_X, weights=trimmed_wt**2, bins=bins)
      synth_true_hist, _  = np.histogram(synth_true[:,col], bins=bins)
      synth_best_fit_hist, _  = np.histogram(synth_best_fit[:,col], bins=bins)

      sim_hist_err_sq = sim_hist_err_sq/np.sum(sim_hist)
      sim_hist = sim_hist/np.sum(sim_hist)
      synth_true_hist = synth_true_hist/np.sum(synth_true_hist)
      synth_best_fit_hist = synth_best_fit_hist/np.sum(synth_best_fit_hist)

      file_extra_name = GetYName(row, purpose="file")
      plot_extra_name_true = GetYName(row, purpose="plot")
      plot_extra_name_bf = GetYName(lkld.best_fit, purpose="plot")

      plot_histograms(
        bins[:-1],
        [synth_best_fit_hist,synth_true_hist],
        [f"Learned y={plot_extra_name_bf}",f"Learned y={plot_extra_name_true}"],
        colors = ["blue","red",],
        linestyles = ["-","-"],
        title_right = "",
        x_label=self.data_parameters["X_columns"][col],
        name=f"{self.plot_dir}/comparison_{self.data_parameters['X_columns'][col]}_y_{file_extra_name}", 
        y_label = "Density",
        error_bar_hists = [sim_hist],
        error_bar_hist_errs = [sim_hist_err_sq],
        error_bar_names = [f"Data y={plot_extra_name_true}"],
      )
