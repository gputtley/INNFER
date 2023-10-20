from data_processor import DataProcessor
from model import Model
from plotting import plot_likelihood, plot_histograms, plot_histogram_with_ratio, plot_2d_likelihood
from scipy.optimize import minimize
from scipy.integrate import simpson
import bayesflow as bf
import tensorflow as tf
import numpy as np
import copy

class Inference():

  def __init__(self, model, data_processor):

    self.model = model
    self.data_processor = data_processor

    self.best_fits = {}
    self.crossings = {}

    self.n_events_in_toy=100
    self.seed_for_toy=42

    self.plot_dir = "plots"

  def Sample(self, from_data=None, from_row=None, n_events=10**5):
    if from_data:
      preprocessed_data = self.data_processor.PreProcess(purpose="inference", only_context=True)[from_data]
    elif isinstance(from_row,np.ndarray):
      array = np.tile(from_row, (n_events, 1))
      data = copy.deepcopy(self.data_processor)
      data.data = {"train": {"Y" : array}}
      preprocessed_data = data.PreProcess(purpose="inference", only_context=True)["train"]

    preprocessed_data["direct_conditions"] = preprocessed_data["direct_conditions"].astype(np.float32)

    synth = self.model.amortizer.sample(preprocessed_data, 1)
    synth = self.data_processor.UnStandardise(synth[:,:,0])

    return synth

  def PlotClosure(self):

    # loop through train and test (if in keys)
    dks = ["train"]
    if "test" in self.data_processor.data.keys(): dks.append("test")
    for dk in dks:

      synth = self.Sample(from_data=dk)

      # Loop through columns
      for col in range(self.data_processor.data[dk]["X"].shape[1]):

        # Make the histograms
        hist, bins = np.histogram(self.data_processor.data[dk]["X"][:,col],bins=40)
        synth_hist, _ = np.histogram(synth[:,col], bins=bins)

        # Plot synthetic dataset against the simulated dataset
        plot_histogram_with_ratio(
          hist, 
          synth_hist, 
          bins, 
          name_1='Simulated', 
          name_2='Synthetic',
          xlabel=f"x[{col}]",
          name=f"{self.plot_dir}/closure_{dk}_x{col}", 
          title_right = dk.capitalize(),
          density = True,
          use_stat_err = True,
          )
  
  def PlotGeneration(self, data_key="val", n_bins=40):

    unique_rows = np.unique(self.data_processor.data[data_key]["Y"], axis=0)
    for ur in unique_rows:

      matching_rows = np.all(self.data_processor.data[data_key]["Y"] == ur, axis=1)
      X_cut = self.data_processor.data[data_key]["X"][matching_rows]

      synth = self.Sample(from_row=ur)

      for col in range(self.data_processor.data[data_key]["X"].shape[1]):
        sim_hist, bins  = np.histogram(X_cut[:,col], bins=n_bins)
        synth_hist, _  = np.histogram(synth[:,col], bins=bins)
        
        file_extra_name = self._GetYName(ur, purpose="file")
        plot_extra_name = self._GetYName(ur, purpose="plot")
    
        plot_histogram_with_ratio(
          sim_hist, 
          synth_hist, 
          bins, 
          name_1='Simulated', 
          name_2='Synthetic',
          xlabel=f"x[{col}]",
          name=f"{self.plot_dir}/generation_x{col}_y{file_extra_name}", 
          title_right = f"y={plot_extra_name}",
          density = True,
          use_stat_err = True,
          )
        
  def PlotProbAndSampleDensityFor1DX(self, data_key="val", n_bins=40):

    unique_rows = np.unique(self.data_processor.data[data_key]["Y"], axis=0)
    for ur in unique_rows:
      synth = self.Sample(from_row=ur)
      synth_hist,bins = np.histogram(synth[:,0],bins=n_bins,density=True)
      bin_centers = [i+(bins[1]-bins[0])/2 for i in bins[:-1]]
      prob_hist = self._GetProb(np.array(bin_centers).reshape(-1,1), ur, integrate_over_dummy=self.data_processor.add_dummy_X)

      file_extra_name = self._GetYName(ur, purpose="file")
      plot_extra_name = self._GetYName(ur, purpose="plot")

      plot_histograms(
        bin_centers,
        [prob_hist,synth_hist],
        ["Probability", "Sampled Density"],
        title_right = "",
        name = f"{self.plot_dir}/probability_x_y{file_extra_name}",
        x_label = "x",
        y_label = f"p(x|{plot_extra_name})",
      )

  def _GetProb(self, data, row, integrate_over_dummy=False, integral_ranges=[-4.0,4.0], integral_bins=100):

    array = np.tile(row, (len(data), 1))
    prob_data = {
      "train" : {
        "X" : data,
        "Y" : array
      }
    }  
    data_proc = copy.deepcopy(self.data_processor)
    data_proc.data = prob_data
    preprocessed_data = data_proc.PreProcess(purpose="inference")["train"]
    preprocessed_data["direct_conditions"] = preprocessed_data["direct_conditions"].astype(np.float32)

    if integrate_over_dummy:
      int_space = np.linspace(integral_ranges[0],integral_ranges[1],num=integral_bins)
      x = preprocessed_data["parameters"]
      x = x[:,:-1]
      x2, x1 = np.meshgrid(int_space, x)
      preprocessed_data["parameters"] = np.column_stack((x1.ravel(), x2.ravel()))
      preprocessed_data["direct_conditions"] = np.tile(preprocessed_data["direct_conditions"][0,:], (len(preprocessed_data["parameters"]), 1))

    tf.random.set_seed(42)
    prob = np.exp(self.model.amortizer.log_posterior(preprocessed_data))

    if integrate_over_dummy:
      p = np.array([])
      for i in range(len(data)):
        start_idx = i * integral_bins
        end_idx = (i + 1) * integral_bins
        p = np.append(p,simpson(prob[start_idx:end_idx], dx=int_space[1]-int_space[0]))  
      prob = p   
    prob = self.data_processor.PostProcessProb(prob)
    return prob

  def _MakeToy(self, ur, data_key="val"):
      
      matching_rows = np.all(self.data_processor.data[data_key]["Y"] == ur, axis=1)
      X_cut = self.data_processor.data[data_key]["X"][matching_rows]
      rng = np.random.RandomState(seed=self.seed_for_toy)
      random_indices = rng.choice(X_cut.shape[0], self.n_events_in_toy, replace=False)
      toy = X_cut[random_indices]
      return toy

  def _NLL(self, row, data, shift=0, absolute=False, bounds=None, freeze=[]):

    for ind, i in enumerate(row):
      if bounds == None: break
      if i > bounds[ind][1] or i < bounds[ind][0]:
        return np.inf

    if len(freeze) != 0:
      new_row = np.zeros(len(freeze))
      for ind, i in enumerate(freeze):
        if i != None:
          new_row[ind] = i
        else:
          new_row[ind] = row[0]
      row = new_row

    nll =-2*np.sum(np.log(self._GetProb(data, row, integrate_over_dummy=self.data_processor.add_dummy_X)))
    nll = nll-shift
    if absolute:
      nll = np.abs(nll)
    return nll
  
  def _GetBestFit(self, data, initial_guess, method='Nelder-Mead', tolerence=0.1, options={'xatol': 0.0001, 'fatol': 0.0001, 'maxiter': 20}, bounds=None, freeze=[]):
    return minimize(self._NLL, initial_guess, args=(data,0,False,bounds,freeze), method=method, tol=tolerence, options=options)

  def _Get1DInterval(self, data, initial_guess, interval, min_nll, method='Nelder-Mead', tolerence=0.1, options={'xatol': 0.0001, 'fatol': 0.0001, 'maxiter': 20}, bounds=None, freeze=[]):
    return minimize(self._NLL, initial_guess, args=(data, min_nll+(interval**2), True, bounds, freeze), method=method, tol=tolerence, options=options)
  
  def GetBestFit(self, initial_guess, data_key="val"):
    unique_rows = np.unique(self.data_processor.data[data_key]["Y"], axis=0)
    for ur in unique_rows:
      data = self._MakeToy(ur)
      self.best_fits[tuple(ur)] = self._GetBestFit(data, initial_guess)
      
  def GetProfiledIntervals(self, data_key="val"):

    unique_rows = np.unique(self.data_processor.data[data_key]["Y"], axis=0)
    for ur in unique_rows:
      self.crossings[tuple(ur)] = {}
      data = self._MakeToy(ur)
      for col in range(len(ur)):
        best_fits = self.best_fits[tuple(ur)].x
        min_lkld = self.best_fits[tuple(ur)].fun
        freeze = [best_fits[c] if c != col else None for c in range(len(ur))]
        self.crossings[tuple(ur)][col] = {}
        up_bounds = [[best_fits[col],np.inf]]
        down_bounds = [[-np.inf,best_fits[col]]]
        self.crossings[tuple(ur)][col]["-1"] = self._Get1DInterval(data, np.array([best_fits[col]]), 1, min_lkld, bounds=down_bounds, freeze=freeze)
        self.crossings[tuple(ur)][col]["+1"] = self._Get1DInterval(data, np.array([best_fits[col]]), 1, min_lkld, bounds=up_bounds, freeze=freeze)


  def PrintBestFitAndIntervals(self,data_key="val", precision=4):

    unique_rows = np.unique(self.data_processor.data[data_key]["Y"], axis=0)
    np.set_printoptions(precision=precision, suppress=True, threshold=5)
    for ur in unique_rows: 
      print(">> Row:",ur)
      for col in range(len(ur)):
        best_fit = self.best_fits[tuple(ur)].x[col]
        up = self.crossings[tuple(ur)][col]["+1"].x[0] - best_fit
        down =  best_fit - self.crossings[tuple(ur)][col]["-1"].x[0]
        print(" >> Varying column:", col)
        print(f"   >> Result: {round(best_fit,precision)} + {round(up,precision)} - {round(down,precision)}")

  def Draw1DLikelihoods(self, data_key="val", use_intervals_for_range=True, est_n_sigmas_shown=3, plot_range=[0.0,1.0], n_points=20):
  
    unique_rows = np.unique(self.data_processor.data[data_key]["Y"], axis=0)
    for ur in unique_rows: 
      data = self._MakeToy(ur)
      for col in range(len(ur)):
        best_fit = self.best_fits[tuple(ur)].x[col]
        up = self.crossings[tuple(ur)][col]["+1"].x[0] - best_fit
        down =  best_fit - self.crossings[tuple(ur)][col]["-1"].x[0]
        if use_intervals_for_range:
          plot_range = [best_fit-(est_n_sigmas_shown*down), best_fit+(est_n_sigmas_shown*up)]
        X = np.linspace(plot_range[0], plot_range[1], num=n_points)
        freeze = [self.best_fits[tuple(ur)].x[c] if c != col else None for c in range(len(ur))]
        nll = []
        for x in X:
          nll.append(self._NLL(np.array([x]), data, shift=self.best_fits[tuple(ur)].fun, absolute=False, bounds=None, freeze=freeze))  

        file_extra_name = self._GetYName(ur, purpose="file")
        plot_extra_name = self._GetYName(ur, purpose="plot")

        plot_likelihood(
          X, 
          nll, 
          {-1:self.crossings[tuple(ur)][col]["-1"].x[0], 1 : self.crossings[tuple(ur)][col]["+1"].x[0]}, 
          name=f"{self.plot_dir}/likelihood_1d_y{col}_truey{file_extra_name}", 
          xlabel=f"y[{col}]", 
          true_value=ur[col],
          title_right=f"y={plot_extra_name}",
          cap_at=est_n_sigmas_shown**2,
          label = "Inferred",
          )
        
  def DrawComparison(self,data_key="val",n_bins=40):
      
      unique_rows = np.unique(self.data_processor.data[data_key]["Y"], axis=0)
      for ur in unique_rows: 
        bf_row = np.array([round(float(self.best_fits[tuple(ur)].x[c]),2) for c in range(len(ur))])
        data = self._MakeToy(ur)
        n_events = copy.deepcopy(self.n_events_in_toy)
        synth_true = self.Sample(from_row=ur)
        synth_bf = self.Sample(from_row=bf_row)
        self.n_events_in_toy = 10**5
        high_stat_data_true = self._MakeToy(ur)
        self.n_events_in_toy = n_events

        file_extra_name = self._GetYName(ur, purpose="file")
        plot_extra_name_true = self._GetYName(ur, purpose="plot")
        plot_extra_name_bf = self._GetYName(bf_row, purpose="plot")

        for col in range(self.data_processor.data[data_key]["X"].shape[1]):

          data_hist, bins = np.histogram(data[:,col], bins=n_bins)
          data_uncert_hist = np.sqrt(data_hist)/sum(data_hist)
          data_hist = data_hist/sum(data_hist)

          synth_true_hist, _ = np.histogram(synth_true[:,col], bins=bins, density=True)
          synth_bf_hist, _ = np.histogram(synth_bf[:,col], bins=bins, density=True)
          high_stat_data_true_hist, _ = np.histogram(high_stat_data_true[:,col], bins=bins, density=True)

          plot_histograms(
            bins[:-1],
            [synth_bf_hist,synth_true_hist,high_stat_data_true_hist],
            [f"Synth y={plot_extra_name_bf}",f"Synth y={plot_extra_name_true}",f"True y={plot_extra_name_true}"],
            colors = ["blue","red","red"],
            linestyles = ["--","--","-"],
            title_right = "",
            name=f"{self.plot_dir}/comparison_x{col}_y{file_extra_name}", 
            x_label = f"x[{col}]",
            y_label = "Density",
            error_bar_hists = [data_hist],
            error_bar_hist_errs = [data_uncert_hist],
            error_bar_names = [f"Data y={plot_extra_name_true}"],
          )

  def _GetYName(self, ur, purpose="plot"):
    label_list = [str(i) for i in ur] 
    if purpose == "file":
      name = "_".join([i.replace(".","p") for i in label_list])
    elif purpose == "plot":
      if len(label_list) > 1:
        name = "({})".format(",".join(label_list))
      else:
        name = label_list[0]
    return name
  
  def Draw2DLikelihoods(self, data_key="val", use_intervals_for_range=True, est_n_sigmas_shown=4, plot_range=[[0.0,1.0],[0.0,1.0]], n_points_per_dim=20):

      unique_rows = np.unique(self.data_processor.data[data_key]["Y"], axis=0)
      for ur in unique_rows: 
        data = self._MakeToy(ur)

        best_fit = [float(self.best_fits[tuple(ur)].x[i]) for i in [0,1]]
        if use_intervals_for_range:
          x_up = self.crossings[tuple(ur)][0]["+1"].x[0] - best_fit[0]
          x_down =  best_fit[0] - self.crossings[tuple(ur)][0]["-1"].x[0]
          y_up = self.crossings[tuple(ur)][1]["+1"].x[0] - best_fit[1]
          y_down =  best_fit[1] - self.crossings[tuple(ur)][1]["-1"].x[0]
          plot_range = [[best_fit[0]-(est_n_sigmas_shown*x_down), best_fit[0]+(est_n_sigmas_shown*x_up)], [best_fit[1]-(est_n_sigmas_shown*y_down), best_fit[1]+(est_n_sigmas_shown*y_up)]]

        X = np.linspace(plot_range[0][0], plot_range[0][1], num=n_points_per_dim)
        Y = np.linspace(plot_range[1][0], plot_range[1][1], num=n_points_per_dim)
        nll = []
        for y in Y:
          nll.append([self._NLL(np.array([x,y]), data, shift=self.best_fits[tuple(ur)].fun) for x in X]) 

        file_extra_name = self._GetYName(ur, purpose="file")
        plot_extra_name_true = self._GetYName(ur, purpose="plot")

        plot_2d_likelihood(
          X, 
          Y, 
          nll, 
          name=f"{self.plot_dir}/likelihood_2d_truey{file_extra_name}", 
          xlabel="y[0]", 
          ylabel="y[1]", 
          best_fit=best_fit, 
          true_value=[ur[0],ur[1]], 
          title_right=f"y={plot_extra_name_true}"
          )