import copy
import os
import textwrap

import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from useful_functions import MakeDirectories, RoundToSF

hep.style.use("CMS")
cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""

def plot_histograms(
    bins,
    hists,
    hist_names,
    colors = [i['color'] for i in plt.rcParams['axes.prop_cycle']]*100,
    linestyles = ["-"]*100,
    title_right = "",
    name = "hists.pdf",
    x_label = "",
    y_label = "",
    error_bar_hists = [],
    error_bar_hist_errs = [],
    error_bar_names = [],
    anchor_y_at_0 = False,
    drawstyle = "default",
    smooth_func = None,
    smooth_func_name = "",
    smooth_func_color = "green",
    vertical_lines = [],
    vertical_line_names = [],
    vertical_line_colors = [i['color'] for i in plt.rcParams['axes.prop_cycle']]*100,
  ):
  """
  Plot histograms with optional error bars.

  Parameters
  ----------
  bins : array-like
      Bin edges.
  hists : list of array-like
      List of histogram values.
  hist_names : list of str
      Names for each histogram.
  colors : list of str, optional
      Colors for each histogram. Defaults to Matplotlib color cycle.
  linestyles : list of str, optional
      Linestyles for each histogram. Defaults to solid line.
  title_right : str, optional
      Text to be displayed at the top-right corner of the plot.
  name : str, optional
      Name of the output file (without extension). Defaults to "hists.pdf".
  x_label : str, optional
      Label for the x-axis.
  y_label : str, optional
      Label for the y-axis.
  error_bar_hists : list of array-like, optional
      List of histograms for error bars.
  error_bar_hist_errs : list of array-like, optional
      List of errors for each error bar histogram.
  error_bar_names : list of str, optional
      Names for each error bar histogram.
  anchor_y_at_0 : bool, optional
      If True, anchor the y-axis at 0. Defaults to False.
  drawstyle : str, optional
      Drawstyle for the histograms. Defaults to "default".
  """

  fig, ax = plt.subplots()
  hep.cms.text(cms_label,ax=ax)

  for ind, hist in enumerate(hists):
    plt.plot(bins, hist, label=hist_names[ind], color=colors[ind], linestyle=linestyles[ind], drawstyle=drawstyle)

  for ind, hist in enumerate(error_bar_hists):
    non_empty_bins = hist != 0
    plt.errorbar(bins[non_empty_bins], hist[non_empty_bins], yerr=error_bar_hist_errs[ind][non_empty_bins], label=error_bar_names[ind], markerfacecolor='none', linestyle='None', fmt='k+')

  ax.text(1.0, 1.0, title_right,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax.transAxes)

  if anchor_y_at_0:
    ax.set_ylim(bottom=0, top=1.2*max(np.max(hist) for hist in hists))

  if smooth_func is not None:
    x_func = np.linspace(min(bins),max(bins),num=200)
    y_func = [smooth_func(x) for x in x_func] 
    plt.plot(x_func, y_func, label=smooth_func_name, color=smooth_func_color)

  for ind, vertical_line in enumerate(vertical_lines):
    ax.axvline(x=vertical_line, color=vertical_line_colors[ind], linestyle='--', linewidth=2, label=vertical_line_names[ind])

  ax.xaxis.get_major_formatter().set_useOffset(False)

  plt.xlabel(x_label)
  plt.ylabel(y_label)
  if not all(item is None for item in hist_names+error_bar_names):
    plt.legend()
  plt.tight_layout()
  MakeDirectories(name+".pdf")
  plt.savefig(name+".pdf")
  print("Created {}.pdf".format(name))
  plt.close()

def plot_likelihood(
    x, 
    y, 
    crossings, 
    name="lkld", 
    xlabel="", 
    true_value=None, 
    cap_at=9, 
    other_lklds={}, 
    label=None, 
    under_result=""
  ):
  """
  Plot likelihood curve.

  Parameters
  ----------
  x : array-like
      X-axis values.
  y : array-like
      Y-axis values.
  crossings : dict
      Dictionary containing special points in the likelihood curve.
  name : str, optional
      Name of the output file (without extension). Defaults to "lkld".
  xlabel : str, optional
      Label for the x-axis.
  true_value : float, optional
      True value to be marked on the plot.
  cap_at : float, optional
      Cap the y-axis at this value. Defaults to 9.
  other_lklds : dict, optional
      Additional likelihood curves to be overlaid.
  label : str, optional
      Label for the likelihood curve.
  """

  if cap_at != None:
    sel_inds = []
    x_plot = []
    y_plot = []
    for ind, i in enumerate(y):
      if i < cap_at:
        x_plot.append(x[ind])
        y_plot.append(i)
        sel_inds.append(ind)
    x = x_plot
    y = y_plot
    y_max = cap_at
  else:
    y_max = max(y)
    sel_inds = range(len(y))

  fig, ax = plt.subplots()
  hep.cms.text(cms_label,ax=ax)
  plt.plot(x, y, label=label)

  ax.xaxis.get_major_formatter().set_useOffset(False)

  colors = rgb_palette = sns.color_palette("Set2", len(list(other_lklds.keys())))
  color_ind = 0
  for k, v in other_lklds.items():
    plt.plot(v[0], v[1], label=k, color=colors[color_ind])
    color_ind += 1

  if true_value != None:
    plt.plot([true_value,true_value], [0,y_max], linestyle='--', color='black')

    ax.text(1.0, 1.0, lumi_label,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes)

  if -1 in crossings.keys():  
    plt.plot([crossings[-1],crossings[-1]], [0,1], linestyle='--', color='orange')
  if -2 in crossings.keys():  
    plt.plot([crossings[-2],crossings[-2]], [0,4], linestyle='--', color='orange')
  if 1 in crossings.keys():  
    plt.plot([crossings[1],crossings[1]], [0,1], linestyle='--', color='orange')
  if 2 in crossings.keys():  
    plt.plot([crossings[2],crossings[2]], [0,4], linestyle='--', color='orange')
  plt.plot([x[0],x[-1]], [1,1], linestyle='--', color='gray')
  plt.plot([x[0],x[-1]], [4,4], linestyle='--', color='gray')
  
  if label is not None:
    plt.legend(loc='upper right')

  if -1 in crossings.keys() and 1 in crossings.keys():
    # round crossings difference to two SF
    sig_figs = 2
    decimal_places_up = sig_figs - int(np.floor(np.log10(abs(crossings[1]-crossings[0])))) - 1
    decimal_places_down = sig_figs - int(np.floor(np.log10(abs(crossings[0]-crossings[-1])))) - 1
    decimal_places = min(decimal_places_down,decimal_places_up)

    text = f'Result: {round(crossings[0],decimal_places)} + {round(crossings[1]-crossings[0],decimal_places)} - {round(crossings[0]-crossings[-1],decimal_places)}'
    ax.text(0.03, 0.96, text, transform=ax.transAxes, va='top', ha='left', fontsize=20)

  if under_result != "":
    ax.text(0.03, 0.9, under_result, transform=ax.transAxes, va='top', ha='left', fontsize=20)

  plt.xlim(x[0],x[-1])
  plt.ylim(0,y_max)
  plt.xlabel(xlabel)
  plt.ylabel(r'$-2\Delta \ln L$')
  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig("{}.pdf".format(name))
  plt.close()

def plot_spline_and_thresholds(
    spline, 
    thresholds, 
    unique_vals, 
    counts, 
    x_label="", 
    y_label="Density",
    name="spline_and_thresholds"
  ):
  """
  Plot a spline curve along with data points and shaded threshold regions.

  Parameters
  ----------
  spline : callable
      A spline function that takes an array of x values and returns corresponding y values.
  thresholds : dict
      Dictionary containing threshold regions with keys as labels and values as tuples (start, end).
  unique_vals : array-like
      Array of unique x values for the data points.
  counts : array-like
      Array of y values (counts) corresponding to the unique x values.
  x_label : str, optional
      Label for the x-axis. Defaults to an empty string.
  y_label : str, optional
      Label for the y-axis. Defaults to "Density".
  name : str, optional
      Name of the output file (without extension) where the plot will be saved. Defaults to "spline_and_thresholds".
  """

  fig, ax = plt.subplots()
  hep.cms.text(cms_label,ax=ax)

  plt.scatter(unique_vals, counts, label='Data')

  x_smooth = np.linspace(min(unique_vals), max(unique_vals), 500)
  y_smooth = spline(x_smooth)
  plt.plot(x_smooth, y_smooth, color='blue', label='Spline')

  for key, val in thresholds.items():
    x_frac = np.linspace(val[0], val[1], 500)
    y_frac = spline(x_frac)    
    plt.fill_between(x_frac, y_frac, alpha=0.5)

  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  plt.subplots_adjust(left=0.2)

  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig(name+".pdf")
  plt.close()

def plot_stacked_histogram_with_ratio(
    data_hist, 
    stack_hist_dict, 
    bin_edges, 
    data_name='Data', 
    xlabel="",
    ylabel="Events",
    name="fig", 
    data_errors=None, 
    stack_hist_errors=None, 
    use_stat_err=False,
    axis_text="",
    top_space=1.2,
  ):
  """
  Plot a stacked histogram with a ratio plot.

  Parameters
  ----------
  data_hist : array-like
      Histogram values for the data.
  stack_hist_dict : dict
      Dictionary of histogram values for stacked components.
  bin_edges : array-like
      Bin edges for the histograms.
  data_name : str, optional
      Label for the data histogram (default is 'Data').
  xlabel : str, optional
      Label for the x-axis (default is '').
  ylabel : str, optional
      Label for the y-axis (default is 'Events').
  name : str, optional
      Name of the output plot file without extension (default is 'fig').
  data_errors : array-like, optional
      Errors for the data histogram (default is None).
  stack_hist_errors : array-like, optional
      Errors for the stacked histograms (default is None).
  use_stat_err : bool, optional
      If True, use statistical errors for the data and stacked histograms (default is False).
  axis_text : str, optional
      Text to be displayed on the top left corner of the plot (default is '').
  """

  fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
  bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # Compute bin centers

  data_hist = data_hist.astype(np.float64)
  for k, v in stack_hist_dict.items():
    stack_hist_dict[k] = v.astype(np.float64)

  total_stack_hist = np.sum(list(stack_hist_dict.values()), axis=0)

  if data_errors is None:
    data_errors = 0*data_hist
  if stack_hist_errors is None:
    stack_hist_errors = 0*total_stack_hist   

  if use_stat_err:
    data_errors = np.sqrt(data_hist)
    stack_hist_errors = np.sqrt(total_stack_hist)

  # Plot the histograms on the top pad
  rgb_palette = sns.color_palette("Set2", 8)
  for ind, (k, v) in enumerate(stack_hist_dict.items()):
    if ind == 0:
      bottom = None
    else:
      bottom = stack_hist_dict[list(stack_hist_dict.keys())[ind-1]]

    ax1.bar(
       bin_edges[:-1], 
       v, 
       bottom=bottom,
       width=np.diff(bin_edges), 
       align='edge', 
       alpha=1.0, 
       label=k, 
       color=tuple(x for x in rgb_palette[ind]), 
       edgecolor=None
      )

  step_edges = np.append(bin_edges,2*bin_edges[-1]-bin_edges[-2])
  summed_stack_hist = np.zeros(len(total_stack_hist))
  for k, v in stack_hist_dict.items():
    summed_stack_hist += v
    step_histvals = np.append(np.insert(summed_stack_hist,0,0.0),0.0)
    ax1.step(step_edges, step_histvals, color='black')

  ax1.set_xlim([bin_edges[0],bin_edges[-1]])
  ax1.set_ylim([0.0,top_space*max(np.maximum(data_hist,total_stack_hist))])

  ax1.fill_between(bin_edges[:],np.append(total_stack_hist,total_stack_hist[-1])-np.append(stack_hist_errors,stack_hist_errors[-1]),np.append(total_stack_hist,total_stack_hist[-1])+np.append(stack_hist_errors,stack_hist_errors[-1]),color="gray",alpha=0.3,step='post',label="Uncertainty")

  # Plot the other histogram as markers with error bars
  ax1.errorbar(bin_centers, data_hist, yerr=data_errors, fmt='o', label=data_name, color="black")

  # Get the current handles and labels of the legend
  handles, labels = ax1.get_legend_handles_labels()

  # Reverse the order of handles and labels
  handles = handles[::-1]
  labels = labels[::-1]

  legend = ax1.legend(handles, labels, loc='upper right', fontsize=18, bbox_to_anchor=(0.9, 0.88), bbox_transform=plt.gcf().transFigure, frameon=True, framealpha=1, facecolor='white', edgecolor="white")

  # Set legend width and wrap text manually
  legend.get_frame().set_linewidth(0)  # Remove legend box border
  legend.get_frame().set_facecolor('none')  # Make legend background transparent
  legend.get_frame().set_edgecolor('none')  # Make legend edge transparent

  max_label_length = 15  # Adjust the maximum length of each legend label
  for text in legend.get_texts():
    text.set_text(textwrap.fill(text.get_text(), max_label_length))

  ax1.set_ylabel(ylabel)
  hep.cms.text(cms_label,ax=ax1)

  ax1.text(1.0, 1.0, lumi_label,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax1.transAxes)

  ax1.text(0.03, 0.96, axis_text, transform=ax1.transAxes, va='top', ha='left')

  # Compute the ratio of the histograms
  zero_indices = np.where(total_stack_hist == 0)
  for i in zero_indices: total_stack_hist[i] = 1.0

  ratio = np.divide(data_hist,total_stack_hist)
  ratio_errors_1 = np.divide(stack_hist_errors,total_stack_hist)
  ratio_errors_2 = np.divide(data_errors,total_stack_hist)

  for i in zero_indices:
      ratio[i] = 0.0
      ratio_errors_1[i] = 0.0
      ratio_errors_2[i] = 0.0

  # Plot the ratio on the bottom pad
  ax2.errorbar(bin_centers, ratio, fmt='o', yerr=ratio_errors_2, label=data_name, color="black")

  ax2.axhline(y=1, color='black', linestyle='--')  # Add a horizontal line at ratio=1
  ax2.fill_between(bin_edges,1-np.append(ratio_errors_1,ratio_errors_1[-1]),1+np.append(ratio_errors_1,ratio_errors_1[-1]),color="gray",alpha=0.3,step='post')
  ax2.set_xlabel(xlabel)
  ax2.set_ylabel('Ratio')
  ax2.set_ylim([0.5,1.5])
  ax2.xaxis.get_major_formatter().set_useOffset(False)

  # Adjust spacing between subplots
  plt.subplots_adjust(hspace=0.1, left=0.15)

  # Show the plot
  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig(name+".pdf")
  plt.close()

def plot_stacked_unrolled_2d_histogram_with_ratio(
    data_hists, 
    stack_hists_dict, 
    bin_edges_1d,
    unrolled_bins,
    unrolled_bin_name,
    data_name='Data', 
    xlabel="",
    ylabel="Events",
    name="fig", 
    data_hists_errors=None, 
    stack_hists_errors=None, 
    use_stat_err=False,
    axis_text="",
    sf_diff=2,
  ):
  """
  Plot a stacked histogram with a ratio plot for 2D unrolled histogram.

  Parameters
  ----------
  data_hists : list of array-like
      Histogram values for the data.
  stack_hists_dict : dict
      Dictionary of histogram values for stacked components.
  bin_edges_1d : array-like
      Bin edges for the 1D histograms.
  unrolled_bins : list
      Bin edges for the unrolled dimension.
  unrolled_bin_name : str
      Name of the unrolled dimension.
  data_name : str, optional
      Label for the data histogram (default is 'Data').
  xlabel : str, optional
      Label for the x-axis (default is '').
  ylabel : str, optional
      Label for the y-axis (default is 'Events').
  name : str, optional
      Name of the output plot file without extension (default is 'fig').
  data_hists_errors : list of array-like, optional
      Errors for the data histogram (default is None).
  stack_hists_errors : list of array-like, optional
      Errors for the stacked histograms (default is None).
  use_stat_err : bool, optional
      If True, use statistical errors for the data and stacked histograms (default is False).
  axis_text : str, optional
      Text to be displayed on the bottom left corner of the plot (default is '').
  """

  fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(15,10))

  bin_edges = []
  for unrolled_ind in range(len(data_hists)):
    for plot_ind in range(len(bin_edges_1d)-1):
      bin_edges.append(((bin_edges_1d[-1]-bin_edges_1d[0])*unrolled_ind) + (bin_edges_1d[plot_ind] - bin_edges_1d[0]))
  bin_edges.append((len(data_hists))*(bin_edges_1d[-1]-bin_edges_1d[0]))
  bin_edges = [be/(bin_edges_1d[-1]-bin_edges_1d[0]) for be in bin_edges]
  bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # Compute bin centers

  # combine data_hists
  for ind, dh in enumerate(data_hists):
    if ind == 0:
      data_hist = copy.deepcopy(dh)
    else:
      data_hist = np.concatenate((data_hist, dh))

  # combine data errors
  if data_hists_errors is not None:
    for ind, dh in enumerate(data_hists_errors):
      if ind == 0:
        data_errors = copy.deepcopy(dh)
      else:
        data_errors = np.concatenate((data_errors, dh))

  # combine stack_hists
  stack_hist_dict = {}
  for k, v in stack_hists_dict.items():
    for ind, hist in enumerate(v):
      if ind == 0:
        stack_hist_dict[k] = copy.deepcopy(hist)
      else:
        stack_hist_dict[k] = np.concatenate((stack_hist_dict[k], hist))
        
  # combine stack_hists_errors
  if stack_hists_errors is not None:
    for ind, hist in enumerate(stack_hists_errors):
      if ind == 0:
        stack_hist_errors = copy.deepcopy(hist)
      else:
        stack_hist_errors = np.concatenate((stack_hist_errors, hist))

  data_hist = data_hist.astype(np.float64)
  for k, v in stack_hist_dict.items():
    stack_hist_dict[k] = v.astype(np.float64)

  total_stack_hist = np.sum(list(stack_hist_dict.values()), axis=0)

  if data_errors is None:
      data_errors = 0*data_hist
  if stack_hist_errors is None:
      stack_hist_errors = 0*total_stack_hist   

  if use_stat_err:
      data_errors = np.sqrt(data_hist)
      stack_hist_errors = np.sqrt(total_stack_hist)

  # Plot the histograms on the top pad
  rgb_palette = sns.color_palette("Set2", 8)
  for ind, (k, v) in enumerate(stack_hist_dict.items()):
    if ind == 0:
      bottom = None
    else:
       bottom = stack_hist_dict[list(stack_hist_dict.keys())[ind-1]]

    ax1.bar(
       bin_edges[:-1], 
       v, 
       bottom=bottom,
       width=np.diff(bin_edges), 
       align='edge', 
       alpha=1.0, 
       label=k, 
       color=tuple(x for x in rgb_palette[ind]), 
       edgecolor=None
      )

  step_edges = np.append(bin_edges,2*bin_edges[-1]-bin_edges[-2])
  summed_stack_hist = np.zeros(len(total_stack_hist))
  for k, v in stack_hist_dict.items():
    summed_stack_hist += v
    step_histvals = np.append(np.insert(summed_stack_hist,0,0.0),0.0)
    ax1.step(step_edges, step_histvals, color='black')

  ax1.set_xlim([bin_edges[0],bin_edges[-1]])
  ax1.set_ylim(0, 1.2*max(np.concatenate((total_stack_hist,data_hist))))

  ax1.fill_between(bin_edges[:],np.append(total_stack_hist,total_stack_hist[-1])-np.append(stack_hist_errors,stack_hist_errors[-1]),np.append(total_stack_hist,total_stack_hist[-1])+np.append(stack_hist_errors,stack_hist_errors[-1]),color="gray",alpha=0.3,step='post',label="Uncertainty")

  # Plot the other histogram as markers with error bars
  ax1.errorbar(bin_centers, data_hist, yerr=data_errors, fmt='o', label=data_name, color="black")

  # Draw lines showing unrolled bin splits
  for i in range(1,len(data_hists)):
    ax1.axvline(x=i, color='black', linestyle='-', linewidth=4)

  # Draw text showing unrolled bin splits
  text_y = 1.1*max(np.concatenate((total_stack_hist,data_hist)))
  unrolled_bin_name = unrolled_bin_name.replace("_","\_")
  for i in range(len(data_hists)):
    text_x = i + 0.5

    unrolled_bin = [unrolled_bins[i], unrolled_bins[i+1]]
    unrolled_bin = [int(i) if i.is_integer() else i for i in unrolled_bin]

    significant_figures = sf_diff - int(np.floor(np.log10(abs(unrolled_bin[-1]-unrolled_bin[0])))) - 1

    if unrolled_bin[0] == -np.inf:
      unrolled_bin_string = rf"${unrolled_bin_name} < {RoundToSF(unrolled_bin[1],significant_figures)}$"
    elif unrolled_bin[1] == np.inf:
      unrolled_bin_string = rf"${unrolled_bin_name} \geq {RoundToSF(unrolled_bin[0],significant_figures)}$"
    else:
      unrolled_bin_string = rf"${RoundToSF(unrolled_bin[0],significant_figures)} \leq {unrolled_bin_name} < {RoundToSF(unrolled_bin[1],significant_figures)}$"

    ax1.text(text_x, text_y, unrolled_bin_string, verticalalignment='center', horizontalalignment='center', fontsize=14)

  # Change x axis labels
  x_locator = ticker.MaxNLocator(integer=True, nbins=3, prune='both', min_n_ticks=3)
  x_tick_positions = x_locator.tick_values(bin_edges_1d[0], bin_edges_1d[-1])
  new_tick_positions = []
  new_tick_labels = []
  for unrolled_ind in range(len(data_hists)):
    for plot_ind in range(len(x_tick_positions)):
      new_tick_positions.append(unrolled_ind + ((x_tick_positions[plot_ind]-bin_edges_1d[0])/(bin_edges_1d[-1] - bin_edges_1d[0])))
      new_tick_labels.append(x_tick_positions[plot_ind])
  new_tick_labels = [int(i) if i.is_integer() else i for i in new_tick_labels]

  ax1.set_xticks(new_tick_positions)
  ax1.set_xticklabels(new_tick_labels)

  minor_tick_positions = []
  for i in range(len(new_tick_positions) - 1):
    minor_tick_positions.extend(np.linspace(new_tick_positions[i], new_tick_positions[i+1], num=4, endpoint=False)[1:])

  while ((2*minor_tick_positions[0])-minor_tick_positions[1]) > 0:
    minor_tick_positions = [((2*minor_tick_positions[0])-minor_tick_positions[1])] + minor_tick_positions

  while (2*minor_tick_positions[-1]) - minor_tick_positions[-2] < len(data_hists):
    minor_tick_positions += [(2*minor_tick_positions[-1]) - minor_tick_positions[-2]]

  ax1.set_xticks(minor_tick_positions, minor=True)
  ax2.set_xticks(minor_tick_positions, minor=True)

  # Get the current handles and labels of the legend
  handles, labels = ax1.get_legend_handles_labels()

  # Reverse the order of handles and labels
  handles = handles[::-1]
  labels = labels[::-1]

  # Create the reversed legend
  legend = ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, framealpha=1, facecolor='white', edgecolor="white")

  max_label_length = 15  # Adjust the maximum length of each legend label
  for text in legend.get_texts():
    text.set_text(textwrap.fill(text.get_text(), max_label_length))

  #ax1.legend()
  ax1.set_ylabel(ylabel)
  hep.cms.text(cms_label,ax=ax1)

  ax1.text(1.0, 1.0, lumi_label,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax1.transAxes)

  ax1.text(0.03, 0.96, axis_text, transform=ax1.transAxes, va='top', ha='left')

  # Compute the ratio of the histograms
  zero_indices = np.where(total_stack_hist == 0)
  for i in zero_indices: total_stack_hist[i] = 1.0

  ratio = np.divide(data_hist,total_stack_hist)
  ratio_errors_1 = np.divide(stack_hist_errors,total_stack_hist)
  ratio_errors_2 = np.divide(data_errors,total_stack_hist)

  for i in zero_indices:
      ratio[i] = 0.0
      ratio_errors_1[i] = 0.0
      ratio_errors_2[i] = 0.0

  # Plot the ratio on the bottom pad
  ax2.errorbar(bin_centers, ratio, fmt='o', yerr=ratio_errors_2, label=data_name, color="black")

  ax2.axhline(y=1, color='black', linestyle='--')  # Add a horizontal line at ratio=1
  ax2.fill_between(bin_edges,1-np.append(ratio_errors_1,ratio_errors_1[-1]),1+np.append(ratio_errors_1,ratio_errors_1[-1]),color="gray",alpha=0.3,step='post')
  ax2.set_xlabel(xlabel)
  ax2.set_ylabel('Ratio')
  ax2.set_ylim([0.5,1.5])
  #ax2.xaxis.get_major_formatter().set_useOffset(False)

  # Draw lines showing unrolled bin splits
  for i in range(1,len(data_hists)):
  #  line_x = i*(bin_edges_1d[-1] - bin_edges_1d[0])
    ax2.axvline(x=i, color='black', linestyle='-', linewidth=4)
  ax2.set_xticks(new_tick_positions)
  ax2.set_xticklabels(new_tick_labels)

  # Adjust spacing between subplots
  plt.subplots_adjust(hspace=0.1)

  # Show the plot
  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig(name+".pdf", bbox_inches='tight')
  plt.close()

def plot_summary(
    crossings, 
    name="summary", 
    nominal_name="",
    show2sigma=False, 
    other_summaries={},
    shift = 0.2,
    subtract = False,
    text = "",
  ):
  """
  Plot a validation summary.

  Parameters
  ----------
  crossings : dict
      Dictionary of crossing points.
  name : str, optional
      Name of the output plot file (default is "validation_summary").
  nominal_name : str, optional
      Name of the nominal value (default is "").
  show2sigma : bool, optional
      Whether to show 2 sigma (default is True).
  other_summaries : dict, optional
      Other summaries to plot (default is {}).
  """

  if nominal_name != "":
    nominal_name += " "

  legend_width = 0.15
  n_pads = len(list(crossings.keys()))
  fig, ax = plt.subplots(1, n_pads+1, gridspec_kw={'width_ratios': [(1-legend_width)/n_pads]*n_pads + [legend_width]}, figsize=(12, 12))
  plt.subplots_adjust(left=0.2, right=0.9)
  hep.cms.text(cms_label,ax=ax[0])

  other_colors = sns.color_palette("bright", len(list(other_summaries.keys()))+1)

  for ind, (col, vals) in enumerate(crossings.items()):

    x = []
    y = []
    x_err_lower = []
    x_err_higher = []
    if show2sigma:
      x_2err_lower = []
      x_2err_higher = []

    other_x = {}
    other_y = {}
    other_x_err_lower = {}
    other_x_err_higher = {}
    if show2sigma:
      other_x_2err_lower = {}
      other_x_2err_higher = {}
    for k in other_summaries.keys():
      other_x[k] = []
      other_y[k] = []
      other_x_err_lower[k] = []
      other_x_err_higher[k] = []
      if show2sigma:
        other_x_2err_lower[k] = []
        other_x_2err_higher[k] = []   

    for val_ind, (key, val) in enumerate(vals.items()):

      num_lines = 1+len(list(other_summaries.keys()))
      starting_point = (-1*val_ind) + (((num_lines-1)/2)*shift)

      y.append(starting_point)
      x.append(val[0])

      for other_ind, (k, v) in enumerate(other_summaries.items()):
        other_x[k].append(v[col][key][0])
        other_y[k].append(starting_point - ((other_ind+1)*shift))

      if -1 in val.keys():
        x_err_lower.append(val[0]-val[-1])
      else:
        x_err_lower.append(0.0)

      if 1 in val.keys():
        x_err_higher.append(val[1]-val[0])
      else:
        x_err_higher.append(0.0)

      for k, v in other_summaries.items():
        if -1 in v[col][key].keys():
          other_x_err_lower[k].append(v[col][key][0]-v[col][key][-1])
        else:
          other_x_err_lower[k].append(0.0)

        if 1 in v[col][key].keys():
          other_x_err_higher[k].append(v[col][key][1]-v[col][key][0])
        else:
          other_x_err_higher[k].append(0.0)

      if show2sigma:
        if -2 in val.keys():
          x_2err_lower.append(val[0]-val[-2])
        else:
          x_2err_lower.append(0.0)

        if 2 in val.keys():
          x_2err_higher.append(val[2]-val[0])
        else:
          x_2err_higher.append(0.0)    

        for k, v in other_summaries.items():
          if -2 in v[col][key].keys():
            other_x_2err_lower[k].append(v[col][key][0]-v[col][key][-2])
          else:
            other_x_2err_lower[k].append(0.0)

          if 2 in v[col][key].keys():
            other_x_2err_higher[k].append(v[col][key][2]-v[col][key][0])
          else:
            other_x_2err_higher[k].append(0.0)

    if text is not None:
      ax[ind].text(0.98, 0.97, text[col],
          verticalalignment='top', horizontalalignment='right',
          transform=ax[ind].transAxes)

    # dummy for legend
    if len(list(other_summaries.keys())) > 0:
      colour = "black"
    else:
      colour = other_colors[0]

    if subtract:
      sign = "-"
      loc = 0.0
    else:
      sign = "/"
      loc = 1.0

    if show2sigma:
      ax[ind].errorbar([loc], [-1000], xerr=[[0.0], [0.0]], fmt='o', capsize=10, linewidth=5, color=mcolors.to_rgba(colour, alpha=0.5), label=rf"1$\sigma$ Intervals of Best Fits {sign} True Values")
      if show2sigma:
        ax[ind].errorbar([loc], [-1000], xerr=[[0.0], [0.0]], fmt='o', capsize=10, linewidth=1, color=mcolors.to_rgba(colour, alpha=1.0), label=rf"2$\sigma$ Intervals of Best Fits {sign} True Values")
    else:
      ax[ind].errorbar([loc], [-1000], xerr=[[0.0], [0.0]], fmt='o', capsize=10, linewidth=1, color=mcolors.to_rgba(colour, alpha=1.0), label=rf"1$\sigma$ Intervals of Best Fits {sign} True Values")


    if show2sigma:
      ax[ind].errorbar(x, y, xerr=[x_2err_lower, x_2err_higher], fmt='o', capsize=10, linewidth=1, color=mcolors.to_rgba(other_colors[0], alpha=0.5))
      for k_ind, k in enumerate(other_summaries.keys()):
        ax[ind].errorbar(other_x[k], other_y[k], xerr=[other_x_2err_lower[k], other_x_2err_higher[k]], fmt='o', capsize=10, linewidth=1, color=mcolors.to_rgba(other_colors[k_ind+1], alpha=0.5))
      ax[ind].errorbar(x, y, xerr=[x_err_lower, x_err_higher], fmt='o', capsize=10, linewidth=5, color=mcolors.to_rgba(other_colors[0], alpha=0.5), label=rf"{nominal_name}")
      for k_ind, k in enumerate(other_summaries.keys()):
        ax[ind].errorbar(other_x[k], other_y[k], xerr=[other_x_err_lower[k], other_x_err_higher[k]], fmt='o', capsize=10, linewidth=5, color=mcolors.to_rgba(other_colors[k_ind+1], alpha=0.5), label=rf"{k}")
    else:
      ax[ind].errorbar(x, y, xerr=[x_err_lower, x_err_higher], fmt='o', capsize=10, linewidth=2, color=mcolors.to_rgba(other_colors[0], alpha=1.0), label=rf"{nominal_name}")
      for k_ind, k in enumerate(other_summaries.keys()):
        ax[ind].errorbar(other_x[k], other_y[k], xerr=[other_x_err_lower[k], other_x_err_higher[k]], fmt='o', capsize=10, linewidth=2, color=mcolors.to_rgba(other_colors[k_ind+1], alpha=1.0), label=rf"{k}")  


    for y in range(len(list(vals.keys()))-1):
      ax[ind].axhline(y=(-1*y)-0.5, color='black', linestyle='--') 
    if not subtract:
      ax[ind].axvline(x=1, color='black', linestyle='--') 
    else:
      ax[ind].axvline(x=0, color='black', linestyle='--') 

    ax[ind].set_xlabel(col)
    if ind == 0:
      ax[ind].set_yticks([-1*i for i in range(len(list(vals.keys())))])
      ax[ind].set_yticklabels(list(vals.keys()))
    else:
      ax[ind].tick_params(labelleft=False)      
    ax[ind].set_ylim((-1*len(list(vals.keys())))+0.5, 0.5)
    ax[ind].xaxis.get_major_formatter().set_useOffset(False)

  ax[-2].text(1.0, 1.0, lumi_label,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax[-2].transAxes)

  # Legend
  ax[-1].spines['top'].set_visible(False)
  ax[-1].spines['right'].set_visible(False)
  ax[-1].spines['left'].set_visible(False)
  ax[-1].spines['bottom'].set_visible(False)
  ax[-1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)

  handles, labels = ax[0].get_legend_handles_labels()

  if show2sigma:
    legend_upper = ax[-1].legend(handles[:2], labels[:2], loc='upper left', frameon=False, framealpha=1, facecolor='white', edgecolor="black", bbox_to_anchor=(-0.6, 1))
    plt.gca().add_artist(legend_upper)
    if len(list(other_summaries.keys())) > 0:
      new_handles = []
      for handle in handles:
        new_handles.append(handle.lines[0])
      legend = ax[-1].legend(new_handles[2:], labels[2:], loc='upper left', frameon=False, framealpha=1, facecolor='white', edgecolor="black", bbox_to_anchor=(-0.6, 0.6))
  else:
    legend_upper = ax[-1].legend(handles[:1], labels[:1], loc='upper left', frameon=False, framealpha=1, facecolor='white', edgecolor="black", bbox_to_anchor=(-0.6, 1))
    plt.gca().add_artist(legend_upper)
    if len(list(other_summaries.keys())) > 0:
      new_handles = []
      for handle in handles:
        new_handles.append(handle.lines[0])
      legend = ax[-1].legend(new_handles[1:], labels[1:], loc='upper left', frameon=False, framealpha=1, facecolor='white', edgecolor="black", bbox_to_anchor=(-0.6, 0.8))
  
  max_label_length = 14  # Adjust the maximum length of each legend label
  if len(list(other_summaries.keys())) > 0:
    for text in legend.get_texts():
      text.set_text(textwrap.fill(text.get_text(), max_label_length))

  for text in legend_upper.get_texts():
    text.set_text(textwrap.fill(text.get_text(), max_label_length))

  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig(name+".pdf")
  plt.close()