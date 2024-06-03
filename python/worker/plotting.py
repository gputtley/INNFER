import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
from matplotlib import gridspec
import mplhep as hep
import seaborn as sns
import pandas as pd
import copy
import textwrap
import os
from useful_functions import MakeDirectories

hep.style.use("CMS")

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

  Parameters:
      bins (array-like): Bin edges.
      hists (list of array-like): List of histogram values.
      hist_names (list of str): Names for each histogram.
      colors (list of str, optional): Colors for each histogram. Defaults to Matplotlib color cycle.
      linestyles (list of str, optional): Linestyles for each histogram. Defaults to solid line.
      title_right (str, optional): Text to be displayed at the top-right corner of the plot.
      name (str, optional): Name of the output file (without extension). Defaults to "hists.pdf".
      x_label (str, optional): Label for the x-axis.
      y_label (str, optional): Label for the y-axis.
      error_bar_hists (list of array-like, optional): List of histograms for error bars.
      error_bar_hist_errs (list of array-like, optional): List of errors for each error bar histogram.
      error_bar_names (list of str, optional): Names for each error bar histogram.
      anchor_y_at_0 (bool, optional): If True, anchor the y-axis at 0. Defaults to False.
      drawstyle (str, optional): Drawstyle for the histograms. Defaults to "default".
  """
  fig, ax = plt.subplots()
  hep.cms.text("Work in progress",ax=ax)

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

  plt.xlabel(x_label)
  plt.ylabel(y_label)
  if not all(item is None for item in hist_names+error_bar_names):
    plt.legend()
  plt.tight_layout()
  MakeDirectories(name+".pdf")
  plt.savefig(name+".pdf")
  print("Created {}.pdf".format(name))
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
    title_right="",
    use_stat_err=False,
    axis_text="",
  ):
  """
  Plot a stacked histogram with a ratio plot.

  Args:
      data_hist (array-like): Histogram values for the data.
      stack_hist_dict (dict): Dictionary of histogram values for stacked components.
      bin_edges (array-like): Bin edges for the histograms.
      data_name (str, optional): Label for the data histogram (default is 'Data').
      xlabel (str, optional): Label for the x-axis (default is '').
      ylabel (str, optional): Label for the y-axis (default is 'Events').
      name (str, optional): Name of the output plot file without extension (default is 'fig').
      data_errors (array-like, optional): Errors for the data histogram (default is None).
      stack_hist_errors (array-like, optional): Errors for the stacked histograms (default is None).
      title_right (str, optional): Text to be displayed on the upper right corner of the plot (default is '').
      use_stat_err (bool, optional): If True, use statistical errors for the data and stacked histograms (default is False).
      axis_text (str, optional): Text to be displayed on the bottom left corner of the plot (default is '').
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
  hep.cms.text("Work in progress",ax=ax1)

  ax1.text(1.0, 1.0, title_right,
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

  # Adjust spacing between subplots
  plt.subplots_adjust(hspace=0.1)

  # Show the plot
  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig(name+".pdf")
  plt.close()