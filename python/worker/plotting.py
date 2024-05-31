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