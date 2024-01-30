import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import mplhep as hep
import seaborn as sns
import pandas as pd
import copy
import textwrap
from other_functions import MakeDirectories

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
    drawstyle = "default"
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

  plt.xlabel(x_label)
  plt.ylabel(y_label)
  if not all(item is None for item in hist_names):
    plt.legend()
  plt.tight_layout()
  MakeDirectories(name+".pdf")
  plt.savefig(name+".pdf")
  print("Created {}.pdf".format(name))
  plt.close()

def plot_histogram_with_ratio(
    hist_values1, 
    hist_values2, 
    bin_edges, 
    name_1='Histogram 1', 
    name_2='Histogram 2',
    xlabel="",
    name="fig", 
    errors_1=None, 
    errors_2=None, 
    title_right="",
    density=False,
    use_stat_err=False,
  ):
  """
  Plot two histograms along with their ratio.

  Parameters:
      hist_values1 (array-like): Values for the first histogram.
      hist_values2 (array-like): Values for the second histogram.
      bin_edges (array-like): Bin edges.
      name_1 (str, optional): Name for the first histogram.
      name_2 (str, optional): Name for the second histogram.
      xlabel (str, optional): Label for the x-axis.
      name (str, optional): Name of the output file (without extension). Defaults to "fig".
      errors_1 (array-like, optional): Errors for the first histogram.
      errors_2 (array-like, optional): Errors for the second histogram.
      title_right (str, optional): Text to be displayed at the top-right corner of the plot.
      density (bool, optional): If True, normalize histograms to unit area. Defaults to False.
      use_stat_err (bool, optional): If True, use square root of hist_values as errors. Defaults to False.
  """
  fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
  bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # Compute bin centers

  hist_values1 = hist_values1.astype(np.float64)
  hist_values2 = hist_values2.astype(np.float64)

  if errors_1 is None:
      errors_1 = 0*hist_values1
  if errors_2 is None:
      errors_2 = 0*hist_values2   

  if use_stat_err:
      errors_1 = np.sqrt(hist_values1)
      errors_2 = np.sqrt(hist_values2)

  if density:
      hist1_norm = hist_values1.sum()
      hist2_norm = hist_values2.sum()
      hist_values1 *= 1/hist1_norm
      hist_values2 *= 1/hist2_norm
      errors_1 *= 1/hist1_norm
      errors_2 *= 1/hist2_norm

  # Plot the histograms on the top pad
  ax1.bar(bin_edges[:-1], hist_values1, width=np.diff(bin_edges), align='edge', alpha=1.0, label=name_1, color=(248/255,206/255,104/255), edgecolor=None)
  step_edges = np.append(bin_edges,2*bin_edges[-1]-bin_edges[-2])
  step_histvals = np.append(np.insert(hist_values1,0,0.0),0.0)
  ax1.step(step_edges, step_histvals, color='black')
  ax1.set_xlim([bin_edges[0],bin_edges[-1]])
  ax1.fill_between(bin_edges[:],np.append(hist_values1,hist_values1[-1])-np.append(errors_1,errors_1[-1]),np.append(hist_values1,hist_values1[-1])+np.append(errors_1,errors_1[-1]),color="gray",alpha=0.3,step='post',label="Uncertainty")

  # Plot the other histogram as markers with error bars
  ax1.errorbar(bin_centers, hist_values2, yerr=errors_2, fmt='o', label=name_2, color="black")

  # Get the current handles and labels of the legend
  handles, labels = ax1.get_legend_handles_labels()

  # Reverse the order of handles and labels
  handles = handles[::-1]
  labels = labels[::-1]

  # Create the reversed legend
  ax1.legend(handles, labels)

  #ax1.legend()
  ax1.set_ylabel('Density')
  hep.cms.text("Work in progress",ax=ax1)

  ax1.text(1.0, 1.0, title_right,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax1.transAxes)

  # Compute the ratio of the histograms
  zero_indices = np.where(hist_values1 == 0)
  for i in zero_indices: hist_values1[i] = 1.0

  ratio = np.divide(hist_values2,hist_values1)
  ratio_errors_1 = np.divide(errors_1,hist_values1)
  ratio_errors_2 = np.divide(errors_2,hist_values1)

  for i in zero_indices:
      ratio[i] = 0.0
      ratio_errors_1[i] = 0.0
      ratio_errors_2[i] = 0.0

  # Plot the ratio on the bottom pad
  ax2.errorbar(bin_centers, ratio, fmt='o', yerr=ratio_errors_2, label=name_2, color="black")

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

def plot_likelihood(
    x, 
    y, 
    crossings, 
    name="lkld", 
    xlabel="", 
    true_value=None, 
    cap_at=9, 
    other_lklds={}, 
    label="", 
    title_right=""
  ):
  """
  Plot likelihood curve.

  Parameters:
      x (array-like): X-axis values.
      y (array-like): Y-axis values.
      crossings (dict): Dictionary containing special points in the likelihood curve.
      name (str, optional): Name of the output file (without extension). Defaults to "lkld".
      xlabel (str, optional): Label for the x-axis.
      true_value (float, optional): True value to be marked on the plot.
      cap_at (float, optional): Cap the y-axis at this value. Defaults to 9.
      other_lklds (dict, optional): Additional likelihood curves to be overlaid.
      label (str, optional): Label for the likelihood curve.
      title_right (str, optional): Text to be displayed at the top-right corner of the plot.
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
  hep.cms.text("Work in progress",ax=ax)
  plt.plot(x, y, label=label)

  colors = ["green", "red", "blue"]
  ind = 0
  for k, v in other_lklds.items():
     plt.plot(x, np.array(v)[np.array(sel_inds)], label=k, color=colors[ind])
     ind += 1

  if true_value != None:
    plt.plot([true_value,true_value], [0,y_max], linestyle='--', color='black')

    ax.text(1.0, 1.0, title_right,
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
  
  plt.legend(loc='upper right')

  if -1 in crossings.keys() and 1 in crossings.keys():
    text = f'Result: {round(crossings[0],2)} + {round(crossings[1]-crossings[0],2)} - {round(crossings[0]-crossings[-1],2)}'
    ax.text(0.03, 0.96, text, transform=ax.transAxes, va='top', ha='left')

  plt.xlim(x[0],x[-1])
  plt.ylim(0,y_max)
  plt.xlabel(xlabel)
  plt.ylabel(r'$-2\Delta \ln L$')
  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig("{}.pdf".format(name))
  plt.close()


def plot_2d_likelihood(
    x, 
    y, 
    z, 
    name="lkld_2d", 
    xlabel="", 
    ylabel="", 
    best_fit=None, 
    true_value=None, 
    title_right=""
  ):
  """
  Plot 2D likelihood surface.

  Parameters:
      x (array-like): X-axis values.
      y (array-like): Y-axis values.
      z (array-like): Z-axis values (likelihood).
      name (str, optional): Name of the output file (without extension). Defaults to "lkld_2d".
      xlabel (str, optional): Label for the x-axis.
      ylabel (str, optional): Label for the y-axis.
      best_fit (tuple, optional): Coordinates of the best-fit point.
      true_value (tuple, optional): Coordinates of the true value to be marked on the plot.
      title_right (str, optional): Text to be displayed at the top-right corner of the plot.
  """
  fig, ax = plt.subplots()
  hep.cms.text("Work in progress",ax=ax)

  plt.scatter([best_fit[0]], [best_fit[1]], marker='+', s=400, color='Black', label='Best Fit')
  plt.scatter([true_value[0]], [true_value[1]], marker='x', s=400, color='Black', label='True Value')

  c1 = (204/255,204/255,255/255)
  c2 = (153/255,153/255,204/255)
  cmap1 = mcolors.LinearSegmentedColormap.from_list("custom", [c1, "white"], N=256)
  cmap2 = mcolors.LinearSegmentedColormap.from_list("custom", [c2, "white"], N=256)
  plt.contourf(x, y, z, levels=[0.0, 5.99], cmap=cmap1)
  plt.contourf(x, y, z, levels=[0.0, 2.28], cmap=cmap2)
  plt.contour(x, y, z, levels=[0.0, 5.99], colors=["black"])
  plt.contour(x, y, z, levels=[0.0, 2.28], colors=["black"])

  legend_handles = [
      Line2D([0], [0], color=c1, lw=15, label='95% CL'),
      Line2D([0], [0], color=c2, lw=15, label='68% CL')
  ]
  
  scatter1 = plt.scatter([best_fit[0]], [best_fit[1]], marker='+', s=400, color='Black', label='Best Fit')
  scatter2 = plt.scatter([true_value[0]], [true_value[1]], marker='x', s=400, color='Black', label='True Value')

  legend_handles.append(scatter1)
  legend_handles.append(scatter2)

  plt.legend(handles=legend_handles)

  ax.text(1.0, 1.0, title_right,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax.transAxes)

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig("{}.pdf".format(name))
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
    title_right="",
    use_stat_err=False,
    axis_text="",
  ):

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

    if unrolled_bin[0] == -np.inf:
      unrolled_bin_string = rf"${unrolled_bin_name} < {unrolled_bin[1]}$"
    elif unrolled_bin[1] == np.inf:
      unrolled_bin_string = rf"${unrolled_bin_name} \geq {unrolled_bin[0]}$"
    else:
      unrolled_bin_string = rf"${unrolled_bin[0]} \leq {unrolled_bin_name} < {unrolled_bin[1]}$"

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

def plot_correlation_matrix(correlation_matrix, labels, name="correlation_matrix", title_right=""):
   
  fig, ax = plt.subplots()
  hep.cms.text("Work in progress",ax=ax)

  corr_df = pd.DataFrame(correlation_matrix, columns=labels, index=labels)
  sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5, fmt='.2f', annot_kws={'size': 8})

  ax.text(1.0, 1.0, title_right,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax.transAxes)

  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)

  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig(name+".pdf")
  plt.close()