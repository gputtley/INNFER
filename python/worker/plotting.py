import copy
import os
import textwrap

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import mplhep as hep
import numpy as np
import seaborn as sns

from scipy.interpolate import UnivariateSpline
from useful_functions import MakeDirectories, RoundToSF

mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.use_locale'] = False

hep.style.use("CMS")

def plot_histograms(
    bins,
    hists,
    hist_names,
    colors = [i['color'] for i in plt.rcParams['axes.prop_cycle']]*100,
    linestyles = ["-"]*100,
    title_right = "",
    name = "hists",
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
    discrete=False,
    hist_errs = None,
    fill_between_bins = None,
    fill_between_hist = None,
    fill_between_step = "mid",
    fill_between_color = 'red',
    fill_between_alpha = 0.7,
    legend_right=False,  # New option to move legend to the right
    y_lim = None,
    spline_hists = False,
):
  """
  Plot histograms with optional error bars and an optional right-side legend.

  Parameters
  ----------
  legend_right : bool, optional
      If True, moves the legend to a separate subplot on the right.
  """
  
  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
  lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""

  # Create figure with two subplots if legend_right is True
  if legend_right:
      fig, (ax, ax_leg) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(12, 8))
      ax_leg.axis("off")  # Hide axis for legend-only panel
  else:
      fig, ax = plt.subplots()

  hep.cms.text(cms_label,ax=ax)

  if isinstance(drawstyle, str):
      drawstyle = [drawstyle] * 100

  if isinstance(bins, list):
      bins = np.array(bins)

  for ind, hist in enumerate(error_bar_hists):
      non_empty_bins = (hist != 0)
      ax.errorbar(
          bins[non_empty_bins], hist[non_empty_bins],
          yerr=error_bar_hist_errs[ind][non_empty_bins],
          label=error_bar_names[ind],
          markerfacecolor='none', linestyle='None', fmt='+', color=colors[ind]
      )

  if discrete:
      for i in range(len(bins)):
          hist_val = sorted([hist[i] for hist in hists])[::-1]
          for ind, hist in enumerate(hist_val):
              ax.vlines(
                  bins[i], ymin=0, ymax=hist, color=colors[ind], lw=2, 
                  label=hist_names[ind] if i == 0 else None
              )
  else:
      for ind, hist in enumerate(hists):
          if not spline_hists:
            ax.plot(
                bins, hist, label=hist_names[ind], color=colors[ind], 
                linestyle=linestyles[ind], drawstyle=drawstyle[ind]
            )
          else:
            spline = np.linspace(min(bins), max(bins), num=500)
            spline_hist = UnivariateSpline(bins, hist, s=0)
            ax.plot(spline, spline_hist(spline), label=hist_names[ind], color=colors[ind], linestyle=linestyles[ind])

  if fill_between_hist is not None:
      ax.fill_between(fill_between_bins, fill_between_hist, step=fill_between_step, 
                      alpha=fill_between_alpha, color=fill_between_color)

  if hist_errs is not None:
      for ind, uncerts in enumerate(hist_errs):
          ax.fill_between(
              bins,
              hists[ind] - uncerts,
              hists[ind] + uncerts,
              color=colors[ind],
              alpha=0.2,
              step='mid'
          )

  ax.text(1.0, 1.0, title_right,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax.transAxes)

  if y_lim is None:
    if anchor_y_at_0:
        ax.set_ylim(bottom=0, top=1.2 * max(np.max(hist) for hist in hists))
  else:
    ax.set_ylim(bottom=y_lim[0], top=y_lim[1])  

  if smooth_func is not None:
      x_func = np.linspace(min(bins), max(bins), num=200)
      y_func = [smooth_func(x) for x in x_func]
      ax.plot(x_func, y_func, label=smooth_func_name, color=smooth_func_color)

  for ind, vertical_line in enumerate(vertical_lines):
      ax.axvline(x=vertical_line, color=vertical_line_colors[ind], linestyle='--', linewidth=2, 
                  label=vertical_line_names[ind])

  ax.xaxis.get_major_formatter().set_useOffset(False)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)

  if not all(item is None for item in error_bar_names + hist_names):
      if legend_right:
          handles, labels = ax.get_legend_handles_labels()
          ax_leg.legend(handles, labels, loc="center", fontsize=10)
      else:
          ax.legend()

  plt.tight_layout()
  MakeDirectories(name+".pdf")
  plt.savefig(f"{name}.pdf")
  print(f"Created {name}.pdf")
  plt.close()


def plot_histograms_with_ratio(
  hists,
  hist_uncerts,
  hist_names,
  bins,
  xlabel = "",
  ylabel="Events",
  name="histogram_with_ratio",    
  ratio_range = [0.5,1.5],
  first_ratio = False,
  anchor_y_at_0 = True,
  axis_text = None,
  draw_error_bars = False,
  draw_error_bar_caps = True
):

  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
  lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""

  fig, ax= plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3,1]})

  hep.cms.text(cms_label,ax=ax[0],fontsize=22)

  ax[0].text(1.0, 1.0, lumi_label,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax[0].transAxes, fontsize=22)

  rgb_palette = sns.color_palette("Set2", max(8,len(hists)))

  if anchor_y_at_0:
    max_hist = None
    for hist in hists:
      if hist[0] is None and hist[1] is None:
        continue
      elif hist[0] is None:
        max_hist_ind = copy.deepcopy(np.max(hist[1]))
      elif hist[1] is None:
        max_hist_ind = copy.deepcopy(np.max(hist[0]))
      else:
        max_hist_ind = max(np.max(hist[0]),np.max(hist[1]))
      if max_hist is None:
        max_hist = copy.deepcopy(max_hist_ind)
      else:
        max_hist = max(max_hist, max_hist_ind)
    ax[0].set_ylim(bottom=0, top=1.2 * max_hist)

  # draw histograms
  legend = {}
  for ind, hist_pairs in enumerate(hists):

    colour = tuple(x for x in rgb_palette[ind])

    if first_ratio and ind == 0:
      ax[0].plot(bins[:-1], hist_pairs[1], label=hist_names[ind][1], color="black", linestyle="-", drawstyle="steps-mid")
    elif not first_ratio:
      ax[0].plot(bins[:-1], hist_pairs[1], label=hist_names[ind][1], color=colour, linestyle="-", drawstyle="steps-mid")

    if not draw_error_bars:
      ax[0].plot(bins[:-1], hist_pairs[0], label=hist_names[ind][0], color=colour, linestyle="--", drawstyle="steps-mid")
      ax[0].fill_between(
        bins,
        np.append(hist_pairs[0],hist_pairs[0][-1])-np.append(hist_uncerts[ind][0],hist_uncerts[ind][0][-1]),
        np.append(hist_pairs[0],hist_pairs[0][-1])+np.append(hist_uncerts[ind][0],hist_uncerts[ind][0][-1]),
        color=colour,
        alpha=0.2,
        step='mid'
        )

    else:
      non_empty_bins = (np.array(hist_pairs[0]) != 0)
      ax[0].errorbar(
        np.array(np.array(bins[:-1])[non_empty_bins]), np.array(hist_pairs[0])[non_empty_bins],
        yerr=np.array(hist_uncerts[ind][0])[non_empty_bins],
        label=hist_names[ind][0],
        linestyle='None', fmt='+' if draw_error_bar_caps else 'o', color=colour, capsize=5 if draw_error_bar_caps else 0, elinewidth=1, capthick=1
      )

    if first_ratio and (ind == 0):
      ax[0].fill_between(
        bins,
        np.append(hist_pairs[1],hist_pairs[1][-1])-np.append(hist_uncerts[ind][1],hist_uncerts[ind][1][-1]),
        np.append(hist_pairs[1],hist_pairs[1][-1])+np.append(hist_uncerts[ind][1],hist_uncerts[ind][1][-1]),
        color="black",
        alpha=0.2,
        step='mid'
        )
    elif not first_ratio:
      ax[0].fill_between(
        bins,
        np.append(hist_pairs[1],hist_pairs[1][-1])-np.append(hist_uncerts[ind][1],hist_uncerts[ind][1][-1]),
        np.append(hist_pairs[1],hist_pairs[1][-1])+np.append(hist_uncerts[ind][1],hist_uncerts[ind][1][-1]),
        color=colour,
        alpha=0.5,
        step='mid'
        )

    # y label
    ax[0].set_ylabel(ylabel)

    # legend
    handles, labels = ax[0].get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    legend = ax[0].legend(handles, labels, loc='upper right', fontsize=18, bbox_to_anchor=(0.9, 0.88), bbox_transform=plt.gcf().transFigure, frameon=True, framealpha=1, facecolor='white', edgecolor="white")
    legend.get_frame().set_linewidth(0)  # Remove legend box border
    legend.get_frame().set_facecolor('none')  # Make legend background transparent
    legend.get_frame().set_edgecolor('none')  # Make legend edge transparent
    max_label_length = 22  # Adjust the maximum length of each legend label
    for text in legend.get_texts():
      text.set_text(textwrap.fill(text.get_text(), max_label_length))

    # draw ratios
    denom = np.array([v if v !=0 else 1.0 for v in hist_pairs[1]])
    ratio = hist_pairs[0]/denom
    if not draw_error_bars:
      ax[-1].plot(bins[:-1], ratio, color=colour, linestyle="--", drawstyle="steps-mid")
      # add uncertainty ro ratio
      ratio_uncert = ((hist_uncerts[ind][0]**2 + hist_uncerts[ind][1]**2)**0.5)/denom
      ax[-1].fill_between(bins,np.append(ratio,ratio[-1])-np.append(ratio_uncert,ratio_uncert[-1]),np.append(ratio,ratio[-1])+np.append(ratio_uncert,ratio_uncert[-1]),color=colour,alpha=0.2,step='mid')
    else:
      non_empty_bins = (ratio != 0)
      ratio_uncert = hist_uncerts[ind][1]/denom

      #ax[-1].errorbar(
      #  np.array(np.array(bins[:-1])[non_empty_bins]), np.array(ratio)[non_empty_bins],
      #  yerr=np.array(((hist_uncerts[ind][0]**2 + hist_uncerts[ind][1]**2)**0.5)/denom)[non_empty_bins],
      #  markerfacecolor='none', linestyle='None', fmt='+', color=colour
      #)
      ax[-1].errorbar(
        np.array(np.array(bins[:-1])[non_empty_bins]), np.array(ratio)[non_empty_bins],
        yerr=(np.array(hist_uncerts[ind][0])/np.array(denom))[non_empty_bins],
        linestyle='None', fmt='+' if draw_error_bar_caps else 'o', color=colour, capsize=5 if draw_error_bar_caps else 0, elinewidth=1, capthick=1
      )
      ax[-1].fill_between(bins,np.ones(len(np.append(ratio,ratio[-1])))-np.append(ratio_uncert,ratio_uncert[-1]),np.ones(len(np.append(ratio,ratio[-1])))+np.append(ratio_uncert,ratio_uncert[-1]),color=colour,alpha=0.1,step='mid')
      # Draw dashed line along edges of the fill between
      #ax[-1].plot(bins, np.ones(len(bins))-np.append(ratio_uncert,ratio_uncert[-1]), color=colour, linestyle='--', linewidth=1, alpha=0.5)
      #ax[-1].plot(bins, np.ones(len(bins))+np.append(ratio_uncert,ratio_uncert[-1]), color=colour, linestyle='--', linewidth=1, alpha=0.5)



  # ratio labels
  ax[-1].axhline(y=1, color='black', linestyle='-')  # Add a horizontal line at ratio=1
  ax[-1].set_xlabel(xlabel)
  ax[-1].set_ylabel('Ratio')
  ax[-1].set_ylim([ratio_range[0],ratio_range[1]])
  ax[-1].xaxis.get_major_formatter().set_useOffset(False)
  if axis_text is not None:
    ax[0].text(0.03, 0.96, axis_text, transform=ax[0].transAxes, va='top', ha='left', fontsize=18)

  # Adjust spacing between subplots
  plt.subplots_adjust(hspace=0.1, left=0.15)

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
    cap_at=7,
    y_max=9,
    other_lklds={},
    other_crossings={},
    label=None, 
    under_result="",
    stat_syst_breakdown=False,
    spline=True,
    draw_1sigma_crossings = True,
    draw_2sigma_crossings = False,
    show_other_results = True
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

  if stat_syst_breakdown and len(other_lklds) != 1:
    raise ValueError("To do the stat and syst breakdown, exactly one other input must be provided representing the stat only scan.")

  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
  lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""

  if cap_at != None:
    x_plot = []
    y_plot = []
    for ind, i in enumerate(y):
      if i < cap_at:
        x_plot.append(x[ind])
        y_plot.append(i)
    x = x_plot
    y = y_plot
    if y_max is None:
      y_max = cap_at
  else:
    if y_max is None:
      y_max = max(y)

  fig, ax = plt.subplots()
  hep.cms.text(cms_label,ax=ax)

  if spline:
    spline_x = np.linspace(min(x), max(x), num=500)
    spline_y = UnivariateSpline(x, y, s=0)
    plt.plot(spline_x, spline_y(spline_x), label=label, color="blue")
  else:
    plt.plot(x, y, label=label, color="blue")

  ax.xaxis.get_major_formatter().set_useOffset(False)

  colors = sns.color_palette("Set2", len(list(other_lklds.keys())))
  color_ind = 0
  for k, v in other_lklds.items():

    if spline:
      spline_x = np.linspace(min(v[0]), max(v[0]), num=500)
      spline_y = UnivariateSpline(v[0], v[1], s=0)
      v[0] = spline_x
      v[1] = spline_y(spline_x)

    x_vals = [i for ind, i in enumerate(v[0]) if v[1][ind] < cap_at]
    y_vals = [i for i in v[1] if i < cap_at]

    if not stat_syst_breakdown:
      plt.plot(x_vals, y_vals, label=k, color=colors[color_ind])
      color_ind += 1
    else:
      plt.plot(x_vals, y_vals, label=k, linestyle='--', color="blue")

  if true_value != None:
    plt.plot([true_value,true_value], [0,cap_at], linestyle='--', color='black')

    ax.text(1.0, 1.0, lumi_label,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes)

  if -1 in crossings.keys() and draw_1sigma_crossings:  
    plt.plot([crossings[-1],crossings[-1]], [0,1], linestyle='--', color='orange')
  if -2 in crossings.keys() and draw_2sigma_crossings:  
    plt.plot([crossings[-2],crossings[-2]], [0,4], linestyle='--', color='orange')
  if 1 in crossings.keys() and draw_1sigma_crossings:  
    plt.plot([crossings[1],crossings[1]], [0,1], linestyle='--', color='orange')
  if 2 in crossings.keys() and draw_2sigma_crossings:  
    plt.plot([crossings[2],crossings[2]], [0,4], linestyle='--', color='orange')


  for other_key, other_val in other_crossings.items():
    if -1 in other_val.keys() and draw_1sigma_crossings:  
      plt.plot([other_val[-1],other_val[-1]], [0,1], linestyle='--', color='orange')
    if 1 in other_val.keys() and draw_1sigma_crossings:  
      plt.plot([other_val[1],other_val[1]], [0,1], linestyle='--', color='orange')
    if -2 in other_val.keys() and draw_2sigma_crossings:  
      plt.plot([other_val[-2],other_val[-2]], [0,4], linestyle='--', color='orange')
    if 2 in other_val.keys() and draw_2sigma_crossings:  
      plt.plot([other_val[2],other_val[2]], [0,4], linestyle='--', color='orange')

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

    if show_other_results and not stat_syst_breakdown and len(other_crossings) > 0:
      color = "blue"
    else:
      color = "black"

    next_position = 0.9
    if not stat_syst_breakdown:
      text = rf"{xlabel} = {round(crossings[0], decimal_places)}" \
              rf"$^{{+{round(crossings[1]-crossings[0], decimal_places)}}}_{{-{round(crossings[0]-crossings[-1], decimal_places)}}}$"
      ax.text(0.03, 0.96, text, transform=ax.transAxes, va='top', ha='left', fontsize=20, color=color)
    else:
      up_total = crossings[1]-crossings[0]
      down_total = crossings[0]-crossings[-1]
      up_stat = other_crossings[ list(other_crossings.keys())[0] ][1] - other_crossings[ list(other_crossings.keys())[0] ][0]
      down_stat = other_crossings[ list(other_crossings.keys())[0] ][0] - other_crossings[ list(other_crossings.keys())[0] ][-1]
      up_syst = (up_total**2 - up_stat**2)**0.5
      down_syst = (down_total**2 - down_stat**2)**0.5
      text_up = rf"{xlabel} = {round(crossings[0], decimal_places)}" \
              rf"$^{{+{round(crossings[1]-crossings[0], decimal_places)}}}_{{-{round(crossings[0]-crossings[-1], decimal_places)}}}$"
      text_down = rf"{xlabel} = {round(crossings[0], decimal_places)}" \
              rf"$^{{+{round(up_stat, decimal_places)}}}_{{-{round(down_stat, decimal_places)}}}$" \
              rf" (stat) " \
              rf"$^{{+{round(up_syst, decimal_places)}}}_{{-{round(down_syst, decimal_places)}}}$" \
              rf" (syst)"
      
      ax.text(0.03, 0.96, text_up, transform=ax.transAxes, va='top', ha='left', fontsize=20, color=color)
      ax.text(0.03, 0.90, text_down, transform=ax.transAxes, va='top', ha='left', fontsize=20, color=color)
      next_position -= 0.06

    if show_other_results and not stat_syst_breakdown:
      color_ind = 0
      for k, v in other_crossings.items():
        text = rf"{xlabel} = {round(v[0], decimal_places)}" \
                rf"$^{{+{round(v[1]-v[0], decimal_places)}}}_{{-{round(v[0]-v[-1], decimal_places)}}}$"
        ax.text(0.03, next_position, text, transform=ax.transAxes, va='top', ha='left', fontsize=20, color=colors[color_ind])
        color_ind += 1
        next_position -= 0.06

  if under_result != "":
    ax.text(0.03, next_position, under_result, transform=ax.transAxes, va='top', ha='left', fontsize=20)

  plt.xlim(x[0],x[-1])
  plt.ylim(0,y_max)
  plt.xlabel(xlabel)
  plt.ylabel(r'$-2\Delta \ln L$')
  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig("{}.pdf".format(name))
  plt.close()


def  plot_many_comparisons(
  sim_hists,
  synth_hists,
  sim_hist_uncerts,
  synth_hist_uncerts,
  bins,
  xlabel = "",
  ylabel="Events",
  name="generation_non_stack",       
):

  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
  lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""

  n_models = len(sim_hists.keys())
  fig, ax= plt.subplots(n_models+1, 1, sharex=True, gridspec_kw={'height_ratios': [3]*n_models+[2]})

  hep.cms.text(cms_label,ax=ax[0])

  rgb_palette = sns.color_palette("Set2", 8)

  # draw histograms
  legend = {}
  for ind, (key, val) in enumerate(sim_hists.items()):
    # plot histograms
    colour = tuple(x for x in rgb_palette[ind])
    ax[ind].plot(bins[:-1], val, label=key, color=colour, linestyle="--", drawstyle="steps-mid")
    other_key = list(synth_hists.keys())[ind]
    ax[ind].plot(bins[:-1], synth_hists[other_key], label=other_key, color=colour, linestyle="-", drawstyle="steps-mid")

    # add uncertainty
    ax[ind].fill_between(bins,np.append(val,val[-1])-np.append(sim_hist_uncerts[key],sim_hist_uncerts[key][-1]),np.append(val,val[-1])+np.append(sim_hist_uncerts[key],sim_hist_uncerts[key][-1]),color=colour,alpha=0.5,step='mid')
    ax[ind].fill_between(bins,np.append(synth_hists[other_key],synth_hists[other_key][-1])-np.append(synth_hist_uncerts[other_key],synth_hist_uncerts[other_key][-1]),np.append(synth_hists[other_key],synth_hists[other_key][-1])+np.append(synth_hist_uncerts[other_key],synth_hist_uncerts[other_key][-1]),color=colour,alpha=0.2,step='mid')

    # y label
    ax[ind].set_ylabel(ylabel)

    # legend
    handles, labels = ax[ind].get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    legend[ind] = ax[ind].legend(handles, labels, loc='upper right', fontsize=18, bbox_to_anchor=(0.95, 0.95), frameon=True, framealpha=1, facecolor='white', edgecolor="white")
    legend[ind].get_frame().set_linewidth(0)  # Remove legend box border
    legend[ind].get_frame().set_facecolor('none')  # Make legend background transparent
    legend[ind].get_frame().set_edgecolor('none')  # Make legend edge transparent
    max_label_length = 15  # Adjust the maximum length of each legend label
    for text in legend[ind].get_texts():
      text.set_text(textwrap.fill(text.get_text(), max_label_length))

    # draw ratios
    denom = np.array([v if v !=0 else 1.0 for v in val])
    ratio = synth_hists[other_key]/denom
    ax[-1].plot(bins[:-1], ratio, color=colour, linestyle="-", drawstyle="steps-mid")

    # add uncertainty ro ratio
    ratio_uncert = ((sim_hist_uncerts[key]**2 + synth_hist_uncerts[other_key]**2)**0.5)/denom
    ax[-1].fill_between(bins,np.append(ratio,ratio[-1])-np.append(ratio_uncert,ratio_uncert[-1]),np.append(ratio,ratio[-1])+np.append(ratio_uncert,ratio_uncert[-1]),color=colour,alpha=0.5,step='mid')

  # ratio labels
  ax[-1].axhline(y=1, color='black', linestyle='--')  # Add a horizontal line at ratio=1
  ax[-1].set_xlabel(xlabel)
  ax[-1].set_ylabel('Ratio')
  ax[-1].set_ylim([0.5,1.5])
  ax[-1].xaxis.get_major_formatter().set_useOffset(False)

  # Adjust spacing between subplots
  plt.subplots_adjust(hspace=0.1, left=0.15)

  # Show the plot
  print("Created "+name+".pdf")
  MakeDirectories(name+".pdf")
  plt.savefig(name+".pdf")
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

  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
  lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""

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
    stack_hist_errors_asym=None,
    use_stat_err=False,
    axis_text="",
    top_space=1.2,
    draw_ratio=True,
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

  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
  lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""

  if draw_ratio:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
  else:
    fig, ax1 = plt.subplots()

  bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # Compute bin centers

  stack_hist_dict = {k: stack_hist_dict[k] for k in reversed(list(stack_hist_dict.keys()))}

  if data_hist is not None:
    data_hist = data_hist.astype(np.float64)
  for k, v in stack_hist_dict.items():
    stack_hist_dict[k] = v.astype(np.float64)

  total_stack_hist = np.sum(list(stack_hist_dict.values()), axis=0)

  if data_hist is not None:
    if data_errors is None:
      data_errors = 0*data_hist
  if stack_hist_errors is None and stack_hist_errors_asym is None:
    stack_hist_errors = 0*total_stack_hist   

  if use_stat_err:
    if data_hist is not None:
      data_errors = np.sqrt(data_hist)
    stack_hist_errors = np.sqrt(total_stack_hist)
    stack_hist_errors_asym = None

  # Plot the histograms on the top pad
  rgb_palette = sns.color_palette("Set2", 8)
  for ind, (k, v) in enumerate(stack_hist_dict.items()):
    if ind == 0:
      bottom = None
    elif bottom is None:
      bottom = copy.deepcopy(stack_hist_dict[list(stack_hist_dict.keys())[ind-1]])
    else:
      bottom += copy.deepcopy(stack_hist_dict[list(stack_hist_dict.keys())[ind-1]])

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
  if data_hist is not None:
    ax1.set_ylim([0.0,top_space*max(np.maximum(data_hist,total_stack_hist))])
  else:
    ax1.set_ylim([0.0,top_space*max(total_stack_hist)])

  if stack_hist_errors_asym is None:
    ax1.fill_between(bin_edges[:],np.append(total_stack_hist,total_stack_hist[-1])-np.append(stack_hist_errors,stack_hist_errors[-1]),np.append(total_stack_hist,total_stack_hist[-1])+np.append(stack_hist_errors,stack_hist_errors[-1]),color="gray",alpha=0.3,step='post',label="Uncertainty")
  else:
    ax1.fill_between(bin_edges[:],np.append(total_stack_hist,total_stack_hist[-1])-np.append(stack_hist_errors_asym["down"],stack_hist_errors_asym["down"][-1]),np.append(total_stack_hist,total_stack_hist[-1])+np.append(stack_hist_errors_asym["up"],stack_hist_errors_asym["up"][-1]),color="gray",alpha=0.3,step='post',label="Uncertainty")

  if data_hist is not None:
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

  if not draw_ratio:
    ax1.set_xlabel(xlabel)

  if draw_ratio:

    # Compute the ratio of the histograms
    zero_indices = np.where(total_stack_hist == 0)
    for i in zero_indices: total_stack_hist[i] = 1.0

    if data_hist is not None:
      ratio = np.divide(data_hist,total_stack_hist)
      ratio_errors_2 = np.divide(data_errors,total_stack_hist)

    if stack_hist_errors_asym is None:
      ratio_errors_1 = np.divide(stack_hist_errors,total_stack_hist)
    else:
      ratio_errors_1_up = np.divide(stack_hist_errors_asym["up"],total_stack_hist)
      ratio_errors_1_down = np.divide(stack_hist_errors_asym["down"],total_stack_hist)

    for i in zero_indices:
      if data_hist is not None:
        ratio[i] = 0.0
        ratio_errors_2[i] = 0.0
      if stack_hist_errors_asym is None:
        ratio_errors_1[i] = 0.0
      else:
        ratio_errors_1_up[i] = 0.0
        ratio_errors_1_down[i] = 0.0

    if data_hist is not None:
      # Plot the ratio on the bottom pad
      ax2.errorbar(bin_centers, ratio, fmt='o', yerr=ratio_errors_2, label=data_name, color="black")

    ax2.axhline(y=1, color='black', linestyle='--')  # Add a horizontal line at ratio=1
    if stack_hist_errors_asym is None:
      ax2.fill_between(bin_edges,1-np.append(ratio_errors_1,ratio_errors_1[-1]),1+np.append(ratio_errors_1,ratio_errors_1[-1]),color="gray",alpha=0.3,step='post')
    else:
      ax2.fill_between(bin_edges,1-np.append(ratio_errors_1_down,ratio_errors_1_down[-1]),1+np.append(ratio_errors_1_up,ratio_errors_1_up[-1]),color="gray",alpha=0.3,step='post')

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

  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
  lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""

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
  for k in list(stack_hists_dict.keys())[::-1]:
    v = stack_hists_dict[k]
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

  bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # Compute bin centers
  ax1.fill_between(bin_edges[:],np.append(total_stack_hist,total_stack_hist[-1])-np.append(stack_hist_errors,stack_hist_errors[-1]),np.append(total_stack_hist,total_stack_hist[-1])+np.append(stack_hist_errors,stack_hist_errors[-1]),color="gray",alpha=0.3,step='post',label="Uncertainty")

  # Plot the other histogram as markers with error bars
  ax1.errorbar(bin_centers, data_hist, yerr=data_errors, fmt='o', label=data_name, color="black")

  # Draw lines showing unrolled bin splits
  for i in range(1,len(data_hists)):
    ax1.axvline(x=i, color='black', linestyle='-', linewidth=4)

  # Draw text showing unrolled bin splits
  text_y = 1.05*max(np.concatenate((total_stack_hist,data_hist)))
  unrolled_bin_name = unrolled_bin_name.replace("_","\_")
  for i in range(len(data_hists)):
    text_x = i + 0.5

    unrolled_bin = [unrolled_bins[i], unrolled_bins[i+1]]
    unrolled_bin = [int(i) if i.is_integer() else i for i in unrolled_bin]

    if "$" in unrolled_bin_name:
      unrolled_bin_name = unrolled_bin_name.replace("$","").replace("\\","")

    if unrolled_bin[0] == -np.inf:
      unrolled_bin_string = rf"${unrolled_bin_name} < {unrolled_bin[1]}$"
    elif unrolled_bin[1] == np.inf:
      unrolled_bin_string = rf"${unrolled_bin_name} \geq {unrolled_bin[0]}$"
    else:
      unrolled_bin_string = rf"${unrolled_bin[0]} \leq {unrolled_bin_name} < {unrolled_bin[1]}$"

    ax1.text(text_x, text_y, unrolled_bin_string, verticalalignment='center', horizontalalignment='center', fontsize=12)

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

  ax1.text(0.03, 0.98, axis_text, transform=ax1.transAxes, 
      va='top', ha='left', fontsize=18,
      bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

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


def plot_unrolled_2d_histogram(
    hists_dict, 
    bin_edges_1d,
    unrolled_bins,
    unrolled_bin_name,
    xlabel="",
    ylabel="Events",
    name="fig", 
    hists_errors=None, 
    use_stat_err=False,
    axis_text="",
    sf_diff=2,
  ):

  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""

  #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(15,10))

  fig, ax1 = plt.subplots(figsize=(15,10))
  hep.cms.text(cms_label,ax=ax1)

  bin_edges = []
  first_hist = hists_dict[list(hists_dict.keys())[0]]
  for unrolled_ind in range(len(first_hist)):
    for plot_ind in range(len(bin_edges_1d)-1):
      bin_edges.append(((bin_edges_1d[-1]-bin_edges_1d[0])*unrolled_ind) + (bin_edges_1d[plot_ind] - bin_edges_1d[0]))
  bin_edges.append((len(first_hist))*(bin_edges_1d[-1]-bin_edges_1d[0]))
  bin_edges = [be/(bin_edges_1d[-1]-bin_edges_1d[0]) for be in bin_edges]

  # combine hists
  hist_dict = {}
  hist_dict_errors = {}
  for k in list(hists_dict.keys())[::-1]:
    v = hists_dict[k]

    for ind, hist in enumerate(v):

      if hists_errors is None:
        hist_dict_errors[k] = 0*hists_errors[k][ind]
      if use_stat_err:
        hists_errors[k][ind] = np.sqrt(hist)

      if ind == 0:
        hist_dict[k] = copy.deepcopy(hist.astype(np.float64))
        hist_dict_errors[k] = copy.deepcopy(hists_errors[k][ind].astype(np.float64))
      else:
        hist_dict[k] = np.concatenate((hist_dict[k], hist.astype(np.float64)))
        hist_dict_errors[k] = np.concatenate((hist_dict_errors[k], hists_errors[k][ind].astype(np.float64)))


  # Plot the histograms on the pad
  rgb_palette = sns.color_palette("Set2", 8)
  step_edges = np.append(bin_edges,2*bin_edges[-1]-bin_edges[-2])
  for ind, (k, v) in enumerate(hist_dict.items()):

    if ind == 0:
      max_hist = copy.deepcopy(v)
    else:
      max_hist = np.maximum(max_hist, copy.deepcopy(v))

    ax1.step(
      step_edges, 
      np.append(np.insert(v,0,0.0),0.0), 
      where='post',
      alpha=1.0,
      label=k,
      color=tuple(x for x in rgb_palette[ind]),
    )
    ax1.fill_between(step_edges,np.append(np.insert(v,0,0.0),0.0)-np.append(np.insert(hist_dict_errors[k],0,0.0),0.0),np.append(np.insert(v,0,0.0),0.0)+np.append(np.insert(hist_dict_errors[k],0,0.0),0.0),color=tuple(x for x in rgb_palette[ind]),alpha=0.3,step='post')


  ax1.set_xlim([bin_edges[0],bin_edges[-1]])
  ax1.set_ylim(0, 1.2*max(max_hist))

  # Draw lines showing unrolled bin splits
  for i in range(1,len(first_hist)):
    ax1.axvline(x=i, color='black', linestyle='-', linewidth=4)

  # Draw text showing unrolled bin splits
  text_y = 1.05*max(max_hist)
  unrolled_bin_name = unrolled_bin_name.replace("_","\_")
  for i in range(len(first_hist)):
    text_x = i + 0.5

    unrolled_bin = [unrolled_bins[i], unrolled_bins[i+1]]
    unrolled_bin = [int(i) if i.is_integer() else i for i in unrolled_bin]

    if "$" in unrolled_bin_name:
      unrolled_bin_name = unrolled_bin_name.replace("$","").replace("\\","")

    if unrolled_bin[0] == -np.inf:
      unrolled_bin_string = rf"${unrolled_bin_name} < {unrolled_bin[1]}$"
    elif unrolled_bin[1] == np.inf:
      unrolled_bin_string = rf"${unrolled_bin_name} \geq {unrolled_bin[0]}$"
    else:
      unrolled_bin_string = rf"${unrolled_bin[0]} \leq {unrolled_bin_name} < {unrolled_bin[1]}$"

    ax1.text(text_x, text_y, unrolled_bin_string, verticalalignment='center', horizontalalignment='center', fontsize=12)

  # Change x axis labels
  x_locator = ticker.MaxNLocator(integer=True, nbins=3, prune='both', min_n_ticks=3)
  x_tick_positions = x_locator.tick_values(bin_edges_1d[0], bin_edges_1d[-1])
  new_tick_positions = []
  new_tick_labels = []
  for unrolled_ind in range(len(first_hist)):
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

  while (2*minor_tick_positions[-1]) - minor_tick_positions[-2] < len(first_hist):
    minor_tick_positions += [(2*minor_tick_positions[-1]) - minor_tick_positions[-2]]

  ax1.set_xticks(minor_tick_positions, minor=True)

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

  ax1.legend()
  ax1.set_ylabel(ylabel)
  hep.cms.text(cms_label,ax=ax1)

  ax1.text(0.03, 0.98, axis_text, transform=ax1.transAxes, 
      va='top', ha='left', fontsize=18,
      bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

  ax1.set_xticks(new_tick_positions)
  ax1.set_xticklabels(new_tick_labels)

  ax1.set_xlabel(xlabel)


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
    y_label = "",
    no_lumi = False,
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

  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
  if not no_lumi:
    lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""

  if nominal_name != "":
    nominal_name += " "

  legend_width = 0.15
  n_pads = len(list(crossings.keys()))
  fig, ax = plt.subplots(1, n_pads+1, gridspec_kw={'width_ratios': [(1-legend_width)/n_pads]*n_pads + [legend_width]}, figsize=(12, 12))
  plt.subplots_adjust(left=0.2, right=0.9)
  hep.cms.text(cms_label,ax=ax[0], fontsize=22)

  other_colors = sns.color_palette("bright", len(list(other_summaries.keys()))+1)

  #other_colors = [sns.color_palette("bright", 10)[3], sns.color_palette("bright", 10)[2], sns.color_palette("bright", 2)[1]]

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

    label = col
    if subtract:
      label = "$\Delta$" + col

    ax[ind].set_xlabel(label)
    if ind == 0:
      ax[ind].set_yticks([-1*i for i in range(len(list(vals.keys())))])
      ax[ind].set_yticklabels(list(vals.keys()))
    else:
      ax[ind].tick_params(labelleft=False)      
    ax[ind].set_ylim((-1*len(list(vals.keys())))+0.5, 0.5)
    ax[ind].xaxis.get_major_formatter().set_useOffset(False)

  if not no_lumi:
    ax[-2].text(1.0, 1.0, lumi_label,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax[-2].transAxes, fontsize=22)

  # y title
  ax[0].set_ylabel(y_label)

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


def plot_summary_per_val(
  results_with_constraints,
  results_without_constraints,
  truth_without_constraints,
  names,
  show2sigma = False,
  plot_name = f"summary_per_val.pdf",
  plot_text = None,
  ):
  """
  Plot a summary of results with and without constraints.
  Parameters
  ----------
  results_with_constraints : dict
      Dictionary of results with constraints.
  results_without_constraints : dict
      Dictionary of results without constraints.
  truth_without_constraints : dict
      Dictionary of truth values without constraints.
  names : list
      List of names for the results.
  show_2_sigma : bool, optional
      Whether to show 2 sigma (default is False).
  plot_name : str, optional
      Name of the output plot file (default is "/summary_per_val.pdf").
  """
  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
  lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""

  # Calculate the number of pads
  n_pads = 1 # legend
  n_pads += len(results_without_constraints.keys())  # results without constraints
  if len(results_with_constraints.keys()) > 0: # results with constraints
    n_pads += 1

  # Calculate the pad sizes
  pad_sizes = [1 for _ in range(len(results_without_constraints.keys()))]

  if len(results_with_constraints.keys()) > 0:
    pad_sizes += [len(results_with_constraints.keys())]

  sum_pad_sizes = sum(pad_sizes)
  pad_sizes = [x/sum_pad_sizes for x in pad_sizes]
  pad_sizes = [0.2] + pad_sizes
  sum_pad_sizes = sum(pad_sizes)
  pad_sizes = [x/sum_pad_sizes for x in pad_sizes]


  # Create the figure and axes
  fig, ax = plt.subplots(n_pads, 1, gridspec_kw={'height_ratios': pad_sizes}, figsize=(8.27, 11.69), constrained_layout=True)
  hep.cms.text(cms_label,ax=ax[0], fontsize=22)

  # Draw lumi label
  ax[0].text(1.0, 1.0, lumi_label,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax[0].transAxes, fontsize=22)

  # turn of top (legend) pad axis labels and ticks
  ax[0].tick_params(
    axis='both', 
    which='both', 
    bottom=False, 
    top=False, 
    left=False, 
    right=False,
    labelbottom=False, 
    labeltop=False, 
    labelleft=False, 
    labelright=False
    )

  # turn off y axis labels and ticks on all axis
  for i in range(1,n_pads):
    ax[i].tick_params(
      axis='both', 
      which='both', 
      bottom=True, 
      top=True, 
      left=False, 
      right=False,
      labelbottom=True, 
      labeltop=False, 
      labelleft=False, 
      labelright=False
      )

  # Draw horizontal lines on constraint pad
  if len(results_with_constraints.keys()) > 0:
    constraint_pad = n_pads - 1
    for i in range(1, len(results_with_constraints.keys())):
      ax[constraint_pad].axhline(y=i, color='black', linestyle='-')
  else:
    constraint_pad = n_pads

  # Set range of non constraint pad to be 0 and 1
  for i in range(1, constraint_pad):
    ax[i].set_ylim(0, 1)

  if len(results_with_constraints.keys()) > 0:
    # Set range of constraint pad to be 0 to len(results_with_constraints.keys())
    ax[constraint_pad].set_ylim(0, len(results_with_constraints.keys()))

  # Write y labels (names) on non constraint pads
  for i, name in enumerate(results_without_constraints.keys()):
    ax[i+1].text(-0.02, 0.5, name, verticalalignment='center', horizontalalignment='right', fontsize=16, transform=ax[i+1].transAxes)

  # Write y labels (names) on constraint pad
  if len(results_with_constraints.keys()) > 0:
    for i, name in enumerate(results_with_constraints.keys()):
      y_coord = ((1/(2*len(results_with_constraints.keys())))) + (i/(len(results_with_constraints.keys())))
      ax[constraint_pad].text(-0.02, y_coord, name, verticalalignment='center', horizontalalignment='right', fontsize=16, transform=ax[constraint_pad].transAxes)

  # Draw vertical lines on non constraint pads for the truth values
  for i in range(1, constraint_pad):
    if truth_without_constraints[list(results_without_constraints.keys())[i-1]] is not None:
      ax[i].axvline(x=truth_without_constraints[list(results_without_constraints.keys())[i-1]], color='black', linestyle='--')

  # Set x axis title for non constraint pads
  for i in range(1, constraint_pad):
    ax[i].set_xlabel(r"$\hat{\theta}$", fontsize=20)

  if len(results_with_constraints.keys()) > 0:
    # Set x axis title for constraint pad
    ax[constraint_pad].set_xlabel(r"$(\hat{\theta}-\theta_{0})/\Delta\theta$", fontsize=20)

    # Draw vertical lines on constraint pad around the truth value 0
    ax[constraint_pad].axvline(x=0, color='black', linestyle='--')

  # Find the deepest vals length
  max_len = 0
  for ind, (col, vals) in enumerate(results_without_constraints.items()):
    if len(vals) > max_len:
      max_len = len(vals)
  for ind, (col, vals) in enumerate(results_with_constraints.items()):
    if len(vals) > max_len:
      max_len = len(vals)

  # Define colours, black first
  colors = sns.color_palette("bright", max_len)[::-1]

  # Draw results without constraints 
  for ind, (col, vals) in enumerate(results_without_constraints.items()):
    x = []
    y = []
    x_err_lower = []
    x_err_higher = []
    if show2sigma:
      x_2err_lower = []
      x_2err_higher = []
    for val_ind, val in enumerate(vals[::-1]):
      x.append(val[0])
      x_err_lower.append(val[0]-val[-1] if -1 in val.keys() else 0.0)
      x_err_higher.append(val[1]-val[0] if 1 in val.keys() else 0.0)
      if show2sigma:
        x_2err_lower.append(val[0]-val[-2] if -2 in val.keys() else 0.0)
        x_2err_higher.append(val[2]-val[0] if 2 in val.keys() else 0.0)
      y.append((val_ind + 1)*(1/(len(vals)+1)))
      if show2sigma:
        ax[ind+1].errorbar([x[-1]], [y[-1]], xerr=[[x_2err_lower[-1]], [x_2err_higher[-1]]], fmt='o', capsize=10, linewidth=1, color=mcolors.to_rgba(colors[val_ind], alpha=0.5))
        ax[ind+1].errorbar([x[-1]], [y[-1]], xerr=[[x_err_lower[-1]], [x_err_higher[-1]]], fmt='o', capsize=10, linewidth=5, color=mcolors.to_rgba(colors[val_ind], alpha=0.5), label=rf"{names[col][::-1][val_ind]}")
      else:
        ax[ind+1].errorbar([x[-1]], [y[-1]], xerr=[[x_err_lower[-1]], [x_err_higher[-1]]], fmt='o', capsize=10, linewidth=2, color=mcolors.to_rgba(colors[val_ind], alpha=1.0), label=rf"{names[col][::-1][val_ind]}")
    if truth_without_constraints[col] is not None:
      if show2sigma:
        up_cap = [abs(x[i] + x_2err_higher[i] - truth_without_constraints[col]) for i in range(len(x))]
        low_cap = [abs(x[i] - x_2err_lower[i] - truth_without_constraints[col]) for i in range(len(x))]
      else:
        up_cap = [abs(x[i] + x_err_higher[i] - truth_without_constraints[col]) for i in range(len(x))]
        low_cap = [abs(x[i] - x_err_lower[i] - truth_without_constraints[col]) for i in range(len(x))]
      max_crossing = 1.2*max(max(up_cap), max(low_cap))
      ax[ind+1].set_xlim(truth_without_constraints[col] - max_crossing, truth_without_constraints[col] + max_crossing)

  # Draw results with constraints
  up_cap = []
  low_cap = []
  for ind, (col, vals) in enumerate(results_with_constraints.items()):
    x = []
    y = []
    x_err_lower = []
    x_err_higher = []
    if show2sigma:
      x_2err_lower = []
      x_2err_higher = []
    for val_ind, val in enumerate(vals[::-1]):
      x.append(val[0])
      x_err_lower.append(val[0]-val[-1] if -1 in val.keys() else 0.0)
      x_err_higher.append(val[1]-val[0] if 1 in val.keys() else 0.0)
      if show2sigma:
        x_2err_lower.append(val[0]-val[-2] if -2 in val.keys() else 0.0)
        x_2err_higher.append(val[2]-val[0] if 2 in val.keys() else 0.0)
      y.append(ind + ((val_ind + 1)/(len(vals)+1)))
      if show2sigma:
        ax[constraint_pad].errorbar([x[-1]], [y[-1]], xerr=[[x_2err_lower[-1]], [x_2err_higher[-1]]], fmt='o', capsize=10, linewidth=1, color=mcolors.to_rgba(colors[val_ind], alpha=0.5))
        ax[constraint_pad].errorbar([x[-1]], [y[-1]], xerr=[[x_err_lower[-1]], [x_err_higher[-1]]], fmt='o', capsize=10, linewidth=5, color=mcolors.to_rgba(colors[val_ind], alpha=0.5), label=rf"{names[col][::-1][val_ind]}")
      else:
        ax[constraint_pad].errorbar([x[-1]], [y[-1]], xerr=[[x_err_lower[-1]], [x_err_higher[-1]]], fmt='o', capsize=10, linewidth=2, color=mcolors.to_rgba(colors[val_ind], alpha=1.0), label=rf"{names[col][::-1][val_ind]}")
    if show2sigma:
      up_cap += [abs(x[i] + x_2err_higher[i]) for i in range(len(x))]
      low_cap += [abs(x[i] - x_2err_lower[i]) for i in range(len(x))]
    else:
      up_cap += [abs(x[i] + x_err_higher[i]) for i in range(len(x))]
      low_cap += [abs(x[i] - x_err_lower[i]) for i in range(len(x))]
  if len(up_cap) > 0 and len(low_cap) > 0:
    max_crossing = 1.2*max(max(up_cap), max(low_cap))
    ax[constraint_pad].set_xlim(-max_crossing, max_crossing)

  # Draw text on the top left corner of top axis
  if plot_text is not None:
    ax[0].text(0.03, 0.9, plot_text, transform=ax[0].transAxes, 
        va='top', ha='left', fontsize=18,
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

  # Draw legend on the top right of the top axis
  handles, labels = ax[1].get_legend_handles_labels()
  handles = handles[::-1]
  labels = labels[::-1]
  legend = ax[0].legend(handles, labels, loc='upper right', frameon=True, framealpha=1, facecolor='white', edgecolor="white", fontsize=16)

  print("Created "+plot_name+".pdf")
  MakeDirectories(plot_name+".pdf")
  plt.savefig(plot_name+".pdf")
  plt.close()

def plot_learned_nuisance_variations(
  nominal_line_hist,
  up_line_hist,
  down_line_hist,
  nominal_errorbar_hist,
  up_errorbar_hist,
  down_errorbar_hist,
  nominal_errorbar_uncert,
  up_errorbar_uncert,
  down_errorbar_uncert,
  bins,
  xlabel="",
  output_name="learned_nuisance_variation",
  ratio_range_line_vs_errorbar=(0.9,1.1),
  ratio_range_variations=(0.9,1.1),
  line_caption="Synth",
  errorbar_caption="Sim",
  axis_text = ""
):
  
  # Build 3 vertical pads
  fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=(10, 12))
  cms_label = str(os.getenv("PLOTTING_CMS_LABEL")) if os.getenv("PLOTTING_CMS_LABEL") is not None else ""
  hep.cms.text(cms_label,ax=ax1)
  lumi_label = str(os.getenv("PLOTTING_LUMINOSITY")) if os.getenv("PLOTTING_LUMINOSITY") is not None else ""
  ax1.text(1.0, 1.0, lumi_label,
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax1.transAxes, fontsize=22)

  # Put axis text in top left corner of pad
  ax1.text(0.03, 0.98, axis_text, transform=ax1.transAxes, 
      va='top', ha='left', fontsize=18,
      bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

  # Plot sim as solid lines on top pad
  bin_edges = np.append(bins, 2*bins[-1]-bins[-2])
  ax1.step(
    bin_edges, 
    np.append(np.insert(nominal_line_hist,0,0.0),0.0), 
    where='pre',
    alpha=1.0,
    label=f"{line_caption} Nominal",
    color='black',
  )
  ax1.step(
    bin_edges, 
    np.append(np.insert(up_line_hist,0,0.0),0.0), 
    where='pre',
    alpha=1.0,
    label=f"{line_caption} Up",
    color='red',
  )
  ax1.step(
    bin_edges, 
    np.append(np.insert(down_line_hist,0,0.0),0.0), 
    where='pre',
    alpha=1.0,
    label=f"{line_caption} Down",
    color='blue',
  )

  # Plot asimov as error bars
  bin_centers = (bins[:-1] + bins[1:])/2
  ax1.errorbar(
    bin_centers, 
    nominal_errorbar_hist, 
    yerr=nominal_errorbar_uncert, 
    fmt='o', 
    color='black', 
    label=f"{errorbar_caption} Nominal",
    capsize=5,
  )
  ax1.errorbar(
    bin_centers, 
    up_errorbar_hist, 
    yerr=up_errorbar_uncert, 
    fmt='o', 
    color='red', 
    label=f"{errorbar_caption} Up",
    capsize=5,
  )
  ax1.errorbar(
    bin_centers, 
    down_errorbar_hist, 
    yerr=down_errorbar_uncert, 
    fmt='o', 
    color='blue', 
    label=f"{errorbar_caption} Down",
    capsize=5,
  )

  ax1.set_ylabel("Events")
  ax1.legend(fontsize=18, loc='upper right')

  # Set y range as 1.1 times the max of all histograms
  max_y = max(
    max(nominal_line_hist),
    max(up_line_hist),
    max(down_line_hist),
    max(nominal_errorbar_hist + nominal_errorbar_uncert),
    max(up_errorbar_hist + up_errorbar_uncert),
    max(down_errorbar_hist + down_errorbar_uncert),
  )
  ax1.set_ylim(0, 1.1*max_y)


  # Draw ratio of sim to asimov on middle pad
  ax2.errorbar(
    bin_centers, 
    nominal_errorbar_hist/nominal_line_hist, 
    yerr=(nominal_errorbar_uncert/nominal_line_hist), 
    fmt='o', 
    color='black', 
    capsize=5,
  )
  ax2.errorbar(
    bin_centers, 
    up_errorbar_hist/up_line_hist, 
    yerr=(up_errorbar_uncert/up_line_hist), 
    fmt='o', 
    color='red', 
    capsize=5,
  )
  ax2.errorbar(
    bin_centers, 
    down_errorbar_hist/down_line_hist, 
    yerr=(down_errorbar_uncert/down_line_hist), 
    fmt='o', 
    color='blue', 
    capsize=5,
  )
  ax2.axhline(y=1.0, color='black', linestyle='--')

  ax2.set_ylabel(f"$\\frac{{\\mathrm{{{errorbar_caption}}}}}{{\\mathrm{{{line_caption}}}}}$")
  ax2.set_ylim(ratio_range_line_vs_errorbar)


  # Draw ratios of up/down to nominal on bottom pad for sim (lines) and synth (points)
  ax3.step(
    bin_edges, 
    np.append(np.insert(up_line_hist/nominal_line_hist,0,1.0),1.0), 
    where='pre',
    alpha=1.0,
    color='red',
  )
  ax3.step(
    bin_edges, 
    np.append(np.insert(down_line_hist/nominal_line_hist,0,1.0),1.0), 
    where='pre',
    alpha=1.0,
    color='blue',
  )
  ax3.errorbar(
    bin_centers, 
    up_errorbar_hist/nominal_errorbar_hist, 
    yerr=up_errorbar_uncert/nominal_errorbar_hist, 
    fmt='o', 
    color='red', 
    capsize=5,
  )
  ax3.errorbar(
    bin_centers, 
    down_errorbar_hist/nominal_errorbar_hist, 
    yerr=down_errorbar_uncert/nominal_errorbar_hist, 
    fmt='o', 
    color='blue', 
    capsize=5,
  )
  ax3.axhline(y=1.0, color='black', linestyle='--')
  #ax3.set_ylabel("Variation / Nominal", fontsize=18)
  ax3.set_ylabel(f"$\\frac{{\\mathrm{{Variation}}}}{{\\mathrm{{Nominal}}}}$")
  ax3.set_xlabel(xlabel)
  ax3.set_ylim(ratio_range_variations)

  # save figure
  print("Created "+output_name+".pdf")
  MakeDirectories(output_name+".pdf")
  plt.savefig(output_name+".pdf", bbox_inches='tight')
  plt.close()