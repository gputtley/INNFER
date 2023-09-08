import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import copy

hep.style.use("CMS")
   
def plot_histogram_with_ratio(
      hist_values1, 
      hist_values2, 
      bin_edges, 
      name_1='Histogram 1', 
      name_2='Histogram 2',
      xlabel="",
      name="fig", 
      errors_1=0, 
      errors_2=0, 
      title_right="",
      density=False,
      use_stat_err=False,
      ):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2  # Compute bin centers

    hist_values1 = hist_values1.astype(np.float64)
    hist_values2 = hist_values2.astype(np.float64)

    if errors_1 == 0:
       errors_1 = 0*hist_values1
    if errors_2 == 0:
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
    ratio = np.divide(hist_values2,hist_values1)
    ratio_errors_1 = np.divide(errors_1,hist_values1)
    ratio_errors_2 = np.divide(errors_2,hist_values1)

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
    plt.savefig(name+".pdf")
    plt.close()

def plot_likelihood(x, y, crossings, name="lkld", xlabel="", true_value=None, cap_at_3=True):
  fig, ax = plt.subplots()
  hep.cms.text("Work in progress",ax=ax)
  plt.plot(x, y)

  if true_value != None:
    plt.plot([true_value,true_value], [0,9], linestyle='--', color='black')
    ax.text(1.0, 1.0, "True Value: {}".format(true_value),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes)

  if -1 in crossings.keys():  
    plt.plot([crossings[-1],crossings[-1]], [0,1], linestyle='--', color='orange')
  if -2 in crossings.keys():  
    plt.plot([crossings[-2],crossings[-2]], [0,4], linestyle='--', color='red')
  if 1 in crossings.keys():  
    plt.plot([crossings[1],crossings[1]], [0,1], linestyle='--', color='orange')
  if 2 in crossings.keys():  
    plt.plot([crossings[2],crossings[2]], [0,4], linestyle='--', color='red')
  plt.plot([x[0],x[-1]], [1,1], linestyle='--', color='gray')
  plt.plot([x[0],x[-1]], [4,4], linestyle='--', color='gray')
  
  plt.xlim(x[0],x[-1])
  if cap_at_3:
    plt.ylim(0,9)
  plt.xlabel(xlabel)
  plt.ylabel(r'$-2\Delta \ln L$')
  print("Created "+name+".pdf")
  plt.savefig("{}.pdf".format(name))