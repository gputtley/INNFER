import numpy as np

from asym_log_normal import AsymLogNormal

def fsr_weight(
    df,
    fsr_value=None,
    wt_name="wt",
    nuisance_column_name=None,
    top_process_mask=False,
    shift_nu = 0.0,
    divide_by_shift = False
  ):

  if fsr_value is not None:
    df["log_fsr"] = np.log(fsr_value)
  
  if nuisance_column_name is not None:
    nu = df[nuisance_column_name]
  else:
    nu = df["log_fsr"]/np.log(2)

  nu += shift_nu

  lower_clip = 0.5
  higher_clip = 2.0
  df["GenWeights_isr1fsr0p5"] = df["GenWeights_isr1fsr0p5"].clip(lower=lower_clip, upper=higher_clip)
  df["GenWeights_isr1fsr2"] = df["GenWeights_isr1fsr2"].clip(lower=lower_clip, upper=higher_clip)

  if not top_process_mask:
    mask = df["top_process"] == 1
  else:
    mask = np.ones(len(df), dtype=bool)

  asymln = AsymLogNormal(nu[mask], kp=df["GenWeights_isr1fsr2"][mask], km=df["GenWeights_isr1fsr0p5"][mask])

  if divide_by_shift:
    asymln /= AsymLogNormal(shift_nu, kp=higher_clip, km=lower_clip)

  df.loc[mask, wt_name] *= asymln

  return df


def three_point_variation_weight(
    df,
    nuisance_column_name = None,
    wt_name = "wt",
    mask_sample_m1 = None,
    mask_sample_nominal = None,
    mask_sample_p1 = None,
    total_mask = None
  ):

  # Get total mask
  if total_mask is not None:
    total_mask = df.query(total_mask).index
  else:
    total_mask = np.ones(len(df), dtype=bool)

  # Get masks for the three samples
  if mask_sample_m1 is not None:
    mask_sample_m1 = df.query(mask_sample_m1).index
  else:
    mask_sample_m1 = np.ones(len(df), dtype=bool)
  if mask_sample_nominal is not None:
    mask_sample_nominal = df.query(mask_sample_nominal).index
  else:
    mask_sample_nominal = np.ones(len(df), dtype=bool)
  if mask_sample_p1 is not None:
    mask_sample_p1 = df.query(mask_sample_p1).index
  else:
    mask_sample_p1 = np.ones(len(df), dtype=bool)

  # Combine with total mask
  mask_sample_m1 = mask_sample_m1.intersection(total_mask)
  mask_sample_nominal = mask_sample_nominal.intersection(total_mask)
  mask_sample_p1 = mask_sample_p1.intersection(total_mask)

  # Get the weights for the three samples
  nominal_weights = ((df.loc[mask_sample_nominal, nuisance_column_name] < 0.0) * (1 + df.loc[mask_sample_nominal, nuisance_column_name])) + ((df.loc[mask_sample_nominal, nuisance_column_name] >= 0.0) * (1 - df.loc[mask_sample_nominal, nuisance_column_name]))
  m1_weights = ((df.loc[mask_sample_m1, nuisance_column_name] < 0.0) * (-df.loc[mask_sample_m1, nuisance_column_name]))
  p1_weights = ((df.loc[mask_sample_p1, nuisance_column_name] >= 0.0) * (df.loc[mask_sample_p1, nuisance_column_name])) 

  # Apply the weights to the original weight column
  df.loc[mask_sample_nominal, wt_name] *= nominal_weights
  df.loc[mask_sample_m1, wt_name] *= m1_weights
  df.loc[mask_sample_p1, wt_name] *= p1_weights

  return df
  
  
def two_point_variation_weight(
    df,
    nuisance_column_name = None,
    wt_name = "wt",
    mask_sample_nominal = None,
    mask_sample_p1 = None,
    total_mask = None
  ):

  # Get total mask
  if total_mask is not None:
    total_mask = df.query(total_mask).index
  else:
    total_mask = np.ones(len(df), dtype=bool)

  # Get masks for the three samples
  if mask_sample_nominal is not None:
    mask_sample_nominal = df.query(mask_sample_nominal).index
  else:
    mask_sample_nominal = np.ones(len(df), dtype=bool)
  if mask_sample_p1 is not None:
    mask_sample_p1 = df.query(mask_sample_p1).index
  else:
    mask_sample_p1 = np.ones(len(df), dtype=bool)

  # Combine with total mask
  mask_sample_nominal = mask_sample_nominal.intersection(total_mask)
  mask_sample_p1 = mask_sample_p1.intersection(total_mask)

  # Get the weights for the three samples
  nominal_weights = (1 - df.loc[mask_sample_nominal, nuisance_column_name])
  p1_weights = df.loc[mask_sample_p1, nuisance_column_name]

  # Apply the weights to the original weight column
  df.loc[mask_sample_nominal, wt_name] *= nominal_weights
  df.loc[mask_sample_p1, wt_name] *= p1_weights

  return df

