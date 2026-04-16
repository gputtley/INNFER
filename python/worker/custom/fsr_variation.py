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