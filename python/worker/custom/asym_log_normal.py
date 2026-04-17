import numpy as np

def AsymLogNormal(nu, kp=1.2, km=0.8, q=0.5, clip_down=0.0, clip_up=2.0):

  nu = np.asarray(nu)

  out = np.empty_like(nu, dtype=float)

  mask_pos = nu >= q
  mask_neg = nu < -q
  mask_mid = ~(mask_pos | mask_neg)

  if isinstance(kp, float):
    kp = np.full_like(nu, kp, dtype=float)
  if isinstance(km, float):
    km = np.full_like(nu, km, dtype=float)

  # nu >= q
  out[mask_pos] = np.exp(nu[mask_pos] * np.log(kp[mask_pos]))

  # nu < -q
  out[mask_neg] = np.exp(-nu[mask_neg] * np.log(km[mask_neg]))

  # -q <= nu < q
  nu_m = nu[mask_mid]
  out[mask_mid] = np.exp(
      nu_m * (
          (np.log(km[mask_mid]) + np.log(kp[mask_mid])) *
          (3 * nu_m**5 / (8 * q**5)
            - 5 * nu_m**3 / (4 * q**3)
            + 15 * nu_m / (8 * q))
          - np.log(km[mask_mid]) + np.log(kp[mask_mid])
      ) / 2
  )

  # Clip the output to avoid extreme values
  out = np.clip(out, clip_down, clip_up)

  return out

def one_weight_variation(
    df,
    nuisance_column_name = None,
    weight_column_name = None,
    nominal_weight_column_name = None,
    wt_name = "wt",
    mask = None,
    weight_clip = (0.5, 2.0),
    log_normal_clip = (0.0, 2.0),
    add_one_to_weight = False
  ):

  # Check that the necessary columns are present
  if nuisance_column_name is None or weight_column_name is None:
    raise ValueError("nuisance_column_name and weight_column_name must be provided")

  # Apply mask if provided
  if mask is None:
    mask = np.ones(len(df), dtype=bool)
  else:
    mask = df.query(mask).index

  # Apply the clip for the weights
  weight_var = df.loc[mask, weight_column_name]
  if add_one_to_weight:
    weight_var += 1.0
  weight_var = weight_var.clip(lower=weight_clip[0], upper=weight_clip[1])

  if nominal_weight_column_name is not None:
    nominal_weight_var = df.loc[mask, nominal_weight_column_name].clip(lower=weight_clip[0], upper=weight_clip[1])
    weight_var /= nominal_weight_var

  # Get asym log normal weights
  asymln = AsymLogNormal(df.loc[mask, nuisance_column_name], kp=weight_var, km=1/weight_var, clip_down=log_normal_clip[0], clip_up=log_normal_clip[1])

  # Apply the weights to the original weight column
  df.loc[mask, wt_name] *= asymln

  return df


def two_weight_variation(
    df,
    nuisance_column_name = None,
    up_weight_column_name = None,
    down_weight_column_name = None,
    nominal_weight_column_name = None,
    wt_name = "wt",
    mask = None,
    weight_clip = (0.5, 2.0),
    log_normal_clip = (0.0, 2.0),
  ):

  # Check that the necessary columns are present
  if nuisance_column_name is None or up_weight_column_name is None or down_weight_column_name is None:
    raise ValueError("nuisance_column_name, up_weight_column_name and down_weight_column_name must be provided")

  # Apply mask if provided
  if mask is None:
    mask = np.ones(len(df), dtype=bool)
  else:
    mask = df.query(mask).index

  # Apply the clip for the weights
  up_weight_var = df.loc[mask, up_weight_column_name].clip(lower=weight_clip[0], upper=weight_clip[1])
  down_weight_var = df.loc[mask, down_weight_column_name].clip(lower=weight_clip[0], upper=weight_clip[1])

  if nominal_weight_column_name is not None:
    nominal_weight_var = df.loc[mask, nominal_weight_column_name].clip(lower=weight_clip[0], upper=weight_clip[1])
    up_weight_var /= nominal_weight_var
    down_weight_var /= nominal_weight_var
    
  # Get asym log normal weights
  asymln = AsymLogNormal(df.loc[mask, nuisance_column_name], kp=up_weight_var, km=down_weight_var, clip_down=log_normal_clip[0], clip_up=log_normal_clip[1])

  # Apply the weights to the original weight column
  df.loc[mask, wt_name] *= asymln

  return df