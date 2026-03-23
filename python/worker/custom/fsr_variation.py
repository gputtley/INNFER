import numpy as np

def fsr_weight(
    df,
    fsr_value=None,
    wt_name="wt"
  ):

  if fsr_value is not None:
    df["log_fsr"] = np.log(fsr_value)
  
  nu = df["log_fsr"]/np.log(2)

  lower_clip = 0.5
  higher_clip = 2.0
  df["GenWeights_isr1fsr0p5"] = df["GenWeights_isr1fsr0p5"].clip(lower=lower_clip, upper=higher_clip)
  df["GenWeights_isr1fsr2"] = df["GenWeights_isr1fsr2"].clip(lower=lower_clip, upper=higher_clip)

  asymln = np.clip(AsymLogNormal(nu, kp=df["GenWeights_isr1fsr2"], km=df["GenWeights_isr1fsr0p5"]), 0.0, 2.0)

  df.loc[:, wt_name] *= asymln

  return df


def AsymLogNormal(nu, kp=1.2, km=0.8, q=0.5):

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

  return out