def btm_fsr_fix(df):

  columns_needed = ['psWeightRel_fsr_G2GG_cNS_dn', 'psWeightRel_fsr_G2GG_cNS_up', 'psWeightRel_fsr_G2GG_muR_dn', 'psWeightRel_fsr_G2GG_muR_up', 'psWeightRel_fsr_G2QQ_cNS_dn', 'psWeightRel_fsr_G2QQ_cNS_up', 'psWeightRel_fsr_G2QQ_muR_dn', 'psWeightRel_fsr_G2QQ_muR_up', 'psWeightRel_fsr_Q2QG_cNS_dn', 'psWeightRel_fsr_Q2QG_cNS_up', 'psWeightRel_fsr_Q2QG_muR_dn', 'psWeightRel_fsr_Q2QG_muR_up', 'psWeightRel_fsr_X2XG_cNS_dn', 'psWeightRel_fsr_X2XG_cNS_up', 'psWeightRel_fsr_X2XG_muR_dn', 'psWeightRel_fsr_X2XG_muR_up']

  # if these columns do not exist then set them to 1.0
  for col in columns_needed + ["matched_to_mini_fraction"]:
    if col not in df.columns:
      df[col] = 1.0
  if "matched_to_mini" not in df.columns:
    df["matched_to_mini"] = 0.0

  # replace any nan values in these columns with 1.0
  for col in columns_needed + ["matched_to_mini_fraction"]:
    df[col] = df[col].fillna(1.0)
  df["matched_to_mini"] = df["matched_to_mini"].fillna(0.0)

  return df