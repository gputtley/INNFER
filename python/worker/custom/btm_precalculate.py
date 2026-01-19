import numpy as np

from useful_functions import CombineObjects

def btm_precal(df):

  vlepb = CombineObjects(
    {"pt": df["BJetLep_pt"], "eta": df["BJetLep_eta"], "phi": df["BJetLep_phi"], "mass": df["BJetLep_mass"]},
    {"pt": df["LeptonSave_pt"], "eta": df["LeptonSave_eta"], "phi": df["LeptonSave_phi"], "mass": df["LeptonSave_mass"]}
  )

  df = df.assign(
      FatJet_tau21 = df["FatJet_tau2"] / df["FatJet_tau1"],
      SubJet1_tau21 = df["SubJet1_tau2"] / df["SubJet1_tau1"],
      LeptonicTop_mass = vlepb["mass"],
      LeptonicTop_pt = vlepb["pt"]
  )

  #df["SubJet1_tau21"] = df["SubJet1_tau2"] / df["SubJet1_tau1"]
  #df["FatJet_tau21"] = df["FatJet_tau2"] / df["FatJet_tau1"]

  #vlepb = CombineObjects(
  #  {"pt": df["BJetLep_pt"], "eta": df["BJetLep_eta"], "phi": df["BJetLep_phi"], "mass": df["BJetLep_mass"]},
  #  {"pt": df["LeptonSave_pt"], "eta": df["LeptonSave_eta"], "phi": df["LeptonSave_phi"], "mass": df["LeptonSave_mass"]}
  #)
  #df["LeptonicTop_mass"] = vlepb["mass"]
  #df["LeptonicTop_pt"] = vlepb["pt"]

  return df