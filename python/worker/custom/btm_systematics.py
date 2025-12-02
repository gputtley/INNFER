import numpy as np

def btm_jec(df, years=["2016_PreVFP","2016_PostVFP","2017","2018","2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"], nuisances=["AbsoluteMPFBias","AbsoluteScale"]):

  jec_uncert = {
    # jec
    "AbsoluteMPFBias": {"Correlation" : 1},
    "AbsoluteScale": {"Correlation" : 1},
    "AbsoluteStat": {"Correlation" : 0},
    "FlavorQCD": {"Correlation" : 1},
    "Fragmentation": {"Correlation" : 1},
    "PileUpDataMC": {"Correlation" : 1},
    "PileUpPtBB": {"Correlation" : 1},
    "PileUpPtEC1": {"Correlation" : 1},
    "PileUpPtEC2": {"Correlation" : 1},
    "PileUpPtHF": {"Correlation" : 1},
    "PileUpPtRef": {"Correlation" : 1},
    "RelativeFSR": {"Correlation" : 1},
    "RelativePtBB": {"Correlation" : 1},
    "RelativePtEC1": {"Correlation" : 0},
    "RelativePtEC2": {"Correlation" : 0},
    "RelativePtHF": {"Correlation" : 1},
    "RelativeBal": {"Correlation" : 1},
    "RelativeSample": {"Correlation" : 0},
    "RelativeStatEC": {"Correlation" : 0},
    "RelativeStatFSR": {"Correlation" : 0},
    "RelativeStatHF": {"Correlation" : 0},
    "SinglePionECAL": {"Correlation" : 1},
    "SinglePionHCAL": {"Correlation" : 1},
    "TimePtEta": {"Correlation" : 0},
    ## flavour
    "FlavorPureGluon" : {"Correlation" : 1},
    "FlavorPureQuark" : {"Correlation" : 1},
    "FlavorPureCharm" : {"Correlation" : 1},
    "FlavorPureBottom" : {"Correlation" : 1},
  }

  # Apply to subjets
  for name, info in jec_uncert.items():
    if info["Correlation"] == 1:
      syst_names = [name]
      scalings = 1.0
    elif info["Correlation"] == 0:
      syst_names = [f"{name}_{yr}" for yr in years]
      scalings = 1.0
    elif info["Correlation"] == 0.5:
      syst_names = [name] + [f"{name}_{yr}" for yr in years]
      scalings = 0.5
    for ind in range(len(syst_names)):

      if syst_names[ind] not in nuisances:
        continue

      if df[syst_names[ind]].eq(0).all():
        continue

      df.loc[:, f"SubJet1_corrFactor_{syst_names[ind]}"] = 1.0 + (scalings*df.loc[:,syst_names[ind]]*df.loc[:,f"SubJet1_corrFactor_{name}"]/df.loc[:,f"SubJet1_corrFactor"])
      df.loc[:, f"SubJet2_corrFactor_{syst_names[ind]}"] = 1.0 + (scalings*df.loc[:,syst_names[ind]]*df.loc[:,f"SubJet2_corrFactor_{name}"]/df.loc[:,f"SubJet2_corrFactor"])
      df.loc[:, "SubJet1_pt"] *= df.loc[:, f"SubJet1_corrFactor_{syst_names[ind]}"]
      df.loc[:, "SubJet2_pt"] *= df.loc[:, f"SubJet2_corrFactor_{syst_names[ind]}"]
      df.loc[:, "SubJet1_mass"] *= df.loc[:, f"SubJet1_corrFactor_{syst_names[ind]}"]
      df.loc[:, "SubJet2_mass"] *= df.loc[:, f"SubJet2_corrFactor_{syst_names[ind]}"]

      # Remove temporary columns
      df.drop(columns=[f"SubJet1_corrFactor_{syst_names[ind]}", f"SubJet2_corrFactor_{syst_names[ind]}"], inplace=True)

  # Combine subjets
  v12 = combine_objects(
    {"pt": df.loc[:, "SubJet1_pt"], "eta": df.loc[:, "SubJet1_eta"], "phi": df.loc[:, "SubJet1_phi"], "mass": df.loc[:, "SubJet1_mass"]},
    {"pt": df.loc[:, "SubJet2_pt"], "eta": df.loc[:, "SubJet2_eta"], "phi": df.loc[:, "SubJet2_phi"], "mass": df.loc[:, "SubJet2_mass"]}
  )

  df.loc[:, "CombinedSubJets_mass"] = v12["mass"]
  df.loc[:, "CombinedSubJets_pt"] = v12["pt"]

  return df


def combine_objects(obj1, obj2):

  px_1 = obj1["pt"] * np.cos(obj1["phi"])
  py_1 = obj1["pt"] * np.sin(obj1["phi"])
  pz_1 = obj1["pt"] * np.sinh(obj1["eta"])
  e_1 = np.sqrt(obj1["mass"]**2 + px_1**2 + py_1**2 + pz_1**2)
  px_2 = obj2["pt"] * np.cos(obj2["phi"])
  py_2 = obj2["pt"] * np.sin(obj2["phi"])
  pz_2 = obj2["pt"] * np.sinh(obj2["eta"])
  e_2 = np.sqrt(obj2["mass"]**2 + px_2**2 + py_2**2 + pz_2**2)
  px = px_1 + px_2
  py = py_1 + py_2
  pz = pz_1 + pz_2
  e = e_1 + e_2
  mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
  pt = np.sqrt(px**2 + py**2)
  eta = 0.5 * np.log((e + pz) / (e - pz))
  phi = np.arctan2(py, px)
  return {"pt": pt, "eta": eta, "phi": phi, "mass": mass}