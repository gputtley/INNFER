import numpy as np
import time

from useful_functions import CombineObjects

def btm_jec(
    df, 
    years=["2016_PreVFP","2016_PostVFP","2017","2018","2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"], 
    nuisances=["AbsoluteMPFBias","AbsoluteScale"], 
    include_b=False,
    include_b_syst=False
  ):

  # JEC and flavour uncertainties
  jec_uncert = {
    # jec
    "AbsoluteMPFBias": {"Correlation" : 1, "Type" : "corrFactor"},
    "AbsoluteScale": {"Correlation" : 1, "Type" : "corrFactor"},
    "AbsoluteStat": {"Correlation" : 0, "Type" : "corrFactor"},
    "FlavorQCD": {"Correlation" : 1, "Type" : "corrFactor"},
    "Fragmentation": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpDataMC": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpPtBB": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpPtEC1": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpPtEC2": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpPtHF": {"Correlation" : 1, "Type" : "corrFactor"},
    "PileUpPtRef": {"Correlation" : 1, "Type" : "corrFactor"},
    "RelativeFSR": {"Correlation" : 1, "Type" : "corrFactor"},
    "RelativePtBB": {"Correlation" : 1, "Type" : "corrFactor"},
    "RelativePtEC1": {"Correlation" : 0, "Type" : "corrFactor"},
    "RelativePtEC2": {"Correlation" : 0, "Type" : "corrFactor"},
    "RelativePtHF": {"Correlation" : 1, "Type" : "corrFactor"},
    "RelativeBal": {"Correlation" : 1, "Type" : "corrFactor"},
    "RelativeSample": {"Correlation" : 0, "Type" : "corrFactor"},
    "RelativeStatEC": {"Correlation" : 0, "Type" : "corrFactor"},
    "RelativeStatFSR": {"Correlation" : 0, "Type" : "corrFactor"},
    "RelativeStatHF": {"Correlation" : 0, "Type" : "corrFactor"},
    "SinglePionECAL": {"Correlation" : 1, "Type" : "corrFactor"},
    "SinglePionHCAL": {"Correlation" : 1, "Type" : "corrFactor"},
    "TimePtEta": {"Correlation" : 0, "Type" : "corrFactor"},
    # flavour
    "FlavorPureGluon" : {"Correlation" : 1, "ObjectSelection" : "abs(MatchedGenJet_$OBJECT_partonFlavour)==21", "Type" : "corrFactor"},
    "FlavorPureQuark" : {"Correlation" : 1, "ObjectSelection" : "abs(MatchedGenJet_$OBJECT_partonFlavour)<=3", "Type" : "corrFactor"},
    "FlavorPureCharm" : {"Correlation" : 1, "ObjectSelection" : "abs(MatchedGenJet_$OBJECT_partonFlavour)==4", "Type" : "corrFactor"},
    "FlavorPureBottom" : {"Correlation" : 1, "ObjectSelection" : "abs(MatchedGenJet_$OBJECT_partonFlavour)==5", "Type" : "corrFactor"},
    # JER
    "JER_eta_lt_1p93": {"Correlation" : 1, "ObjectSelection": "$OBJECT_eta < 1.93", "Type" : "smearFactor"},
    "JER_eta_1p93_to_2p5": {"Correlation" : 1, "ObjectSelection": "($OBJECT_eta >= 1.93) & ($OBJECT_eta < 2.5)", "Type" : "smearFactor"},
    "JER_eta_2p5_to_3p0_pt_0_to_50": {"Correlation" : 1, "ObjectSelection": "($OBJECT_eta >= 2.5) & ($OBJECT_eta < 3.0) & ($OBJECT_pt < 50)", "Type" : "smearFactor"},
    "JER_eta_2p5_to_3p0_pt_gt_50": {"Correlation" : 1, "ObjectSelection": "($OBJECT_eta >= 2.5) & ($OBJECT_eta < 3.0) & ($OBJECT_pt >= 50)", "Type" : "smearFactor"},
    "JER_eta_3p0_to_5p0_pt_0_to_50": {"Correlation" : 1, "ObjectSelection": "($OBJECT_eta >= 3.0) & ($OBJECT_eta < 5.0) & ($OBJECT_pt < 50)", "Type" : "smearFactor"},
    "JER_eta_3p0_to_5p0_pt_gt_50": {"Correlation" : 1, "ObjectSelection": "($OBJECT_eta >= 3.0) & ($OBJECT_eta < 5.0) & ($OBJECT_pt >= 50)", "Type" : "smearFactor"}
  }

  # Apply JEC to jets
  added_columns = []
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

      # Initiate corrFactor syst columns
      df[f"SubJet1_shiftFactor_{syst_names[ind]}"] = 1.0
      df[f"SubJet2_shiftFactor_{syst_names[ind]}"] = 1.0
      added_columns.extend([f"SubJet1_shiftFactor_{syst_names[ind]}", f"SubJet2_shiftFactor_{syst_names[ind]}"])
      if include_b:
        df[f"BJetLep_shiftFactor_{syst_names[ind]}"] = 1.0
        added_columns.append(f"BJetLep_shiftFactor_{syst_names[ind]}")

      # Apply only to selected events if specified
      selected_indices = np.ones(len(df), dtype=bool)
      if "Selection" in info:
        selected_indices &= df.eval(info["Selection"])
      subjet1_indices = selected_indices.copy()
      subjet2_indices = selected_indices.copy()
      if include_b and include_b_syst:
        bjetlep_indices = selected_indices.copy()
      if "ObjectSelection" in info:
        subjet1_indices &= df.eval(info["ObjectSelection"].replace("$OBJECT", "SubJet1"))
        subjet2_indices &= df.eval(info["ObjectSelection"].replace("$OBJECT", "SubJet2"))
        if include_b and include_b_syst:
          bjetlep_indices &= df.eval(info["ObjectSelection"].replace("$OBJECT", "BJetLep"))

      # Get variations
      if info["Type"] == "corrFactor":

        corrFactor = lambda df, obj, indices, syst_name, name, scalings: 1.0 + (scalings*df.loc[indices,syst_name]*df.loc[indices,f"{obj}_corrFactor_{name}"]/df.loc[indices,f"{obj}_corrFactor"])

        df.loc[subjet1_indices, f"SubJet1_shiftFactor_{syst_names[ind]}"] = corrFactor(df, "SubJet1", subjet1_indices, syst_names[ind], name, scalings)
        df.loc[subjet2_indices, f"SubJet2_shiftFactor_{syst_names[ind]}"] = corrFactor(df, "SubJet2", subjet2_indices, syst_names[ind], name, scalings)
        if include_b and include_b_syst:
          df.loc[bjetlep_indices, f"BJetLep_shiftFactor_{syst_names[ind]}"] = corrFactor(df, "BJetLep", bjetlep_indices, syst_names[ind], name, scalings)

      elif info["Type"] == "smearFactor":
        
        smearFactor = lambda df, obj, indices, syst_name, scalings: 1.0 + ((df.loc[indices,syst_name]>=0) * scalings * df.loc[indices,syst_name] * (df.loc[indices, f"{obj}_smearFactor_up"]-df.loc[indices, f"{obj}_smearFactor"]) / df.loc[indices, f"{obj}_smearFactor"]) + ((df.loc[indices,syst_name]<0) * scalings * abs(df.loc[indices,syst_name]) * (df.loc[indices, f"{obj}_smearFactor_down"]-df.loc[indices, f"{obj}_smearFactor"]) / df.loc[indices, f"{obj}_smearFactor"])

        df.loc[subjet1_indices, f"SubJet1_shiftFactor_{syst_names[ind]}"] = smearFactor(df, "SubJet1", subjet1_indices, syst_names[ind], scalings)
        df.loc[subjet2_indices, f"SubJet2_shiftFactor_{syst_names[ind]}"] = smearFactor(df, "SubJet2", subjet2_indices, syst_names[ind], scalings)
        if include_b and include_b_syst:
          df.loc[bjetlep_indices, f"BJetLep_shiftFactor_{syst_names[ind]}"] = smearFactor(df, "BJetLep", bjetlep_indices, syst_names[ind], scalings)

      # Apply variations
      df["SubJet1_pt"] *= df[f"SubJet1_shiftFactor_{syst_names[ind]}"]
      df["SubJet2_pt"] *= df[f"SubJet2_shiftFactor_{syst_names[ind]}"]
      df["SubJet1_mass"] *= df[f"SubJet1_shiftFactor_{syst_names[ind]}"]
      df["SubJet2_mass"] *= df[f"SubJet2_shiftFactor_{syst_names[ind]}"]
      if include_b and include_b_syst:
        df["BJetLep_pt"] *= df[f"BJetLep_shiftFactor_{syst_names[ind]}"]
        df["BJetLep_mass"] *= df[f"BJetLep_shiftFactor_{syst_names[ind]}"]

  # Remove temporary columns
  #df = df.drop(columns=added_columns)

  # Combine subjets
  v12 = CombineObjects(
    {"pt": df["SubJet1_pt"].values, "eta": df["SubJet1_eta"].values, "phi": df["SubJet1_phi"].values, "mass": df["SubJet1_mass"].values},
    {"pt": df["SubJet2_pt"].values, "eta": df["SubJet2_eta"].values, "phi": df["SubJet2_phi"].values, "mass": df["SubJet2_mass"].values}
  )

  df["CombinedSubJets_mass"] = v12["mass"]
  df["CombinedSubJets_pt"] = v12["pt"]

  if include_b:
    vlepb = CombineObjects(
      {"pt": df["BJetLep_pt"], "eta": df["BJetLep_eta"], "phi": df["BJetLep_phi"], "mass": df["BJetLep_mass"]},
      {"pt": df["LeptonSave_pt"], "eta": df["LeptonSave_eta"], "phi": df["LeptonSave_phi"], "mass": df["LeptonSave_mass"]}
    )
    df["LeptonicTop_mass"] = vlepb["mass"]
    df["LeptonicTop_pt"] = vlepb["pt"]

  return df
