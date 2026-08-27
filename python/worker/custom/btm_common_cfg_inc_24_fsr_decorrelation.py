import os
import copy
import glob
import socket

import numpy as np

prep_data_dir = os.getenv('PREP_DATA_DIR')

def make_common_config(run_name, categories, categories_per_era, add_classifier=False):

  ### Define core variables for config ###

  nom_weight = "weight"
  pre_selection = "((ClosestJetWithLeptonRemoved_deltaR>0.25) | (ClosestJetWithLeptonRemoved_ptrel>30))"
  post_selection = "((CombinedSubJets_pt > 400) & (LeptonicTop_mass < CombinedSubJets_mass) & (MET_pt>50) & (BJetLep_pt>30) & (CombinedSubJets_pt < 800) & (CombinedSubJets_mass > 50) & (CombinedSubJets_mass < 300) & (SubJet1_mass > 5) & (SubJet1_mass < 175) & (SubJet1_pt > 200) & (SubJet1_pt < 700) & (SubJet1_tau21 > 0.01) & (SubJet1_tau21 < 0.99) & (FatJet_tau21 > 0.05) & (FatJet_tau21 < 0.9) & (SubJet2_btagDeepB > 0) & (SubJet2_btagDeepB < 1))"

  host = socket.getfqdn()
  if host.endswith(".cern.ch"):
    raise NotImplementedError("This data is not currently stored on eos.")
  else:
    data_loc = "/vols/cms/gu18/innfer_v1/data/top_reco/270726_parquet_with_psweights"

  variables = [
    "SubJet1_tau21",
    "FatJet_tau21",
  ]
  pois = []

  years_per_era = {
    "run2": ["2016_PreVFP", "2016_PostVFP", "2017", "2018"],
    "2223": ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"],
    "24": ["2024"],
  }
  year_index = {
    "2016_PreVFP": 0,
    "2016_PostVFP": 1,
    "2017": 2,
    "2018": 3,
    "2022_preEE": 4,
    "2022_postEE": 5,
    "2023_preBPix": 6,
    "2023_postBPix": 7,
    "2024": 8,
  }
  ttbar_names = {
    172.5 : [
      "TTToSemiLeptonic",
      "TTTo2L2Nu",
      "TTToHadronic",
      "TTMtt700To1000",
      "TTMtt1000",
    ]
  }
  other_names = [
    "WJetsToLNu",
    "WJetsToLNuHT70To100",
    "WJetsToLNuHT100To200",
    "WJetsToLNuHT200To400",
    "WJetsToLNuHT400To600",
    "WJetsToLNuHT600To800",
    "WJetsToLNuHT800To1200",
    "WJetsToLNuHT1200To2500",
    "WJetsToLNuHT2500",
    "WJetsToLNuHT40To100MLNu0To120",
    "WJetsToLNuHT100To400MLNu0To120",
    "WJetsToLNuHT400To800MLNu0To120",
    "WJetsToLNuHT800To1500MLNu0To120",
    "WJetsToLNuHT1500To2500MLNu0To120",
    "WJetsToLNuHT2500MLNu0To120",
    "WJetsToLNuHT40To100MLNu120",
    "WJetsToLNuHT100To400MLNu120",
    "WJetsToLNuHT400To800MLNu120",
    "WJetsToLNuHT800To1500MLNu120",
    "WJetsToLNuHT1500To2500MLNu120",
    "WJetsToLNuHT2500MLNu120",
    "ST_t_channel_top",
    "ST_t_channel_antitop",
    "ST_s_channel",
    "ST_s_channel_top",
    "ST_s_channel_antitop",
    "ST_tW_antitop",
    "ST_tW_top",
  ]


  data_names = [
    "DATA_SingleEle",
    "DATA_SingleMuon",
  ]

  rate_parameters = ["ttbar"]
  default_values = {}
  preprocess = {
    "train_test_val_split": "0.8:0.1:0.1",
    "save_extra_columns": {
      "ttbar": ["sim_mass","GenTop1_mass","GenTop2_mass"],
      "other": []
    },
    "standardisation": {
      "ttbar": {},
      "other": {}
    },
    "stratify_to" : "sim_mass",
    "density_pretransform_to_gaussian" : True,
    "density_pretransform_pca_whitening" : True,
  }
  lnN = {
    "ttbar" : [],
    "other" : []
  }



  ##################################################################################

  # Define dictionaries
  base_files = {}
  sub_models = {"density_models":[], "classifier_models":[], "regression_models":[], "yields":[]}
  models = {"ttbar": copy.deepcopy(sub_models), "other": copy.deepcopy(sub_models)}
  nuisances = []
  ttbar_files = {}  
  ttbar_yield_files = {}
  other_files = {}
  data_files = {}

  # Loop through categories
  for cat, years in years_per_era.items():

    if cat not in categories.keys(): continue

    # Get information for ttbar (and ttbar_yield)
    ttbar_files[cat] = {}
    ttbar_yield_files[cat] = {}
    for year in years:
      for name in ttbar_names[172.5]:
        file_path = f"{data_loc}/{name}_{year}.parquet"
        if os.path.exists(file_path):
          ttbar_files[cat][file_path] = {
            "sim_mass": 172.5,
            "year_ind": year_index[year],
            "top_process" : 1.0,
          }

          ttbar_yield_files[cat][file_path] = {
            "sim_mass": 172.5,
            "year_ind": year_index[year],
            "top_process" : 1.0,
          }

    # Get information for other
    other_files[cat] = {}
    for year in years:
      for name in other_names:
        file_path = f"{data_loc}/{name}_{year}.parquet"
        if os.path.exists(file_path):
          other_files[cat][file_path] = {
            "year_ind": year_index[year],
            "top_process" : 1.0 if name.startswith("ST") else 0.0
          }

    # Get information for data
    data_files[cat] = {}
    for year in years:
      for name in data_names:
        all_files = glob.glob(f"{data_loc}/{name}_{year}_*.parquet")
        for file_path in all_files:
          if os.path.exists(file_path):
            data_files[cat][file_path] = {
              "year_ind": year_index[year],
            }


    ### Precalculate (at LoadData step) ###

    common_pre_calculate = {}
    ttbar_pre_calculate = {}
    ttbar_yield_pre_calculate = {}
    other_pre_calculate = {}
    other_yield_pre_calculate = {}

    # Set common precalculate variables to the rest
    for k, v in common_pre_calculate.items():
      ttbar_pre_calculate[k] = v
      ttbar_yield_pre_calculate[k] = v
      other_pre_calculate[k] = v
      other_yield_pre_calculate[k] = v

    ttbar_pre_calculate["btm_fsr_fix"] = {
      "type": "function",
      "file": "btm_fsr",
      "name": "btm_fsr_fix",
      "args": {},
      "inputs": ['psWeightRel_fsr_G2GG_cNS_dn', 'psWeightRel_fsr_G2GG_cNS_up', 'psWeightRel_fsr_G2GG_muR_dn', 'psWeightRel_fsr_G2GG_muR_up', 'psWeightRel_fsr_G2QQ_cNS_dn', 'psWeightRel_fsr_G2QQ_cNS_up', 'psWeightRel_fsr_G2QQ_muR_dn', 'psWeightRel_fsr_G2QQ_muR_up', 'psWeightRel_fsr_Q2QG_cNS_dn', 'psWeightRel_fsr_Q2QG_cNS_up', 'psWeightRel_fsr_Q2QG_muR_dn', 'psWeightRel_fsr_Q2QG_muR_up', 'psWeightRel_fsr_X2XG_cNS_dn', 'psWeightRel_fsr_X2XG_cNS_up', 'psWeightRel_fsr_X2XG_muR_dn', 'psWeightRel_fsr_X2XG_muR_up', 'matched_to_mini_fraction'],
      "outputs" : []
    }
    ttbar_yield_pre_calculate["btm_fsr_fix"] = ttbar_pre_calculate["btm_fsr_fix"]


    ### Calculate (at PreProcess step) ###

    jec_inputs = []
    objs = ["SubJet1", "SubJet2", "BJetLep", "MET"]
    jec_inputs += [f"{obj}_corrFactor" for obj in objs if obj not in ["MET"]]
    jec_inputs += [f"{obj}_{var}" for obj in objs for var in ["eta", "pt", "mass", "phi"] if obj not in ["MET"]]
    jec_inputs += ["MET_pt"]

    common_calculate = {
      "btm_calculate": {
        "type": "function", 
        "file": "btm_calculate", 
        "name": "btm_cal", 
        "args": {}, 
        "inputs": ["SubJet1_tau1", "SubJet1_tau2", "FatJet_tau1", "FatJet_tau2", "BJetLep_pt", "BJetLep_eta", "BJetLep_phi", "BJetLep_mass", "LeptonSave_pt", "LeptonSave_eta", "LeptonSave_phi", "LeptonSave_mass", "Extra_BTagWeightCorrection_up", "Extra_BTagWeightCorrection_down"],
        "outputs": ["CombinedSubJets_mass", "CombinedSubJets_pt", "SubJet1_tau21", "FatJet_tau21", "LeptonicTop_mass", "LeptonicTop_pt", "Extra_BTagWeightCorrection"]
      },
      "btm_jec" : {
        "type": "function",
        "file": "btm_experimental_systematics",
        "name": "btm_jec",
        "args": {"years": years, "nuisances": [], "include_b": True, "include_b_syst": True, "include_met_syst": True},
        "inputs": jec_inputs,
        "outputs": ["CombinedSubJets_mass", "CombinedSubJets_pt", "SubJet1_mass", "SubJet1_pt", "LeptonicTop_mass", "LeptonicTop_pt", "MET_pt"],
      },
    }
    if add_classifier:
      common_calculate["btm_merged_classifier"] = {
        "type": "function",
        "file": "btm_merged_classifier",
        "name": "btm_merged_classifier",
        "args": {},
        "inputs": ["CombinedSubJets_pt", "SubJet1_mass", "SubJet1_pt", "SubJet1_tau21", "FatJet_tau21", "SubJet2_btagDeepB"],
        "outputs": ["btm_merged_classifier", "btm_merged_classifier_pass"]
      }


    ttbar_calculate = {}
    ttbar_yield_calculate = {}
    other_calculate = {}
    other_yield_calculate = {}

    # JEC is still computed (for CombinedSubJets_mass/pt, LeptonicTop_mass/pt used in selections)
    ttbar_classifier_nuisances = []
    other_classifier_nuisances = []

    # Set common calculate variables to the rest
    for k, v in common_calculate.items():
      ttbar_calculate[k] = v
      ttbar_yield_calculate[k] = v
      other_calculate[k] = v
      other_yield_calculate[k] = v



    ### Weight shifts (at PreProcess step) ###

    # Build weight shifts (only the 8 fsr_* nuisances below are used)
    common_weight_shifts = {}

    ttbar_weight_shifts = {
      "fsr_G2GG_muR" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "fsr_G2GG_muR", "up_weight_column_name": "psWeightRel_fsr_G2GG_muR_up", "down_weight_column_name": "psWeightRel_fsr_G2GG_muR_dn", "wt_name": "wt", "scale_mask_and_set_other_to_zero": "(1/matched_to_mini_fraction)", "mask": "((top_process == 1) & (fsr_G2GG_muR != 0))", "mask_and_set_other_to_zero": "(matched_to_mini == 1)", "log_normal_clip" : (0.1, 10.0)},
        "inputs": ["psWeightRel_fsr_G2GG_muR_up", "psWeightRel_fsr_G2GG_muR_dn", "top_process", "matched_to_mini", "matched_to_mini_fraction", "fsr_G2GG_muR"],
      },
      "fsr_G2QQ_muR" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "fsr_G2QQ_muR", "up_weight_column_name": "psWeightRel_fsr_G2QQ_muR_up", "down_weight_column_name": "psWeightRel_fsr_G2QQ_muR_dn", "wt_name": "wt", "scale_mask_and_set_other_to_zero": "(1/matched_to_mini_fraction)", "mask": "((top_process == 1) & (fsr_G2QQ_muR != 0))", "mask_and_set_other_to_zero": "(matched_to_mini == 1)", "log_normal_clip" : (0.1, 10.0)},
        "inputs": ["psWeightRel_fsr_G2QQ_muR_up", "psWeightRel_fsr_G2QQ_muR_dn", "top_process", "matched_to_mini", "matched_to_mini_fraction", "fsr_G2QQ_muR"],
      },
      "fsr_Q2QG_muR" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "fsr_Q2QG_muR", "up_weight_column_name": "psWeightRel_fsr_Q2QG_muR_up", "down_weight_column_name": "psWeightRel_fsr_Q2QG_muR_dn", "wt_name": "wt", "scale_mask_and_set_other_to_zero": "(1/matched_to_mini_fraction)", "mask": "((top_process == 1) & (fsr_Q2QG_muR != 0))", "mask_and_set_other_to_zero": "(matched_to_mini == 1)", "log_normal_clip" : (0.1, 10.0)},
        "inputs": ["psWeightRel_fsr_Q2QG_muR_up", "psWeightRel_fsr_Q2QG_muR_dn", "top_process", "matched_to_mini", "matched_to_mini_fraction", "fsr_Q2QG_muR"],
      },
      "fsr_X2XG_muR" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "fsr_X2XG_muR", "up_weight_column_name": "psWeightRel_fsr_X2XG_muR_up", "down_weight_column_name": "psWeightRel_fsr_X2XG_muR_dn", "wt_name": "wt", "scale_mask_and_set_other_to_zero": "(1/matched_to_mini_fraction)", "mask": "((top_process == 1) & (fsr_X2XG_muR != 0))", "mask_and_set_other_to_zero": "(matched_to_mini == 1)", "log_normal_clip" : (0.1, 10.0)},
        "inputs": ["psWeightRel_fsr_X2XG_muR_up", "psWeightRel_fsr_X2XG_muR_dn", "top_process", "matched_to_mini", "matched_to_mini_fraction", "fsr_X2XG_muR"],
      },
      "fsr_G2GG_cNS" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "fsr_G2GG_cNS", "up_weight_column_name": "psWeightRel_fsr_G2GG_cNS_up", "down_weight_column_name": "psWeightRel_fsr_G2GG_cNS_dn", "wt_name": "wt", "scale_mask_and_set_other_to_zero": "(1/matched_to_mini_fraction)", "mask": "((top_process == 1) & (fsr_G2GG_cNS != 0))", "mask_and_set_other_to_zero": "(matched_to_mini == 1)", "log_normal_clip" : (0.1, 10.0)},
        "inputs": ["psWeightRel_fsr_G2GG_cNS_up", "psWeightRel_fsr_G2GG_cNS_dn", "top_process", "matched_to_mini", "matched_to_mini_fraction", "fsr_G2GG_cNS"],
      },
      "fsr_G2QQ_cNS" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "fsr_G2QQ_cNS", "up_weight_column_name": "psWeightRel_fsr_G2QQ_cNS_up", "down_weight_column_name": "psWeightRel_fsr_G2QQ_cNS_dn", "wt_name": "wt", "scale_mask_and_set_other_to_zero": "(1/matched_to_mini_fraction)", "mask": "((top_process == 1) & (fsr_G2QQ_cNS != 0))", "mask_and_set_other_to_zero": "(matched_to_mini == 1)", "log_normal_clip" : (0.1, 10.0)},
        "inputs": ["psWeightRel_fsr_G2QQ_cNS_up", "psWeightRel_fsr_G2QQ_cNS_dn", "top_process", "matched_to_mini", "matched_to_mini_fraction", "fsr_G2QQ_cNS"],
      },
      "fsr_Q2QG_cNS" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "fsr_Q2QG_cNS", "up_weight_column_name": "psWeightRel_fsr_Q2QG_cNS_up", "down_weight_column_name": "psWeightRel_fsr_Q2QG_cNS_dn", "wt_name": "wt", "scale_mask_and_set_other_to_zero": "(1/matched_to_mini_fraction)", "mask": "((top_process == 1) & (fsr_Q2QG_cNS != 0))", "mask_and_set_other_to_zero": "(matched_to_mini == 1)", "log_normal_clip" : (0.1, 10.0)},
        "inputs": ["psWeightRel_fsr_Q2QG_cNS_up", "psWeightRel_fsr_Q2QG_cNS_dn", "top_process", "matched_to_mini", "matched_to_mini_fraction", "fsr_Q2QG_cNS"],
      },
      "fsr_X2XG_cNS" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "fsr_X2XG_cNS", "up_weight_column_name": "psWeightRel_fsr_X2XG_cNS_up", "down_weight_column_name": "psWeightRel_fsr_X2XG_cNS_dn", "wt_name": "wt", "scale_mask_and_set_other_to_zero": "(1/matched_to_mini_fraction)", "mask": "((top_process == 1) & (fsr_X2XG_cNS != 0))", "mask_and_set_other_to_zero": "(matched_to_mini == 1)", "log_normal_clip" : (0.1, 10.0)},
        "inputs": ["psWeightRel_fsr_X2XG_cNS_up", "psWeightRel_fsr_X2XG_cNS_dn", "top_process", "matched_to_mini", "matched_to_mini_fraction", "fsr_X2XG_cNS"],
      },
    }

    ttbar_yield_weight_shifts = copy.deepcopy(ttbar_weight_shifts)

    other_weight_shifts = {}
    other_yield_weight_shifts = copy.deepcopy(other_weight_shifts)

    # Add to nuisances
    common_classifier_nuisances = list(common_weight_shifts.keys())

    ttbar_classifier_nuisances += common_classifier_nuisances + list(ttbar_weight_shifts.keys())
    other_classifier_nuisances += common_classifier_nuisances + list(other_weight_shifts.keys())
    nuisances += common_classifier_nuisances + list(ttbar_weight_shifts.keys()) + list(other_weight_shifts.keys())

    # Set common weight shifts to the rest
    for k, v in common_weight_shifts.items():
      ttbar_weight_shifts[k] = v
      ttbar_yield_weight_shifts[k] = v
      other_weight_shifts[k] = v
      other_yield_weight_shifts[k] = v


    # Define preselection for ttbar (and ttbar_yield)
    ttbar_pre_selection = f"(({pre_selection}) & (GenTop1_mass>150) & (GenTop1_mass<190) & (GenTop2_mass>150) & (GenTop2_mass<190))"

    add_missing_columns_with_nans = ['psWeightRel_fsr_G2GG_cNS_dn', 'psWeightRel_fsr_G2GG_cNS_up', 'psWeightRel_fsr_G2GG_muR_dn', 'psWeightRel_fsr_G2GG_muR_up', 'psWeightRel_fsr_G2QQ_cNS_dn', 'psWeightRel_fsr_G2QQ_cNS_up', 'psWeightRel_fsr_G2QQ_muR_dn', 'psWeightRel_fsr_G2QQ_muR_up', 'psWeightRel_fsr_Q2QG_cNS_dn', 'psWeightRel_fsr_Q2QG_cNS_up', 'psWeightRel_fsr_Q2QG_muR_dn', 'psWeightRel_fsr_Q2QG_muR_up', 'psWeightRel_fsr_X2XG_cNS_dn', 'psWeightRel_fsr_X2XG_cNS_up', 'psWeightRel_fsr_X2XG_muR_dn', 'psWeightRel_fsr_X2XG_muR_up', 'matched_to_mini_fraction', 'matched_to_mini']

    # Make base files
    base_files[f"base_ttbar_{cat}"] = {
      "inputs": list(ttbar_files[cat].keys()),
      "add_columns": {k: [v[k] for v in ttbar_files[cat].values()] for k in list(ttbar_files[cat].values())[0].keys()},
      "add_missing_columns_with_nans": add_missing_columns_with_nans,
      "selection": ttbar_pre_selection,
      "weight": nom_weight,
      "parameters": [],
      "pre_calculate": ttbar_pre_calculate,
      "calculate": ttbar_calculate,
      "post_calculate_selection": post_selection,
      "weight_shifts": ttbar_weight_shifts
    }
    base_files[f"base_ttbar_yield_{cat}"] = {
      "inputs": list(ttbar_yield_files[cat].keys()),
      "add_columns": {k: [v[k] for v in ttbar_yield_files[cat].values()] for k in list(ttbar_yield_files[cat].values())[0].keys()},
      "add_missing_columns_with_nans": add_missing_columns_with_nans,
      "selection": pre_selection,
      "weight": nom_weight,
      "parameters": [],
      "pre_calculate": ttbar_yield_pre_calculate,
      "calculate": ttbar_yield_calculate,
      "post_calculate_selection": post_selection,
      "weight_shifts": ttbar_yield_weight_shifts
    }
    base_files[f"base_other_{cat}"] = {
      "inputs": list(other_files[cat].keys()),
      "add_columns": {k: [v[k] for v in other_files[cat].values()] for k in list(other_files[cat].values())[0].keys()},
      "selection": pre_selection,
      "weight": nom_weight,
      "parameters": [],
      "pre_calculate": other_pre_calculate,
      "calculate": other_calculate,
      "post_calculate_selection": post_selection,
      "weight_shifts": other_weight_shifts
    }
    base_files[f"base_other_yield_{cat}"] = {
      "inputs": list(other_files[cat].keys()),
      "add_columns": {k: [v[k] for v in other_files[cat].values()] for k in list(other_files[cat].values())[0].keys()},
      "selection": pre_selection,
      "weight": nom_weight,
      "parameters": [],
      "pre_calculate": other_yield_pre_calculate,
      "calculate": other_yield_calculate,
      "post_calculate_selection": post_selection,
      "weight_shifts": other_weight_shifts
    }
    
    # Make density models
    models["ttbar"]["density_models"] += [
      {
        "parameters": [],
        "file": f"base_ttbar_{cat}",
        "shifts": {},
        "n_copies": 1,
        "categories": categories_per_era[cat],
      }
    ]
    models["other"]["density_models"] += [
      {
        "parameters": [],
        "file": f"base_other_{cat}",
        "shifts": {},
        "n_copies": 1,
        "categories": categories_per_era[cat],
      },
    ]

    # Make classifier models
    #models["ttbar"]["classifier_models"] += [{"parameter":k, "file":f"base_ttbar_{cat}", "shifts":{k: {"type": "flat_top", "range": [-3.0,3.0], "other": {"sigma_out": 0.6}}}, "n_copies":3, "categories": categories_per_era[cat]} for k in ttbar_classifier_nuisances]

    for k in ttbar_classifier_nuisances:
      if k in ttbar_weight_shifts.keys() and ttbar_weight_shifts[k]["name"] == "three_point_variation_weight":
        models["ttbar"]["classifier_models"] += [{"parameter":k, "file":f"base_ttbar_{cat}", "shifts":{k: {"type": "discrete", "values": [-1.0,1.0]}}, "n_copies":3, "categories": categories_per_era[cat]}]
      elif k in ttbar_weight_shifts.keys() and ttbar_weight_shifts[k]["name"] == "two_point_variation_weight":
        models["ttbar"]["classifier_models"] += [{"parameter":k, "file":f"base_ttbar_{cat}", "shifts":{k: {"type": "fixed", "value": 1.0}}, "n_copies":3, "categories": categories_per_era[cat]}]
      else:
        models["ttbar"]["classifier_models"] += [{"parameter":k, "file":f"base_ttbar_{cat}", "shifts":{k: {"type": "flat_top", "range": [-3.0,3.0], "other": {"sigma_out": 0.6}}}, "n_copies":3, "categories": categories_per_era[cat]}]

    models["other"]["classifier_models"] += [{"parameter":k, "file":f"base_other_{cat}", "shifts":{k: {"type": "flat_top", "range": [-3.0,3.0], "other": {"sigma_out": 0.6}}}, "n_copies":3, "categories": categories_per_era[cat]} for k in other_classifier_nuisances]

    # Make yields
    models["ttbar"]["yields"] += [{"file": f"base_ttbar_yield_{cat}", "categories": categories_per_era[cat]}]
    models["other"]["yields"] += [{"file": f"base_other_yield_{cat}", "categories": categories_per_era[cat]}]

  # Define validation loop for POIs
  validation_loop = [{}]
  validation = {
    "loop": validation_loop,
    "files": {
      "ttbar": [{"file": f"base_ttbar_{era}", "categories": categories_per_era[era]} for era in years_per_era.keys() if era in ttbar_files.keys()],
      "other": [{"file": f"base_other_{era}", "categories": categories_per_era[era]} for era in years_per_era.keys() if era in other_files.keys()]
    }
  }

  base_data_files = []
  base_data_add_columns = {}
  for era in years_per_era.keys():
    if era not in data_files.keys(): continue
    base_data_files += list(data_files[era].keys())
    for k in data_files[era][list(data_files[era].keys())[0]].keys():
      if k not in base_data_add_columns:
        base_data_add_columns[k] = []
      base_data_add_columns[k] += [v[k] for v in data_files[era].values()]

  # Ensure no nuisance is repeated
  nuisances = list(set(nuisances))

  # Define data calculate
  data_calculate = {
    "btm_calculate": common_calculate["btm_calculate"],
  }
  if add_classifier:
    data_calculate["btm_merged_classifier"] = common_calculate["btm_merged_classifier"]

  # Define config
  config = {
    "variables": variables,
    "pois": pois,
    "nuisances": nuisances,
    "categories": categories,
    "data_file": base_data_files,
    "data_add_columns": base_data_add_columns,
    "data_selection": f"(({pre_selection}) & ({post_selection}))",
    "data_calculate" : data_calculate,
    "inference": {
      "nuisance_constraints": [nui for nui in nuisances if not nui.startswith("fsr_")],
      "rate_parameters": rate_parameters,
      "lnN": lnN,
      "binned_fit": {}
    },
    "default_values": default_values,
    "models": models,
    "validation": validation,
    "preprocess": preprocess,
    "files": base_files
  }

  return config