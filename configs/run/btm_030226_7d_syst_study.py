import os
import copy
import glob

import numpy as np


##################################################################################

# Setup useful information
run_name = "BTM_030226_7D_syst_study"
nom_weight = "weight"
pre_selection = "((JetLepton_deltaR>0.25) & (JetLepton_ptrel>30) & (MET_pt>50))"
post_selection = "((CombinedSubJets_pt > 400) & (LeptonicTop_mass < CombinedSubJets_mass) & (CombinedSubJets_pt < 800) & (CombinedSubJets_mass > 50) & (CombinedSubJets_mass < 300) & (SubJet1_mass > 5) & (SubJet1_mass < 175) & (SubJet1_pt > 200) & (SubJet1_pt < 700) & (SubJet1_tau21 > 0.01) & (SubJet1_tau21 < 0.99) & (FatJet_tau21 > 0.05) & (FatJet_tau21 < 0.9) & (SubJet2_btagDeepB > 0) & (SubJet2_btagDeepB < 1))"
data_loc = "/vols/cms/gu18/innfer_v1/data/top_reco/260126_v2_parquet"
variables = [
  "CombinedSubJets_mass",
  "CombinedSubJets_pt",
  "SubJet1_mass",
  "SubJet1_pt",
  "SubJet1_tau21",
  "FatJet_tau21",
  "SubJet2_btagDeepB",
]
pois = ["bw_mass"]
categories = {
  "run3": "(run==3.0)"
}
years_per_category = {
  "run3": ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]
}
year_index = {
  "2016_PreVFP": 0,
  "2016_PostVFP": 1,
  "2017": 2,
  "2018": 3,
  "2022_preEE": 4,
  "2022_postEE": 5,
  "2023_preBPix": 6,
  "2023_postBPix": 7
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
for shift_mass in [166.5, 169.5, 171.5, 173.5, 175.5, 178.5]:
  shift_mass_name = str(shift_mass).replace(".", "p")
  ttbar_names[shift_mass] = [
    f"TTToSemiLeptonic{shift_mass_name}",
    f"TTTo2L2Nu{shift_mass_name}",
    f"TTToHadronic{shift_mass_name}",
  ]
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
default_values = {"bw_mass": 172.5}
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
  "stratify_to" : "sim_mass"
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
ttbar_1725_files = {}
other_files = {}
data_files = {}

# Loop through categories
for cat, years in years_per_category.items():

  # Get information for ttbar (and ttbar_1725)
  ttbar_files[cat] = {}
  ttbar_1725_files[cat] = {}
  for year in years:
    for mass, files in ttbar_names.items():
      for name in files:
        file_path = f"{data_loc}/{name}_{year}.parquet"
        if os.path.exists(file_path):
          ttbar_files[cat][file_path] = {
            "bw_mass" : 172.5,
            "sim_mass": mass,
            "run": 2.0 if cat == "run2" else 3.0,
            "year_ind": year_index[year],
          }
          if mass == 172.5:
            ttbar_1725_files[cat][file_path] = {
              "bw_mass" : 172.5,
              "sim_mass": mass,
              "run": 2.0 if cat == "run2" else 3.0,
              "year_ind": year_index[year],
            }

  # Get information for other
  other_files[cat] = {}
  for year in years:
    for name in other_names:
      file_path = f"{data_loc}/{name}_{year}.parquet"
      if os.path.exists(file_path):
        other_files[cat][file_path] = {
          "run": 2.0 if cat == "run2" else 3.0,
          "year_ind": year_index[year],
        }

  # Get information for data
  data_files[cat] = {}
  for year in years:
    for name in data_names:
      all_files = glob.glob(f"{data_loc}/{name}_{year}_*.parquet")
      for file_path in all_files:
        if os.path.exists(file_path):
          data_files[cat][file_path] = {
            "run": 2.0 if cat == "run2" else 3.0,
            "year_ind": year_index[year],
          }

  # Build precalculate
  jec_nuisances = ["AbsoluteMPFBias","FlavorQCD","TimePtEta_2023_postBPix","FlavorPureBottom"]
  jec_inputs = []
  objs = ["SubJet1", "SubJet2"]
  jec_inputs += [f"{obj}_corrFactor" for obj in objs]
  for nui in jec_nuisances:
    no_year_nui = nui.split("_20")[0]
    jec_inputs += [f"{obj}_{var}" for obj in objs for var in ["eta", "pt", "mass", "phi"]]
    if "FlavorPure" in nui:
      jec_inputs += [f"MatchedGenJet_{obj}_partonFlavour" for obj in objs]
    if nui.startswith("JER"):
      jec_inputs += [f"{obj}_smearFactor_{no_year_nui}" for obj in objs]
    else:
      jec_inputs += [f"{obj}_corrFactor_{no_year_nui}" for obj in objs]


  common_precalculate = {
    "btm_precalculate": {
      "type": "function", 
      "file": "btm_precalculate", 
      "name": "btm_precal", 
      "args": {}, 
      "inputs": ["SubJet1_tau1", "SubJet1_tau2", "FatJet_tau1", "FatJet_tau2", "BJetLep_pt", "BJetLep_eta", "BJetLep_phi", "BJetLep_mass", "LeptonSave_pt", "LeptonSave_eta", "LeptonSave_phi", "LeptonSave_mass"],
      "outputs": ["SubJet1_tau21", "FatJet_tau21", "LeptonicTop_mass", "LeptonicTop_pt"]
    },
    "btm_jec" : {
      "type": "function",
      "file": "btm_systematics",
      "name": "btm_jec",
      "args": {"years": years, "nuisances": jec_nuisances, "include_b": True, "include_b_syst": False},
      "inputs": jec_inputs,
      "outputs": ["CombinedSubJets_mass", "CombinedSubJets_pt", "SubJet1_mass", "SubJet1_pt", "LeptonicTop_mass", "LeptonicTop_pt"],
    }
  }
  ttbar_precalculate = {}
  ttbar_1725_precalculate = {}
  other_precalculate = {}

  # Build classifier nuisances
  ttbar_classifier_nuisances = jec_nuisances.copy()
  other_classifier_nuisances = jec_nuisances.copy()
  nuisances += jec_nuisances

  # Set common precalculate variables to the rest
  for k, v in common_precalculate.items():
    ttbar_precalculate[k] = v
    ttbar_1725_precalculate[k] = v
    other_precalculate[k] = v

  # Build weight shifts
  common_weight_shifts = {}

  ttbar_weight_shifts = {}
  ttbar_1725_weight_shifts = {}
  other_weight_shifts = {}

  ttbar_weight_shifts["bw_mass_leading_top"] = {
    "type": "function", 
    "file": "breit_wigner_reweighting", 
    "name": "bw_reweight", 
    "args": {"mass_to":"bw_mass", "mass_from":"sim_mass", "gen_mass": "GenTop1_mass"},
    "inputs": ["bw_mass", "sim_mass", "GenTop1_mass"],
  }
  ttbar_weight_shifts["bw_mass_subleading_top"] = {
    "type": "function", 
    "file": "breit_wigner_reweighting", 
    "name": "bw_reweight", 
    "args": {"mass_to":"bw_mass", "mass_from":"sim_mass", "gen_mass": "GenTop2_mass"},
    "inputs": ["bw_mass", "sim_mass", "GenTop2_mass"],
  }
  ttbar_weight_shifts["bw_fractions"] = {
    "type": "function", 
    "file": "breit_wigner_reweighting", 
    "name": "bw_fractions", 
    "args": {"spline_locations":"/vols/cms/sbi_top_mass/data/BTM_310126_7D/top_bw_fractions/top_bw_fraction_locations.yaml", "mass_to":"bw_mass", "mass_from":"sim_mass", "category":cat},
    "inputs": ["bw_mass", "sim_mass"],
  }
  ttbar_weight_shifts["log_fsr"] = {
    "type": "function",
    "file": "fsr_variation",
    "name": "fsr_weight",
    "args": {"fsr_value" : 0.37},
    "inputs": ["GenWeights_isr1fsr2", "GenWeights_isr1fsr0p5"],
  }

  # Set common weight shifts to the rest
  for k, v in common_weight_shifts.items():
    ttbar_weight_shifts[k] = v
    ttbar_1725_weight_shifts[k] = v
    other_weight_shifts[k] = v

  # Make base files
  base_files[f"base_ttbar_{cat}"] = {
    "inputs": list(ttbar_files[cat].keys()),
    "add_columns": {k: [v[k] for v in ttbar_files[cat].values()] for k in list(ttbar_files[cat].values())[0].keys()},
    "selection": pre_selection,
    "weight": nom_weight,
    "parameters": [],
    "pre_calculate": ttbar_precalculate,
    "post_calculate_selection": post_selection,
    "weight_shifts": ttbar_weight_shifts
  }
  base_files[f"base_ttbar_1725_{cat}"] = {
    "inputs": list(ttbar_1725_files[cat].keys()),
    "add_columns": {k: [v[k] for v in ttbar_1725_files[cat].values()] for k in list(ttbar_1725_files[cat].values())[0].keys()},
    "selection": pre_selection,
    "weight": nom_weight,
    "parameters": [],
    "pre_calculate": ttbar_1725_precalculate,
    "post_calculate_selection": post_selection,
    "weight_shifts": ttbar_1725_weight_shifts
  }
  base_files[f"base_other_{cat}"] = {
    "inputs": list(other_files[cat].keys()),
    "add_columns": {k: [v[k] for v in other_files[cat].values()] for k in list(other_files[cat].values())[0].keys()},
    "selection": pre_selection,
    "weight": nom_weight,
    "parameters": [],
    "pre_calculate": other_precalculate,
    "post_calculate_selection": post_selection,
    "weight_shifts": other_weight_shifts
  }

  # Make density models
  ttbar_center = 172.5
  ttbar_radius = 5.0
  sigma_out = 1.0
  models["ttbar"]["density_models"] += [
    {
      "parameters": ["bw_mass"],
      "file": f"base_ttbar_{cat}",
      "shifts": {
        "bw_mass": {"type": "flat_top", "range": [ttbar_center - ttbar_radius, ttbar_center + ttbar_radius], "other": {"sigma_out": sigma_out}}
      },
      "n_copies": 10,
      "categories": [cat],
    }
  ]
  models["other"]["density_models"] += [
    {
      "parameters": [],
      "file": f"base_other_{cat}",
      "shifts": {},
      "n_copies": 1,
      "categories": [cat],
    },
  ]

  # Make classifier models
  models["ttbar"]["classifier_models"] += [{"parameter":k, "file":f"base_ttbar_{cat}", "shifts":{k: {"type": "flat_top", "range": [-3.0,3.0], "other": {"sigma_out": 0.6}}}, "n_copies":3, "categories": [cat]} for k in ttbar_classifier_nuisances]
  models["other"]["classifier_models"] += [{"parameter":k, "file":f"base_other_{cat}", "shifts":{k: {"type": "flat_top", "range": [-3.0,3.0], "other": {"sigma_out": 0.6}}}, "n_copies":3, "categories": [cat]} for k in other_classifier_nuisances]

  # Make yields
  models["ttbar"]["yields"] += [{"file": f"base_ttbar_1725_{cat}", "categories": [cat]}]
  models["other"]["yields"] += [{"file": f"base_other_{cat}", "categories": [cat]}]

# Define validation loop for POIs
bw_mass_val_points = [171.5, 172.0, 172.5, 173.0, 173.5]
validation_loop = [{"bw_mass" : m} for m in bw_mass_val_points]
validation = {
  "loop": validation_loop,
  "files": {
    "ttbar": [{"file": f"base_ttbar_{cat}", "categories": [cat]} for cat in categories.keys()],
    "other": [{"file": f"base_other_{cat}", "categories": [cat]} for cat in categories.keys()]
  }
}

base_data_files = []
base_data_add_columns = {}
for cat in categories.keys():
  base_data_files += list(data_files[cat].keys())
  for k in data_files[cat][list(data_files[cat].keys())[0]].keys():
    if k not in base_data_add_columns:
      base_data_add_columns[k] = []
    base_data_add_columns[k] += [v[k] for v in data_files[cat].values()]

# Ensure no nuisance is repeated
nuisances = list(set(nuisances))

# Define config
config = {
  "name": run_name,
  "variables": variables,
  "pois": pois,
  "nuisances": nuisances,
  "categories": categories,
  "data_file": base_data_files,
  "data_add_columns": base_data_add_columns,
  "data_selection": f"(({pre_selection}) & ({post_selection}))",
  "data_calculate" : {"btm_precalculate": common_precalculate["btm_precalculate"]},
  "inference": {
    "nuisance_constraints": nuisances,
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