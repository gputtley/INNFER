import os
import copy
import numpy as np


##################################################################################

# Setup useful information
run_name = "BTM_141025_2d_stat_only"
nom_weight = "weight"
pre_selection = "((JetLepton_deltaR>0.25) & (JetLepton_ptrel>30))"
post_selection = "(CombinedSubJets_pt > 400)"
data_loc = "/vols/cms/gu18/innfer_v1/data/top_reco/091025"
variables = [
  "CombinedSubJets_mass",
  "CombinedSubJets_pt"
]
pois = ["bw_mass"]
categories = {
  # "run2": "(run==2.0)",
  "run3": "(run==3.0)"
}
years_per_category = {
  "run2": ["2016_PreVFP", "2016_PostVFP", "2017", "2018"],
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
  #"WW",
  #"WZ",
  #"ZZ",
  #"DY",
]
stat_only = True
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
  # flavour
  "FlavorPureGluon" : {"Correlation" : 1},
  "FlavorPureQuark" : {"Correlation" : 1},
  "FlavorPureCharm" : {"Correlation" : 1},
  "FlavorPureBottom" : {"Correlation" : 1},
}
jer_uncerts_regions = {
  "eta_lt_1p93": "eta < 1.93",
  "eta_1p93_to_2p5": "(eta >= 1.93) & (eta < 2.5)",
  "eta_2p5_to_3p0_pt_0_to_50": "(eta >= 2.5) & (eta < 3.0) & (pt < 50)",
  "eta_2p5_to_3p0_pt_gt_50": "(eta >= 2.5) & (eta < 3.0) & (pt >= 50)",
  "eta_3p0_to_5p0_pt_0_to_50": "(eta >= 3.0) & (eta < 5.0) & (pt < 50)",
  "eta_3p0_to_5p0_pt_gt_50": "(eta >= 3.0) & (eta < 5.0) & (pt >= 50)",
}
rate_parameters = ["ttbar"]
default_values = {"bw_mass": 172.5}
preprocess = {
  "train_test_val_split": "0.8:0.1:0.1",
  "save_extra_columns": {
    "ttbar": ["sim_mass"],
    "other": []
  }
}
lnN = {
  "ttbar" : [],
  "other" : []
}



##################################################################################

# Define bw shifts
def width(m):
  return f"((0.0270*{m}) - 3.3455)"
def bw(m, gen_m):
  return f"((({gen_m}!=-999)*(1/((({gen_m}**2)-({m}**2))**2 + (({m}*{width(m)})**2))) + (1.0*({gen_m}==-999))))"
def bw_reweight(mf, mi, gen_m):
  return f"({bw(mf, gen_m)}/{bw(mi, gen_m)})"

# JER uncertainty shifts
def jer_shift_function(nui_name, var_name, collection_name, region_cut=None):
  #shift = f"{collection_name}_{var_name}"
  shift = f" * ( ({nui_name}>=0) *  ( ( ({nui_name} * ({collection_name}_smearFactor_up-{collection_name}_smearFactor)) + {collection_name}_smearFactor)/{collection_name}_smearFactor)"
  shift += f" + ({nui_name}<0) * ( ( (-{nui_name} * ({collection_name}_smearFactor_down-{collection_name}_smearFactor)) + {collection_name}_smearFactor)/{collection_name}_smearFactor) )"
  if region_cut is not None:
    region_cut = region_cut.replace("eta", f"{collection_name}_eta").replace("pt", f"{collection_name}_pt")
    shift += f" * ( ({region_cut}) + ( 1.0*(~({region_cut})) ) )"
  return shift

# Define dictionaries
base_files = {}
sub_models = {"density_models":[], "classifier_models":[], "regression_models":[], "yields":[]}
models = {"ttbar": copy.deepcopy(sub_models), "other": copy.deepcopy(sub_models)}
nuisances = []
ttbar_files = {}  
ttbar_1725_files = {}
other_files = {}

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
            "sim_mass": mass,
            "run": 2.0 if cat == "run2" else 3.0,
            "year_ind": year_index[year],
          }
          if mass == 172.5:
            ttbar_1725_files[cat][file_path] = {
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

  # Build precalculate
  common_precalculate = {}
  ttbar_precalculate = {}
  ttbar_1725_precalculate = {}
  other_precalculate = {}

  # Build classifier nuisances
  ttbar_classifier_nuisances = []
  other_classifier_nuisances = []

  # Systematic uncertainties for feature morphing
  if not stat_only:

    # JEC/flavour shifts
    for var in ["SubJet1_pt","SubJet1_mass","SubJet2_pt","SubJet2_mass"]:
      common_precalculate[var] = copy.deepcopy(var)
    for name, info in jec_uncert.items():
      if info["Correlation"] == 1:
        corr_years = [years]
        syst_names = [name]
        scalings = 1.0
      elif info["Correlation"] == 0:
        corr_years = [[yr] for yr in years]
        syst_names = [f"{name}_{yr}" for yr in years]
        scalings = 1.0
      elif info["Correlation"] == 0.5:
        corr_years = [years] + [[yr] for yr in years]
        syst_names = [name] + [f"{name}_{yr}" for yr in years]
        scalings = 0.5
      for ind in range(len(syst_names)):
        nuisances.append(syst_names[ind])
        ttbar_classifier_nuisances.append(syst_names[ind])
        other_classifier_nuisances.append(syst_names[ind])
        SubJet1_factor = f"*(1 + ({scalings}*{syst_names[ind]}*SubJet1_corrFactor_{name}/SubJet1_corrFactor))"
        common_precalculate["SubJet1_pt"] += SubJet1_factor
        common_precalculate["SubJet1_mass"] += SubJet1_factor
        SubJet2_factor = f"*(1 + ({scalings}*{syst_names[ind]}*SubJet2_corrFactor_{name}/SubJet2_corrFactor))"
        common_precalculate["SubJet2_pt"] += SubJet2_factor
        common_precalculate["SubJet2_mass"] += SubJet2_factor

    # JER shifts
    for name, region in jer_uncerts_regions.items():
      for yr in years:
        nuisance_name = f"JER_{name}_{yr}"
        nuisances.append(nuisance_name)
        ttbar_classifier_nuisances.append(nuisance_name)
        other_classifier_nuisances.append(nuisance_name)
        common_precalculate["SubJet1_pt"] += jer_shift_function(nuisance_name, "pt", "SubJet1", region_cut=region)
        common_precalculate["SubJet1_mass"] += jer_shift_function(nuisance_name, "mass", "SubJet1", region_cut=region)
        common_precalculate["SubJet2_pt"] += jer_shift_function(nuisance_name, "pt", "SubJet2", region_cut=region)
        common_precalculate["SubJet2_mass"] += jer_shift_function(nuisance_name, "mass", "SubJet2", region_cut=region)


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
  ttbar_weight_shifts["bw_mass"] = f"{bw_reweight('bw_mass','sim_mass','GenTop1_mass')}*{bw_reweight('bw_mass','sim_mass','GenTop2_mass')}"

  # Systematic uncertainties for weight variations
  if not stat_only:
    pass

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
  models["ttbar"]["density_models"] += [
    {
      "parameters": ["bw_mass"],
      "file": f"base_ttbar_{cat}",
      "shifts": {
        "bw_mass": {"type": "continuous", "range": [169.5,175.5]}
      },
      "n_copies": 5,
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
  models["ttbar"]["classifier_models"] += [{"parameter":k, "file":f"base_ttbar_{cat}", "shifts":{k: {"type": "continuous", "range": [-5.0,5.0]}}, "n_copies":3, "categories": [cat]} for k in ttbar_classifier_nuisances]
  models["other"]["classifier_models"] += [{"parameter":k, "file":f"base_other_{cat}", "shifts":{k: {"type": "continuous", "range": [-5.0,5.0]}}, "n_copies":3, "categories": [cat]} for k in other_classifier_nuisances]

  # Make yields
  models["ttbar"]["yields"] += [{"file": f"base_ttbar_1725_{cat}", "categories": [cat]}]
  models["other"]["yields"] += [{"file": f"base_other_{cat}", "categories": [cat]}]

# Define validation loop for POIs
bw_mass_val_points = [171.5, 172.5, 173.5]
validation_loop = [{"bw_mass" : m} for m in bw_mass_val_points]

# Define validation loop for nuisances
if not stat_only:
  n_nuisance_val_points = 10
  possible_nuisance_vals = [-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0]
  np.random.seed(42)
  for _ in range(n_nuisance_val_points):
    val_info = {nui: float(np.random.choice(possible_nuisance_vals)) for nui in nuisances}
    val_info["bw_mass"] = float(np.random.choice(bw_mass_val_points))
    validation_loop.append(val_info)

validation = {
  "loop": validation_loop,
  "files": {
    "ttbar": [{"file": f"base_ttbar_{cat}", "categories": [cat]} for cat in categories.keys()],
    "other": [{"file": f"base_other_{cat}", "categories": [cat]} for cat in categories.keys()]
  }
}

# Define config
config = {
  "name": run_name,
  "variables": variables,
  "pois": pois,
  "nuisances": nuisances,
  "categories": categories,
  "data_file": None,
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