import os
import copy
import glob
import socket

import numpy as np

from btm_experimental_systematics import get_jec_nuisances

def make_common_config(run_name, categories, categories_per_era):

  ### Define core variables for config ###

  nom_weight = "weight"
  pre_selection = "((ClosestJetWithLeptonRemoved_deltaR>0.25) | (ClosestJetWithLeptonRemoved_ptrel>30))"
  post_selection = "((CombinedSubJets_pt > 400) & (LeptonicTop_mass < CombinedSubJets_mass) & (MET_pt>50) & (BJetLep_pt>30) & (CombinedSubJets_pt < 800) & (CombinedSubJets_mass > 50) & (CombinedSubJets_mass < 300) & (SubJet1_mass > 5) & (SubJet1_mass < 175) & (SubJet1_pt > 200) & (SubJet1_pt < 700) & (SubJet1_tau21 > 0.01) & (SubJet1_tau21 < 0.99) & (FatJet_tau21 > 0.05) & (FatJet_tau21 < 0.9) & (SubJet2_btagDeepB > 0) & (SubJet2_btagDeepB < 1))"

  host = socket.getfqdn()
  if host.endswith(".cern.ch"):
    data_loc = "/eos/user/g/guttley/pc_output/060526_parquet" # cern
  else:
    data_loc = "/vols/cms/gu18/innfer_v1/data/top_reco/060526_parquet" #imperial

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

  years_per_era = {
    "run2": ["2016_PreVFP", "2016_PostVFP", "2017", "2018"],
    "2223": ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]
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
  ttbar_semileptonic_modelling_syst_names = {
    "TTToSemiLeptonic_CR1" : 1, # Numbering for whether it is a systemtic file or note
    "TTToSemiLeptonic_CR2" : 2,
    "TTToSemiLeptonic_hdamp_Up" : 3,
    "TTToSemiLeptonic_hdamp_Down" : 4,
    "TTToSemiLeptonic_ue_Up" : 5,
    "TTToSemiLeptonic_ue_Down" : 6,
    "TTToSemiLeptonic_ERDOn" : 7
  }
  # Need a TTToSemiLeptonic flag and then a is syst file flag for each type of systematic
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
  default_values = {"bw_mass" : 172.5}
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
    "no_nuisance_yield_effect" : ["fsr"]
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

  # Define the lumi uncertainties
  lumi_uncerts = {
    "lumi_13TeV_1516" : {"2016_PreVFP" : 1.0118, "2016_PostVFP" : 1.0118},
    "lumi_13TeV_151617" : {"2016_PreVFP" : 1.0004, "2016_PostVFP" : 1.0004, "2017" : 1.0055},
    "lumi_13TeV_15161718" : {"2016_PreVFP" : 1.0035, "2016_PostVFP" : 1.0035, "2017" : 1.0061, "2018" : 1.0084},
    "lumi_13p6TeV_2223" : {"2022_preEE" : 1.0138, "2022_postEE" : 1.0138, "2023_preBPix" : 1.0017, "2023_postBPix" : 1.0017},
    "lumi_13p6TeV_23" : {"2023_preBPix" : 1.0127, "2023_postBPix" : 1.0127},
  }

  # Loop through categories
  for cat, years in years_per_era.items():

    # Get information for ttbar (and ttbar_yield)
    ttbar_files[cat] = {}
    ttbar_yield_files[cat] = {}
    for year in years:
      for mass, files in ttbar_names.items():
        for name in files:
          file_path = f"{data_loc}/{name}_{year}.parquet"
          if os.path.exists(file_path):
            ttbar_files[cat][file_path] = {
              "bw_mass" : 172.5,
              "sim_mass": mass,
              "year_ind": year_index[year],
              "top_process" : 1.0,
              "TTtoSL_modelling_syst" : 0.0,
              "could_be_TTToSL" : 1.0 if name.startswith("TTToSemiLeptonic") or name in ["TTMtt700To1000","TTMtt1000"] else 0.0,
            }
            for k, v in lumi_uncerts.items():
              if year in v:
                ttbar_files[cat][file_path][f"{k}_factor"] = v[year]
              else:
                ttbar_files[cat][file_path][f"{k}_factor"] = 1.0

            if mass == 172.5:
              ttbar_yield_files[cat][file_path] = {
                "bw_mass" : 172.5,
                "sim_mass": mass,
                "year_ind": year_index[year],
                "top_process" : 1.0,
                "TTtoSL_modelling_syst" : 0.0,
                "could_be_TTToSL" : 1.0 if name.startswith("TTToSemiLeptonic") or name in ["TTMtt700To1000","TTMtt1000"] else 0.0,
              }
              for k, v in lumi_uncerts.items():
                if year in v:
                  ttbar_yield_files[cat][file_path][f"{k}_factor"] = v[year]
                else:
                  ttbar_yield_files[cat][file_path][f"{k}_factor"] = 1.0

    # Add ttbar semileptonic modelling systematics
    for syst_name, syst_value in ttbar_semileptonic_modelling_syst_names.items():
      for year in years:
        file_path = f"{data_loc}/{syst_name}_{year}.parquet"
        if os.path.exists(file_path):
          ttbar_files[cat][file_path] = {
            "bw_mass" : 172.5,
            "sim_mass": 172.5,
            "year_ind": year_index[year],
            "top_process" : 1.0,
            "TTtoSL_modelling_syst" : syst_value,
            "could_be_TTToSL" : 1.0 if syst_name.startswith("TTToSemiLeptonic") or syst_name in ["TTMtt700To1000","TTMtt1000"] else 0.0,
          }
          for k, v in lumi_uncerts.items():
            if year in v:
              ttbar_files[cat][file_path][f"{k}_factor"] = v[year]
            else:
              ttbar_files[cat][file_path][f"{k}_factor"] = 1.0
          ttbar_yield_files[cat][file_path] = ttbar_files[cat][file_path]

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
          for k, v in lumi_uncerts.items():
            if year in v:
              other_files[cat][file_path][f"{k}_factor"] = v[year]
            else:
              other_files[cat][file_path][f"{k}_factor"] = 1.0

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



    ### Calculate (at PreProcess step) ###

    # Get jec nuisances
    jec_nuisances = get_jec_nuisances(years=years_per_era[cat])
    jec_inputs = []
    objs = ["SubJet1", "SubJet2", "BJetLep", "MET"]
    jec_inputs += [f"{obj}_corrFactor" for obj in objs if obj not in ["MET"]]
    for nui in jec_nuisances:
      no_year_nui = nui.split("_20")[0]
      jec_inputs += [f"{obj}_{var}" for obj in objs for var in ["eta", "pt", "mass", "phi"] if obj not in ["MET"]]
      jec_inputs += ["MET_pt"]
      if "FlavorPure" in nui:
        jec_inputs += [f"MatchedGenJet_{obj}_partonFlavour" for obj in objs if obj not in ["MET"]]
      if nui.startswith("JER"):
        jec_inputs += [f"{obj}_smearFactor" for obj in objs if f"{obj}_smearFactor" not in jec_inputs and obj not in ["MET"]]
        jec_inputs += [f"{obj}_smearFactor_up" for obj in objs if f"{obj}_smearFactor_up" not in jec_inputs]
        jec_inputs += [f"{obj}_smearFactor_down" for obj in objs if f"{obj}_smearFactor_down" not in jec_inputs]
      else:
        jec_inputs += [f"{obj}_corrFactor_{no_year_nui}" for obj in objs]

    common_calculate = {
      "btm_calculate": {
        "type": "function", 
        "file": "btm_calculate", 
        "name": "btm_cal", 
        "args": {}, 
        "inputs": ["SubJet1_tau1", "SubJet1_tau2", "FatJet_tau1", "FatJet_tau2", "BJetLep_pt", "BJetLep_eta", "BJetLep_phi", "BJetLep_mass", "LeptonSave_pt", "LeptonSave_eta", "LeptonSave_phi", "LeptonSave_mass", "Extra_BTagWeightCorrection_up", "Extra_BTagWeightCorrection_down"],
        "outputs": ["SubJet1_tau21", "FatJet_tau21", "LeptonicTop_mass", "LeptonicTop_pt", "Extra_BTagWeightCorrection"]
      },
      "btm_jec" : {
        "type": "function",
        "file": "btm_experimental_systematics",
        "name": "btm_jec",
        "args": {"years": years, "nuisances": jec_nuisances, "include_b": True, "include_b_syst": True, "include_met_syst": True},
        "inputs": jec_inputs,
        "outputs": ["CombinedSubJets_mass", "CombinedSubJets_pt", "SubJet1_mass", "SubJet1_pt", "LeptonicTop_mass", "LeptonicTop_pt", "MET_pt"],
      },
      "btm_merged_classifier": {
        "type": "function",
        "file": "btm_merged_classifier",
        "name": "btm_merged_classifier",
        "args": {},
        "inputs": ["CombinedSubJets_pt", "SubJet1_mass", "SubJet1_pt", "SubJet1_tau21", "FatJet_tau21", "SubJet2_btagDeepB"],
        "outputs": ["btm_merged_classifier", "btm_merged_classifier_pass"]
      }
    }
    ttbar_calculate = {}
    ttbar_yield_calculate = {}
    other_calculate = {}
    other_yield_calculate = {}

    # Add to nuisances
    ttbar_classifier_nuisances = jec_nuisances.copy()
    other_classifier_nuisances = jec_nuisances.copy()
    nuisances += jec_nuisances

    # Set common calculate variables to the rest
    for k, v in common_calculate.items():
      ttbar_calculate[k] = v
      ttbar_yield_calculate[k] = v
      other_calculate[k] = v
      other_yield_calculate[k] = v



    ### Weight shifts (at PreProcess step) ###

    # Build weight shifts
    common_weight_shifts = {
      "fsr" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "fsr", "up_weight_column_name": "GenWeights_isr1fsr2", "down_weight_column_name": "GenWeights_isr1fsr0p5", "wt_name": "wt", "mask": "(top_process == 1)"},
        "inputs": ["GenWeights_isr1fsr2", "GenWeights_isr1fsr0p5", "top_process", "fsr"],
      },
      "muF" : {
        "type": "function", 
        "file": "asym_log_normal", 
        "name": "two_weight_variation", 
        "args": {"nuisance_column_name": "muF", "up_weight_column_name": "GenWeights_muF2muR1", "down_weight_column_name": "GenWeights_muF0p5muR1", "wt_name": "wt", "mask": "(top_process == 1)"},
        "inputs": ["GenWeights_muF0p5muR1", "GenWeights_muF2muR1", "top_process", "muF"],
      },
      "muR" : {
        "type": "function", 
        "file": "asym_log_normal", 
        "name": "two_weight_variation", 
        "args": {"nuisance_column_name": "muR", "up_weight_column_name": "GenWeights_muF1muR2", "down_weight_column_name": "GenWeights_muF1muR0p5", "wt_name": "wt"},
        "inputs": ["GenWeights_muF1muR0p5", "GenWeights_muF1muR2", "top_process", "muR"],
      },
      "pdf" : {
        "type": "function", 
        "file": "asym_log_normal", 
        "name": "one_weight_variation", 
        "args": {"nuisance_column_name": "pdf", "weight_column_name": "GenWeights_pdf_rmse", "wt_name": "wt", "add_one_to_weight": True},
        "inputs": ["GenWeights_pdf_rmse", "pdf"],
      },
      f"btag_weight_correlated_{cat}" : {
        "type": "function",
        "file": "asym_log_normal", 
        "name": "two_weight_variation", 
        "args": {"nuisance_column_name": f"btag_weight_correlated_{cat}", "up_weight_column_name": "Extra_BTagWeightCorrection_up_correlated", "down_weight_column_name": "Extra_BTagWeightCorrection_down_correlated", "nominal_weight_column_name": "Extra_BTagWeightCorrection", "wt_name": "wt"},
        "inputs": ["Extra_BTagWeightCorrection_down_correlated", "Extra_BTagWeightCorrection_up_correlated", "Extra_BTagWeightCorrection", f"btag_weight_correlated_{cat}"],      
      },
      f"btag_shape_hf_{cat}" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": f"btag_shape_hf_{cat}", "up_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_up_hf", "down_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_down_hf", "nominal_weight_column_name": f"Extra_BTagShapeCorrectionSubjets", "wt_name": "wt"},
        "inputs": [f"Extra_BTagShapeCorrectionSubjets_down_hf", f"Extra_BTagShapeCorrectionSubjets_up_hf", f"Extra_BTagShapeCorrectionSubjets", f"btag_shape_hf_{cat}"],
      },
      f"btag_shape_lf_{cat}" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": f"btag_shape_lf_{cat}", "up_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_up_lf", "down_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_down_lf", "nominal_weight_column_name": f"Extra_BTagShapeCorrectionSubjets", "wt_name": "wt"},
        "inputs": [f"Extra_BTagShapeCorrectionSubjets_down_lf", f"Extra_BTagShapeCorrectionSubjets_up_lf", f"Extra_BTagShapeCorrectionSubjets", f"btag_shape_lf_{cat}"],
      },
      f"btag_shape_hfstats1_{cat}" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": f"btag_shape_hfstats1_{cat}", "up_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_up_hfstats1", "down_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_down_hfstats1", "nominal_weight_column_name": f"Extra_BTagShapeCorrectionSubjets", "wt_name": "wt"},
        "inputs": [f"Extra_BTagShapeCorrectionSubjets_down_hfstats1", f"Extra_BTagShapeCorrectionSubjets_up_hfstats1", f"Extra_BTagShapeCorrectionSubjets", f"btag_shape_hfstats1_{cat}"],
      },
      f"btag_shape_hfstats2_{cat}" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": f"btag_shape_hfstats2_{cat}", "up_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_up_hfstats2", "down_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_down_hfstats2", "nominal_weight_column_name": f"Extra_BTagShapeCorrectionSubjets", "wt_name": "wt"},
        "inputs": [f"Extra_BTagShapeCorrectionSubjets_down_hfstats2", f"Extra_BTagShapeCorrectionSubjets_up_hfstats2", f"Extra_BTagShapeCorrectionSubjets", f"btag_shape_hfstats2_{cat}"],
      },
      f"btag_shape_lfstats1_{cat}" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": f"btag_shape_lfstats1_{cat}", "up_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_up_lfstats1", "down_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_down_lfstats1", "nominal_weight_column_name": f"Extra_BTagShapeCorrectionSubjets", "wt_name": "wt"},
        "inputs": [f"Extra_BTagShapeCorrectionSubjets_down_lfstats1", f"Extra_BTagShapeCorrectionSubjets_up_lfstats1", f"Extra_BTagShapeCorrectionSubjets", f"btag_shape_lfstats1_{cat}"],
      },
      f"btag_shape_lfstats2_{cat}" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": f"btag_shape_lfstats2_{cat}", "up_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_up_lfstats2", "down_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_down_lfstats2", "nominal_weight_column_name": f"Extra_BTagShapeCorrectionSubjets", "wt_name": "wt"},
        "inputs": [f"Extra_BTagShapeCorrectionSubjets_down_lfstats2", f"Extra_BTagShapeCorrectionSubjets_up_lfstats2", f"Extra_BTagShapeCorrectionSubjets", f"btag_shape_lfstats2_{cat}"],
      },
      f"btag_shape_cferr1_{cat}" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": f"btag_shape_cferr1_{cat}", "up_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_up_cferr1", "down_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_down_cferr1", "wt_name": "wt"},
        "inputs": [f"Extra_BTagShapeCorrectionSubjets_down_cferr1", f"Extra_BTagShapeCorrectionSubjets_up_cferr1", f"btag_shape_cferr1_{cat}"],
      },
      f"btag_shape_cferr2_{cat}" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": f"btag_shape_cferr2_{cat}", "up_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_up_cferr2", "down_weight_column_name": f"Extra_BTagShapeCorrectionSubjets_down_cferr2", "wt_name": "wt"},
        "inputs": [f"Extra_BTagShapeCorrectionSubjets_down_cferr2", f"Extra_BTagShapeCorrectionSubjets_up_cferr2", f"btag_shape_cferr2_{cat}"],
      },
      "pileup" : {
        "type": "function", 
        "file": "asym_log_normal", 
        "name": "two_weight_variation", 
        "args": {"nuisance_column_name": "pileup", "up_weight_column_name": "Extra_pileup_up", "down_weight_column_name": "Extra_pileup_down", "nominal_weight_column_name": "Extra_pileup", "wt_name": "wt"},
        "inputs": ["Extra_pileup_down", "Extra_pileup_up", "Extra_pileup", "pileup"],
      },
      "top_pt_reweighting" : {
        "type": "function", 
        "file": "asym_log_normal", 
        "name": "one_weight_variation",
        "args": {"nuisance_column_name": "top_pt_reweighting", "weight_column_name": "Extra_TopPTReweighting", "wt_name": "wt", "mask": "(top_process == 1)"},
        "inputs": ["Extra_TopPTReweighting", "top_process", "top_pt_reweighting"],
      },
      "electron_id" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "electron_id", "up_weight_column_name": "Extra_sf_ele_id_custom_up", "down_weight_column_name": "Extra_sf_ele_id_custom_down", "nominal_weight_column_name": "Extra_sf_ele_id_custom", "wt_name": "wt"},
        "inputs": ["Extra_sf_ele_id_custom_down", "Extra_sf_ele_id_custom_up", "Extra_sf_ele_id_custom", "electron_id"],
      },
      "electron_reco" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "electron_reco", "up_weight_column_name": "Extra_sf_ele_reco_custom_up", "down_weight_column_name": "Extra_sf_ele_reco_custom_down", "nominal_weight_column_name": "Extra_sf_ele_reco_custom", "wt_name": "wt"},
        "inputs": ["Extra_sf_ele_reco_custom_down", "Extra_sf_ele_reco_custom_up", "Extra_sf_ele_reco_custom", "electron_reco"],
      },
      "electron_trigger" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "electron_trigger", "up_weight_column_name": "Extra_sf_ele_trigger_custom_up", "down_weight_column_name": "Extra_sf_ele_trigger_custom_down", "nominal_weight_column_name": "Extra_sf_ele_trigger_custom", "wt_name": "wt"},
        "inputs": ["Extra_sf_ele_trigger_custom_down", "Extra_sf_ele_trigger_custom_up", "Extra_sf_ele_trigger_custom", "electron_trigger"],
      },
      "muon_id" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "muon_id", "up_weight_column_name": "Extra_sf_mu_id_custom_up", "down_weight_column_name": "Extra_sf_mu_id_custom_down", "nominal_weight_column_name": "Extra_sf_mu_id_custom", "wt_name": "wt"},
        "inputs": ["Extra_sf_mu_id_custom_down", "Extra_sf_mu_id_custom_up", "Extra_sf_mu_id_custom", "muon_id"],
      },
      "muon_iso" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "muon_iso", "up_weight_column_name": "Extra_sf_mu_iso_custom_up", "down_weight_column_name": "Extra_sf_mu_iso_custom_down", "nominal_weight_column_name": "Extra_sf_mu_iso_custom", "wt_name": "wt"},
        "inputs": ["Extra_sf_mu_iso_custom_down", "Extra_sf_mu_iso_custom_up", "Extra_sf_mu_iso_custom", "muon_iso"],
      },
      "muon_trigger" : {
        "type": "function",
        "file": "asym_log_normal",
        "name": "two_weight_variation",
        "args": {"nuisance_column_name": "muon_trigger", "up_weight_column_name": "Extra_sf_mu_trigger_custom_up", "down_weight_column_name": "Extra_sf_mu_trigger_custom_down", "nominal_weight_column_name": "Extra_sf_mu_trigger_custom", "wt_name": "wt"},
        "inputs": ["Extra_sf_mu_trigger_custom_down", "Extra_sf_mu_trigger_custom_up", "Extra_sf_mu_trigger_custom", "muon_trigger"],
      },
    }

    ttbar_weight_shifts = {
      "hdamp" : {
        "type": "function",
        "file": "btm_modelling_systematics",
        "name": "three_point_variation_weight",
        "args": {"nuisance_column_name": "hdamp", "wt_name": "wt", "total_mask": "((could_be_TTToSL == 1) & (GenTT_count_l == 1))", "mask_sample_nominal" : "(TTtoSL_modelling_syst == 0)", "mask_sample_p1" : "(TTtoSL_modelling_syst == 3)", "mask_sample_m1" : "(TTtoSL_modelling_syst == 4)"},
        "inputs": ["TTtoSL_modelling_syst", "GenTT_count_l", "hdamp"],
      },
      "ue" : {
        "type": "function",
        "file": "btm_modelling_systematics",
        "name": "three_point_variation_weight",
        "args": {"nuisance_column_name": "ue", "wt_name": "wt", "total_mask": "((could_be_TTToSL == 1) & (GenTT_count_l == 1))", "mask_sample_nominal" : "(TTtoSL_modelling_syst == 0)", "mask_sample_p1" : "(TTtoSL_modelling_syst == 5)", "mask_sample_m1" : "(TTtoSL_modelling_syst == 6)"},
        "inputs": ["TTtoSL_modelling_syst", "GenTT_count_l", "ue"],
      },
      "CR1" : {
        "type": "function",
        "file": "btm_modelling_systematics",
        "name" : "two_point_variation_weight",
        "args": {"nuisance_column_name": "CR1", "wt_name": "wt", "total_mask": "((could_be_TTToSL == 1) & (GenTT_count_l == 1))", "mask_sample_nominal" : "(TTtoSL_modelling_syst == 0)", "mask_sample_p1" : "(TTtoSL_modelling_syst == 1)"},
        "inputs": ["TTtoSL_modelling_syst", "GenTT_count_l", "CR1"],
      },
      "CR2" : {
        "type": "function",
        "file": "btm_modelling_systematics",
        "name" : "two_point_variation_weight",
        "args": {"nuisance_column_name": "CR2", "wt_name": "wt", "total_mask": "((could_be_TTToSL == 1) & (GenTT_count_l == 1))", "mask_sample_nominal" : "(TTtoSL_modelling_syst == 0)", "mask_sample_p1" : "(TTtoSL_modelling_syst == 2)"},
        "inputs": ["TTtoSL_modelling_syst", "GenTT_count_l", "CR2"],
      },
      "ERDOn" : {
        "type": "function",
        "file": "btm_modelling_systematics",
        "name" : "two_point_variation_weight",
        "args": {"nuisance_column_name": "ERDOn", "wt_name": "wt", "total_mask": "((could_be_TTToSL == 1) & (GenTT_count_l == 1))", "mask_sample_nominal" : "(TTtoSL_modelling_syst == 0)", "mask_sample_p1" : "(TTtoSL_modelling_syst == 7)"},
        "inputs": ["TTtoSL_modelling_syst", "GenTT_count_l", "ERDOn"],
      },
      "isr_ttbar" : {
        "type": "function", 
        "file": "asym_log_normal", 
        "name": "two_weight_variation", 
        "args": {"nuisance_column_name": "isr_ttbar", "up_weight_column_name": "GenWeights_isr2fsr1", "down_weight_column_name": "GenWeights_isr0p5fsr1", "wt_name": "wt", "mask": "(top_process == 1)"},
        "inputs": ["GenWeights_isr2fsr1", "GenWeights_isr0p5fsr1", "top_process", "isr_ttbar"],
      },
    }
    ttbar_yield_weight_shifts = copy.deepcopy(ttbar_weight_shifts)

    for k, v in lumi_uncerts.items():
      yr_in_cat = False
      for yr in years:
        if yr in v:
          yr_in_cat = True
          break
      if yr_in_cat:
        common_weight_shifts[k] = {
          "type": "function", 
          "file": "asym_log_normal", 
          "name": "one_weight_variation", 
          "args": {"nuisance_column_name": k, "weight_column_name": f"{k}_factor", "wt_name": "wt"},
          "inputs": [k, f"{k}_factor"],
        }
    for yr in years:
      common_weight_shifts[f"btag_weight_{yr}"] = {
        "type": "function",
        "file": "asym_log_normal", 
        "name": "two_weight_variation", 
        "args": {"nuisance_column_name": f"btag_weight_{yr}", "up_weight_column_name": f"Extra_BTagWeightCorrection_up", "down_weight_column_name": f"Extra_BTagWeightCorrection_down", "nominal_weight_column_name": f"Extra_BTagWeightCorrection", "wt_name": "wt", "mask": f"((top_process == 1) & (year_ind == {year_index[yr]}))"},
        "inputs": [f"Extra_BTagWeightCorrection_down", f"Extra_BTagWeightCorrection_up", f"Extra_BTagWeightCorrection", f"btag_weight_{yr}"],
      }
    if "run2" in cat:
      common_weight_shifts["prefiring"] = {
        "type": "function", 
        "file": "asym_log_normal", 
        "name": "two_weight_variation", 
        "args": {"nuisance_column_name": "prefiring", "up_weight_column_name": "Extra_prefiring_up", "down_weight_column_name": "Extra_prefiring_down", "nominal_weight_column_name": "Extra_prefiring", "wt_name": "wt"},
        "inputs": ["Extra_prefiring_down", "Extra_prefiring_up", "Extra_prefiring", "prefiring"],
      }
    other_weight_shifts = {
      "isr_st" : {
        "type": "function", 
        "file": "asym_log_normal", 
        "name": "two_weight_variation", 
        "args": {"nuisance_column_name": "isr_st", "up_weight_column_name": "GenWeights_isr2fsr1", "down_weight_column_name": "GenWeights_isr0p5fsr1", "wt_name": "wt", "mask": "(top_process == 1)"},
        "inputs": ["GenWeights_isr2fsr1", "GenWeights_isr0p5fsr1", "top_process", "isr_st"],
      },
    }
    other_yield_weight_shifts = copy.deepcopy(other_weight_shifts)

    # Add to nuisances
    common_classifier_nuisances = list(common_weight_shifts.keys())

    ttbar_classifier_nuisances += common_classifier_nuisances + list(ttbar_weight_shifts.keys())
    other_classifier_nuisances += common_classifier_nuisances + list(other_weight_shifts.keys())
    nuisances += common_classifier_nuisances + list(ttbar_weight_shifts.keys()) + list(other_weight_shifts.keys())

    # Add Breit-Wigner reweighting weight shifts
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
      "args": {"spline_locations":f"/vols/cms/gu18/innfer_v1/data/{run_name}/top_bw_fractions/top_bw_fraction_locations.yaml", "mass_to":"bw_mass", "mass_from":"sim_mass", "category":cat},
      "inputs": ["bw_mass", "sim_mass"],
    }

    # Set common weight shifts to the rest
    for k, v in common_weight_shifts.items():
      ttbar_weight_shifts[k] = v
      ttbar_yield_weight_shifts[k] = v
      other_weight_shifts[k] = v
      other_yield_weight_shifts[k] = v


    # Define preselection for ttbar (and ttbar_yield)
    ttbar_pre_selection = f"(({pre_selection}) & (GenTop1_mass>150) & (GenTop1_mass<190) & (GenTop2_mass>150) & (GenTop2_mass<190))"

    # Make base files
    base_files[f"base_ttbar_{cat}"] = {
      "inputs": list(ttbar_files[cat].keys()),
      "add_columns": {k: [v[k] for v in ttbar_files[cat].values()] for k in list(ttbar_files[cat].values())[0].keys()},
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
        "n_copies": 50,
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
    models["ttbar"]["classifier_models"] += [{"parameter":k, "file":f"base_ttbar_{cat}", "shifts":{k: {"type": "flat_top", "range": [-3.0,3.0], "other": {"sigma_out": 0.6}}}, "n_copies":3, "categories": categories_per_era[cat]} for k in ttbar_classifier_nuisances]
    models["other"]["classifier_models"] += [{"parameter":k, "file":f"base_other_{cat}", "shifts":{k: {"type": "flat_top", "range": [-3.0,3.0], "other": {"sigma_out": 0.6}}}, "n_copies":3, "categories": categories_per_era[cat]} for k in other_classifier_nuisances]

    # Make yields
    models["ttbar"]["yields"] += [{"file": f"base_ttbar_yield_{cat}", "categories": categories_per_era[cat]}]
    models["other"]["yields"] += [{"file": f"base_other_yield_{cat}", "categories": categories_per_era[cat]}]

  # Define validation loop for POIs
  bw_mass_val_points = [171.5, 172.0, 172.5, 173.0, 173.5]
  validation_loop = [{"bw_mass" : m} for m in bw_mass_val_points] + [{"bw_mass" : 172.5, "fsr": np.log(0.37)/np.log(2.0)}]
  validation = {
    "loop": validation_loop,
    "files": {
      "ttbar": [{"file": f"base_ttbar_{era}", "categories": categories_per_era[era]} for era in years_per_era.keys()],
      "other": [{"file": f"base_other_{era}", "categories": categories_per_era[era]} for era in years_per_era.keys()]
    }
  }

  base_data_files = []
  base_data_add_columns = {}
  for era in years_per_era.keys():
    base_data_files += list(data_files[era].keys())
    for k in data_files[era][list(data_files[era].keys())[0]].keys():
      if k not in base_data_add_columns:
        base_data_add_columns[k] = []
      base_data_add_columns[k] += [v[k] for v in data_files[era].values()]

  # Ensure no nuisance is repeated
  nuisances = list(set(nuisances))

  # Define config
  config = {
    "variables": variables,
    "pois": pois,
    "nuisances": nuisances,
    "categories": categories,
    "data_file": base_data_files,
    "data_add_columns": base_data_add_columns,
    "data_selection": f"(({pre_selection}) & ({post_selection}))",
    "data_calculate" : {"btm_calculate": common_calculate["btm_calculate"], "btm_merged_classifier": common_calculate["btm_merged_classifier"]},
    "inference": {
      "nuisance_constraints": [nui for nui in nuisances if nui not in ["fsr"]],
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
