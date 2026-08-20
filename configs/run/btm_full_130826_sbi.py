from btm_common_cfg_inc_24_fsr_decorrelation import make_common_config

run_name = "BTM_Full_130826"
categories = {
  #"run2": f"((year_ind>=0) & (year_ind<=3))",
  "2223": f"((year_ind>=4) & (year_ind<=7))",
  "24" : f"(year_ind==8)",
}
categories_per_era = {
  #"run2": ["run2"],
  "2223": ["2223"],
  "24": ["24"]
}

config = make_common_config(run_name, categories, categories_per_era)
config["name"] = run_name
config["variables"] = {
  "run2" : [
    "CombinedSubJets_mass",
    "CombinedSubJets_pt",
    "SubJet1_mass",
    "SubJet1_pt",
    "SubJet1_tau21",
    "FatJet_tau21",
    "SubJet2_btagDeepB",
  ],
  "2223" : [
    "CombinedSubJets_mass",
    "CombinedSubJets_pt",
    "SubJet1_mass",
    "SubJet1_pt",
    "SubJet1_tau21",
    "FatJet_tau21",
    "SubJet2_btagDeepB",
  ],
  "24" : [
    "CombinedSubJets_mass",
    "CombinedSubJets_pt",
    "SubJet1_mass",
    "SubJet1_pt",
    "SubJet1_tau21",
    "FatJet_tau21",
  ]
}