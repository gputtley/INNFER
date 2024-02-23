defaults = {
  "Tree_Name" : "AnalysisTree",
  "Selection" : None,
  "Weight" : "gen_weight*rec_weight",
  "Columns" : [
    "sub1_E_rec",
    "sub1_px_rec",
    "sub1_py_rec",
    "sub1_pz_rec",
    "sub2_E_rec",
    "sub2_px_rec",
    "sub2_py_rec",
    "sub2_pz_rec",  
    "sub3_E_rec",
    "sub3_px_rec",
    "sub3_py_rec",
    "sub3_pz_rec",        
  ],
  "Scale" : 1.0,
  "Weight_Shifts" : {},
  "Tree_Shifts" : {},
  "Calculate_Extra_Columns" : {
    "sub12_E_rec" : "sub1_E_rec + sub2_E_rec",
    "sub12_px_rec" : "sub1_px_rec + sub2_px_rec",
    "sub12_py_rec" : "sub1_py_rec + sub2_py_rec",
    "sub12_pz_rec" : "sub1_pz_rec + sub2_pz_rec",
    "sub12_mass_rec" : "(sub12_E_rec**2 - sub12_px_rec**2 - sub12_py_rec**2 - sub1_pz_rec**2)**0.5", 
    "sub13_E_rec" : "sub1_E_rec + sub3_E_rec",
    "sub13_px_rec" : "sub1_px_rec + sub3_px_rec",
    "sub13_py_rec" : "sub1_py_rec + sub3_py_rec",
    "sub13_pz_rec" : "sub1_pz_rec + sub3_pz_rec",
    "sub13_mass_rec" : "(sub13_E_rec**2 - sub13_px_rec**2 - sub13_py_rec**2 - sub13_pz_rec**2)**0.5",  
    "sub23_E_rec" : "sub2_E_rec + sub3_E_rec",
    "sub23_px_rec" : "sub2_px_rec + sub3_px_rec",
    "sub23_py_rec" : "sub2_py_rec + sub3_py_rec",
    "sub23_pz_rec" : "sub2_pz_rec + sub3_pz_rec",
    "sub23_mass_rec" : "(sub23_E_rec**2 - sub23_px_rec**2 - sub23_py_rec**2 - sub23_pz_rec**2)**0.5",
    "sub123_E_rec" : "sub1_E_rec + sub2_E_rec + sub3_E_rec",
    "sub123_px_rec" : "sub1_px_rec + sub2_px_rec + sub3_px_rec",
    "sub123_py_rec" : "sub1_py_rec + sub2_py_rec + sub3_py_rec",
    "sub123_pz_rec" : "sub1_pz_rec + sub2_pz_rec + sub3_pz_rec",
    "sub123_mass_rec" : "(sub123_E_rec**2 - sub123_px_rec**2 - sub123_py_rec**2 - sub123_pz_rec**2)**0.5",
    "sub123_pt_rec" : "(sub123_px_rec**2 + sub123_pz_rec**2)**0.5",
    "sub123_eta_rec" : "-log(sin(arccos(sub123_pz_rec / sqrt(sub123_px_rec**2 + sub123_py_rec**2 + sub123_pz_rec**2)) / 2)/cos(arccos(sub123_pz_rec / sqrt(sub123_px_rec**2 + sub123_py_rec**2 + sub123_pz_rec**2)) / 2))",
    "sub123_y_rec" : "0.5*log((sub123_E_rec + sub123_pz_rec) / (sub123_E_rec - sub123_pz_rec))",
    "sub123_phi_rec" : "arctan2(sub123_py_rec, sub123_px_rec)",
    "sub1_mass_rec" : "(sub1_E_rec**2 - sub1_px_rec**2 - sub1_py_rec**2 - sub1_pz_rec**2)**0.5",
    "sub1_pt_rec" : "(sub1_px_rec**2 + sub1_pz_rec**2)**0.5",
    "sub1_eta_rec" : "-log(sin(arccos(sub1_pz_rec / sqrt(sub1_px_rec**2 + sub1_py_rec**2 + sub1_pz_rec**2)) / 2)/cos(arccos(sub1_pz_rec / sqrt(sub1_px_rec**2 + sub1_py_rec**2 + sub1_pz_rec**2)) / 2))",
    "sub1_y_rec" : "0.5*log((sub1_E_rec + sub1_pz_rec) / (sub1_E_rec - sub1_pz_rec))",
    "sub1_phi_rec" : "arctan2(sub1_py_rec, sub1_px_rec)",
    "sub2_mass_rec" : "(sub2_E_rec**2 - sub2_px_rec**2 - sub2_py_rec**2 - sub2_pz_rec**2)**0.5",
    "sub2_pt_rec" : "(sub2_px_rec**2 + sub2_pz_rec**2)**0.5",
    "sub2_eta_rec" : "-log(sin(arccos(sub2_pz_rec / sqrt(sub2_px_rec**2 + sub2_py_rec**2 + sub2_pz_rec**2)) / 2)/cos(arccos(sub2_pz_rec / sqrt(sub2_px_rec**2 + sub2_py_rec**2 + sub2_pz_rec**2)) / 2))",
    "sub2_y_rec" : "0.5*log((sub2_E_rec + sub2_pz_rec) / (sub2_E_rec - sub2_pz_rec))",
    "sub2_phi_rec" : "arctan2(sub2_py_rec, sub2_px_rec)",
    "sub3_mass_rec" : "(sub3_E_rec**2 - sub3_px_rec**2 - sub3_py_rec**2 - sub3_pz_rec**2)**0.5",
    "sub3_pt_rec" : "(sub3_px_rec**2 + sub3_pz_rec**2)**0.5",
    "sub3_eta_rec" : "-log(sin(arccos(sub3_pz_rec / sqrt(sub3_px_rec**2 + sub3_py_rec**2 + sub3_pz_rec**2)) / 2)/cos(arccos(sub3_pz_rec / sqrt(sub3_px_rec**2 + sub3_py_rec**2 + sub3_pz_rec**2)) / 2))",
    "sub3_y_rec" : "0.5*log((sub3_E_rec + sub3_pz_rec) / (sub3_E_rec - sub3_pz_rec))",
    "sub3_phi_rec" : "arctan2(sub3_py_rec, sub3_px_rec)",
  }
}

input = {
  "data/topmass_toy_data_v3.parquet" : {
    "data/top_mc/ttbar_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/ttbar_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/ttbar_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/st_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/st_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/st_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/wjets_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/wjets_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/wjets_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/st_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/st_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/st_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2018}
    },
    "data/top_mc/wjets_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/wjets_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/wjets_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2018}
    },
  },
  "data/topmass_ttbar_mass_v3.parquet" : { 
    "data/top_mc/ttbar_1665_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 166.5, "e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1665_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 166.5, "e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1665_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 166.5, "e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/ttbar_1695_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 169.5, "e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1695_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 169.5, "e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1695_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 169.5, "e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/ttbar_1715_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 171.5, "e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1715_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 171.5, "e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1715_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 171.5, "e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/ttbar_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 172.5, "e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/ttbar_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 172.5, "e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/ttbar_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 172.5, "e_or_mu" : 1.0, "era" : 2018}
    },   
    "data/top_mc/ttbar_1735_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 173.5, "e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1735_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 173.5, "e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1735_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 173.5, "e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/ttbar_1755_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 175.5, "e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1755_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 175.5, "e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1755_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 175.5, "e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/ttbar_1785_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 178.5, "e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1785_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 178.5, "e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1785_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 178.5, "e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/ttbar_1665_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 166.5, "e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1665_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 166.5, "e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1665_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 166.5, "e_or_mu" : 0.0, "era" : 2018}
    },
    "data/top_mc/ttbar_1695_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 169.5, "e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1695_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 169.5, "e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1695_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 169.5, "e_or_mu" : 0.0, "era" : 2018}
    },
    "data/top_mc/ttbar_1715_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 171.5, "e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1715_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 171.5, "e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1715_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 171.5, "e_or_mu" : 0.0, "era" : 2018}
    },
    "data/top_mc/ttbar_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 172.5, "e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/ttbar_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 172.5, "e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/ttbar_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 172.5, "e_or_mu" : 0.0, "era" : 2018}
    },   
    "data/top_mc/ttbar_1735_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 173.5, "e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1735_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 173.5, "e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1735_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 173.5, "e_or_mu" : 0.0, "era" : 2018}
    },
    "data/top_mc/ttbar_1755_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 175.5, "e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1755_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 175.5, "e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1755_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 175.5, "e_or_mu" : 0.0, "era" : 2018}
    },
    "data/top_mc/ttbar_1785_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 178.5, "e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/ttbar_1785_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 178.5, "e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/ttbar_1785_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 178.5, "e_or_mu" : 0.0, "era" : 2018}
    },
  },
  "data/topmass_other_mass_v3.parquet" : {
    "data/top_mc/st_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/st_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/st_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/wjets_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2016}
    },
    "data/top_mc/wjets_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2017}
    },
    "data/top_mc/wjets_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 1.0, "era" : 2018}
    },
    "data/top_mc/st_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/st_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/st_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2018}
    },
    "data/top_mc/wjets_2016_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2016}
    },
    "data/top_mc/wjets_2017_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2017}
    },
    "data/top_mc/wjets_2018_elec.root" : {
      **defaults,
      "Extra_Columns" : {"e_or_mu" : 0.0, "era" : 2018}
    },
  }
}

print(input)