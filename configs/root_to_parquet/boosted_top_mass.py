import copy
import numpy as np

version = "v3_uncorrelated"

nui_scale = 1

defaults = {
  "Tree_Name" : "AnalysisTree",
  "Selection" : None,
  "Weight" : "gen_weight*rec_weight*sub1_factor_jer*sub2_factor_jer*sub3_factor_jer*sub1_factor_jec*sub2_factor_jec*sub3_factor_jec*sub1_factor_cor*sub2_factor_cor*sub3_factor_cor",
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
    "index_b",
    "passed_btagmigration_rec",
    "passed_leptonptmigration_rec",
    "passed_massmigration_rec",
    "passed_measurement_rec",
    "passed_ptmigration_rec",
    "passed_subptmigration_rec",
    "same_leading_order",
  ],
  "Scale" : 1.0,
  "Uncorrelated_Weight_Shifts" : {
    "jec" : {i : f"(((sub1_factor_jec+({i}*sub1_sigma_jec))*(sub2_factor_jec+({i}*sub2_sigma_jec))*(sub3_factor_jec+({i}*sub3_sigma_jec)))/(sub1_factor_jec*sub2_factor_jec*sub3_factor_jec))" for i in [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]},
    "cor" : {i : f"(((sub1_factor_cor+({i}*sub1_sigma_cor))*(sub2_factor_cor+({i}*sub2_sigma_cor))*(sub3_factor_cor+({i}*sub3_sigma_cor)))/(sub1_factor_cor*sub2_factor_cor*sub3_factor_cor))" for i in [-3.0,-2.0,-1.0,1.0,2.0,3.0]}
  },
  #"Correlated_Weight_Shifts" : {
  #  "jec" : {i : f"(((sub1_factor_jec+({i}*sub1_sigma_jec))*(sub2_factor_jec+({i}*sub2_sigma_jec))*(sub3_factor_jec+({i}*sub3_sigma_jec)))/(sub1_factor_jec*sub2_factor_jec*sub3_factor_jec))" for i in [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]},
  #  "cor" : {i : f"(((sub1_factor_cor+({i}*sub1_sigma_cor))*(sub2_factor_cor+({i}*sub2_sigma_cor))*(sub3_factor_cor+({i}*sub3_sigma_cor)))/(sub1_factor_cor*sub2_factor_cor*sub3_factor_cor))" for i in [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]}
  #},
  #"Correlated_Weight_Shifts" : {
  #  f"jec_scaled_by_{nui_scale}" : {i : f"(((sub1_factor_jec+({i}*{nui_scale}*sub1_sigma_jec))*(sub2_factor_jec+({i}*{nui_scale}*sub2_sigma_jec))*(sub3_factor_jec+({i}*{nui_scale}*sub3_sigma_jec)))/(sub1_factor_jec*sub2_factor_jec*sub3_factor_jec))" for i in [-1.0,0.0,1.0]},
  #  f"cor_scaled_by_{nui_scale}" : {i : f"(((sub1_factor_cor+({i}*{nui_scale}*sub1_sigma_cor))*(sub2_factor_cor+({i}*{nui_scale}*sub2_sigma_cor))*(sub3_factor_cor+({i}*{nui_scale}*sub3_sigma_cor)))/(sub1_factor_cor*sub2_factor_cor*sub3_factor_cor))" for i in [-1.0,0.0,1.0]}
  #},
  "Calculate_Extra_Columns" : {
    "sub12_E_rec" : "sub1_E_rec + sub2_E_rec",
    "sub12_px_rec" : "sub1_px_rec + sub2_px_rec",
    "sub12_py_rec" : "sub1_py_rec + sub2_py_rec",
    "sub12_pz_rec" : "sub1_pz_rec + sub2_pz_rec",
    "sub13_E_rec" : "sub1_E_rec + sub3_E_rec",
    "sub13_px_rec" : "sub1_px_rec + sub3_px_rec",
    "sub13_py_rec" : "sub1_py_rec + sub3_py_rec",
    "sub13_pz_rec" : "sub1_pz_rec + sub3_pz_rec",
    "sub23_E_rec" : "sub2_E_rec + sub3_E_rec",
    "sub23_px_rec" : "sub2_px_rec + sub3_px_rec",
    "sub23_py_rec" : "sub2_py_rec + sub3_py_rec",
    "sub23_pz_rec" : "sub2_pz_rec + sub3_pz_rec",
    "sub123_E_rec" : "sub1_E_rec + sub2_E_rec + sub3_E_rec",
    "sub123_px_rec" : "sub1_px_rec + sub2_px_rec + sub3_px_rec",
    "sub123_py_rec" : "sub1_py_rec + sub2_py_rec + sub3_py_rec",
    "sub123_pz_rec" : "sub1_pz_rec + sub2_pz_rec + sub3_pz_rec",
    "b_px_rec" : "((index_b==1)*(sub1_px_rec)) + ((index_b==2)*(sub2_px_rec)) + ((index_b==3)*(sub3_px_rec))",
    "b_py_rec" : "((index_b==1)*(sub1_py_rec)) + ((index_b==2)*(sub2_py_rec)) + ((index_b==3)*(sub3_py_rec))",
    "b_pz_rec" : "((index_b==1)*(sub1_pz_rec)) + ((index_b==2)*(sub2_pz_rec)) + ((index_b==3)*(sub3_pz_rec))",
    "b_E_rec" : "((index_b==1)*(sub1_E_rec)) + ((index_b==2)*(sub2_E_rec)) + ((index_b==3)*(sub3_E_rec))",
    "w1_px_rec" : "((index_b==1)*(sub2_px_rec)) + ((index_b==2)*(sub1_px_rec)) + ((index_b==3)*(sub1_px_rec))",
    "w1_py_rec" : "((index_b==1)*(sub2_py_rec)) + ((index_b==2)*(sub1_py_rec)) + ((index_b==3)*(sub1_py_rec))",
    "w1_pz_rec" : "((index_b==1)*(sub2_pz_rec)) + ((index_b==2)*(sub1_pz_rec)) + ((index_b==3)*(sub1_pz_rec))",
    "w1_E_rec" : "((index_b==1)*(sub2_E_rec)) + ((index_b==2)*(sub1_E_rec)) + ((index_b==3)*(sub1_E_rec))",
    "w2_px_rec" : "((index_b==1)*(sub3_px_rec)) + ((index_b==2)*(sub3_px_rec)) + ((index_b==3)*(sub2_px_rec))",
    "w2_py_rec" : "((index_b==1)*(sub3_py_rec)) + ((index_b==2)*(sub3_py_rec)) + ((index_b==3)*(sub2_py_rec))",
    "w2_pz_rec" : "((index_b==1)*(sub3_pz_rec)) + ((index_b==2)*(sub3_pz_rec)) + ((index_b==3)*(sub2_pz_rec))",
    "w2_E_rec" : "((index_b==1)*(sub3_E_rec)) + ((index_b==2)*(sub3_E_rec)) + ((index_b==3)*(sub2_E_rec))",
    "w_px_rec" : "w1_px_rec + w2_px_rec",
    "w_py_rec" : "w1_py_rec + w2_py_rec",
    "w_pz_rec" : "w1_pz_rec + w2_pz_rec",
    "w_E_rec" : "w1_E_rec + w2_E_rec",
  }
}

def mass(E,px,py,pz): return f"({E}**2 - {px}**2 - {py}**2 - {pz}**2)**0.5"
def eta(E,px,py,pz): return f"-log(sin(arccos({pz} / sqrt({px}**2 + {py}**2 + {pz}**2)) / 2)/cos(arccos({pz} / sqrt({px}**2 + {py}**2 + {pz}**2)) / 2))"
def y(E,px,py,pz): return f"0.5*log(({E} + {pz}) / ({E} - {pz}))"
def pt(E,px,py,pz): return f"({px}**2 + {pz}**2)**0.5"
def phi(E,px,py,pz): return f"arctan2({py}, {px})"

add_extra_var = [
  "sub1",
  "sub2",
  "sub3",
  "sub12",
  "sub13",
  "sub23",
  "sub123",
  "b",
  "w1",
  "w2",
  "w"
]

for var in add_extra_var:
  defaults["Calculate_Extra_Columns"][f"{var}_mass_rec"] = mass(f"{var}_E_rec", f"{var}_px_rec", f"{var}_py_rec", f"{var}_pz_rec")
  defaults["Calculate_Extra_Columns"][f"{var}_eta_rec"] = eta(f"{var}_E_rec", f"{var}_px_rec", f"{var}_py_rec", f"{var}_pz_rec")
  defaults["Calculate_Extra_Columns"][f"{var}_y_rec"] = y(f"{var}_E_rec", f"{var}_px_rec", f"{var}_py_rec", f"{var}_pz_rec")
  defaults["Calculate_Extra_Columns"][f"{var}_pt_rec"] = pt(f"{var}_E_rec", f"{var}_px_rec", f"{var}_py_rec", f"{var}_pz_rec")
  defaults["Calculate_Extra_Columns"][f"{var}_phi_rec"] = phi(f"{var}_E_rec", f"{var}_px_rec", f"{var}_py_rec", f"{var}_pz_rec")

defaults_sel = copy.deepcopy(defaults)
defaults_sel["Selection"] = "((passed_massmigration_rec==0) & (passed_btagmigration_rec==0) & (passed_leptonptmigration_rec==0) & (passed_ptmigration_rec==0) & (passed_subptmigration_rec==0) & (passed_measurement_rec==1))"
defaults["Selection"] = "((passed_massmigration_rec==0) & (passed_btagmigration_rec==0) & (passed_leptonptmigration_rec==0) & (passed_ptmigration_rec==0) & (passed_subptmigration_rec==0))"

ttbar_files = {
  "ttbar_1665" : 166.5,
  "ttbar_1695": 169.5,
  "ttbar_1715": 171.5,
  "ttbar" : 172.5,
  "ttbar_1735" : 173.5,
  "ttbar_1755" : 175.5,
  "ttbar_1785" : 178.5,
}

bkg_files = [
  "st",
  "wjets",
]

years = ["2016","2017","2018"]
#years = ["2016"]
e_or_mu_bools = {"elec":0.0, "muon":1.0}
input_dir = "data/top_mc/"

input = {
  #f"data/topmass_toy_data_{version}.parquet" : {},
  f"data/boostedtopmass_ttbar_mass_{version}.parquet" : {},
  f"data/boostedtopmass_other_mass_{version}.parquet" : {},
}

for yr in years:
  for e_or_mu, e_or_mu_bool in e_or_mu_bools.items():
    
    for f in bkg_files:
      #input[f"data/topmass_toy_data_{version}.parquet"][f"{input_dir}/{f}_{yr}_{e_or_mu}.root"] = {
      #  **defaults,
      #  "Extra_Columns" : {"e_or_mu" : e_or_mu_bool, "era" : yr}          
      #}

      input[f"data/boostedtopmass_other_mass_{version}.parquet"][f"{input_dir}/{f}_{yr}_{e_or_mu}.root"] = {
        **defaults,
        "Extra_Columns" : {"e_or_mu" : e_or_mu_bool, "era" : yr}          
      }
          
    for f, m in ttbar_files.items():
      input[f"data/boostedtopmass_ttbar_mass_{version}.parquet"][f"{input_dir}/{f}_{yr}_{e_or_mu}.root"] = {
        **defaults_sel,
        "Extra_Columns" : {"e_or_mu" : e_or_mu_bool, "era" : yr, "mass" : m}          
      }

    #input[f"data/topmass_toy_data_{version}.parquet"][f"{input_dir}/ttbar_{yr}_{e_or_mu}.root"] = {
    #  **defaults_sel,
    #  "Extra_Columns" : {"e_or_mu" : e_or_mu_bool, "era" : yr}          
    #}
    

print(input)