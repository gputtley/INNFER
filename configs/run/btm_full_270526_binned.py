import numpy as np

from btm_common_cfg import make_common_config

run_name = "BTM_Full_270526_Binned"

categories_per_era = {
  "run2": ["run2_signal", "run2_control"],
  "2223": ["2223_signal", "2223_control"],
}
binned_fit_input = {
  "files" : {
    "ttbar": [{"file": f"base_ttbar_{era}", "categories": categories_per_era[era]} for era in categories_per_era.keys()],
    "other": [{"file": f"base_other_{era}", "categories": categories_per_era[era]} for era in categories_per_era.keys()]
  },
  "shape_poi_values" : [170.0, 170.5, 171.0, 171.5, 172.0, 172.5, 173.0, 173.5, 174.0, 174.5, 175.0],
  "variable": "CombinedSubJets_mass",
  "input" : {
    "run2_signal" : {
      "variable" : "CombinedSubJets_mass",
      "binning" : list(np.linspace(100,250,num=30))
    },
    "2223_signal" : {
      "variable" : "CombinedSubJets_mass",
      "binning" : list(np.linspace(100,250,num=30))
    },
    "run2_control" : {
      "variable" : "CombinedSubJets_mass",
      "binning" : list(np.linspace(50,250,num=20))
    },
    "2223_control" : {
      "variable" : "CombinedSubJets_mass",
      "binning" : list(np.linspace(50,250,num=20))
    }
  }
}

categorisation_selection = "(btm_merged_classifier_pass > 0.5)"
categories = {
  "run2_signal": f"((year_ind>=0) & (year_ind<=3) & ({categorisation_selection}))",
  "2223_signal": f"((year_ind>=4) & (year_ind<=7) & ({categorisation_selection}))",
  "run2_control": f"((year_ind>=0) & (year_ind<=3) & ~({categorisation_selection}))",
  "2223_control": f"((year_ind>=4) & (year_ind<=7) & ~({categorisation_selection}))",
}
category_keys = list(categories.keys())
for cat in category_keys:
  categories[cat] = f"(({categories[cat]}) & ({binned_fit_input['input'][cat]['variable']} >= {min(binned_fit_input['input'][cat]['binning'])}) & ({binned_fit_input['input'][cat]['variable']} < {max(binned_fit_input['input'][cat]['binning'])}))"

config = make_common_config(run_name, categories, categories_per_era, add_classifier=True)
config["name"] = run_name
config["inference"]["binned_fit"] = binned_fit_input
