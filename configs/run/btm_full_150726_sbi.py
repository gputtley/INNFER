from btm_common_cfg_inc_24 import make_common_config

run_name = "BTM_Full_150726"
categories = {
  "run2": f"((year_ind>=0) & (year_ind<=3))",
  "2223": f"((year_ind>=4) & (year_ind<=7))",
  "24" : f"(year_ind==8)",
}
categories_per_era = {
  "run2": ["run2"],
  "2223": ["2223"],
  "24": ["24"]
}

config = make_common_config(run_name, categories, categories_per_era)
config["name"] = run_name
config["preprocess"]["no_nuisance_yield_effect"] = ["fsr"]
