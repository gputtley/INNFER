from btm_common_cfg import make_common_config

run_name = "BTM_Full_270526"
categories = {
  "run2": f"((year_ind>=0) & (year_ind<=3))",
  "2223": f"((year_ind>=4) & (year_ind<=7))",
}
categories_per_era = {
  "run2": ["run2"],
  "2223": ["2223"],
}

config = make_common_config(run_name, categories, categories_per_era)
config["name"] = run_name