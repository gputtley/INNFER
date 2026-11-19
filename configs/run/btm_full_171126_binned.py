from btm_common_cfg_inc_24_fsr_decorrelation_smaller_range import make_common_config

run_name = "BTM_Full_171126_Binned"
data_loc = "data/top_reco/270726_parquet_with_psweights"

signal_cat_run2 = {
    'SubJet1_mass_10_60': {
        'selection': '(SubJet1_mass>=10) & (SubJet1_mass<60)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [55.0, 160.0, 180.0, 285.0]
    },
    'SubJet1_mass_60_70': {
        'selection': '(SubJet1_mass>=60) & (SubJet1_mass<70)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [100.0, 155.0, 165.0, 175.0, 180.0, 185.0, 300.0]
    },
    'SubJet1_mass_70_80': {
        'selection': '(SubJet1_mass>=70) & (SubJet1_mass<80)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [100.0, 150.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 205.0, 300.0]
    },
    'SubJet1_mass_80_90': {
        'selection': '(SubJet1_mass>=80) & (SubJet1_mass<90)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [115.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 205.0, 220.0, 300.0]
    },
    'SubJet1_mass_90_100': {
        'selection': '(SubJet1_mass>=90) & (SubJet1_mass<100)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [125.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 215.0, 300.0]
    },
    'SubJet1_mass_100_110': {
        'selection': '(SubJet1_mass>=100) & (SubJet1_mass<110)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [125.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 205.0, 300.0]
    },
    'SubJet1_mass_110_120': {
        'selection': '(SubJet1_mass>=110) & (SubJet1_mass<120)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [145.0, 165.0, 175.0, 180.0, 185.0, 195.0, 205.0, 300.0]
    },
    'SubJet1_mass_120_130': {
        'selection': '(SubJet1_mass>=120) & (SubJet1_mass<130)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [145.0, 180.0, 190.0, 200.0, 295.0]
    },
    'SubJet1_mass_130_180': {
        'selection': '(SubJet1_mass>=130) & (SubJet1_mass<180)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [155.0, 190.0, 205.0, 220.0, 300.0]
    }
}

signal_cat_2223 = {
    'SubJet1_mass_10_70': {
        'selection': '(SubJet1_mass>=10) & (SubJet1_mass<70)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [80.0, 160.0, 170.0, 180.0, 190.0, 300.0]
    },
    'SubJet1_mass_70_80': {
        'selection': '(SubJet1_mass>=70) & (SubJet1_mass<80)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [105.0, 150.0, 165.0, 170.0, 175.0, 180.0, 190.0, 285.0]
    },
    'SubJet1_mass_80_90': {
        'selection': '(SubJet1_mass>=80) & (SubJet1_mass<90)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [105.0, 150.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 200.0, 290.0]
    },
    'SubJet1_mass_90_100': {
        'selection': '(SubJet1_mass>=90) & (SubJet1_mass<100)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [125.0, 160.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 295.0]
    },
    'SubJet1_mass_100_110': {
        'selection': '(SubJet1_mass>=100) & (SubJet1_mass<110)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [135.0, 170.0, 180.0, 190.0, 200.0, 300.0]
    },
    'SubJet1_mass_110_120': {
        'selection': '(SubJet1_mass>=110) & (SubJet1_mass<120)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [145.0, 180.0, 200.0, 295.0]
    },
    'SubJet1_mass_120_180': {
        'selection': '(SubJet1_mass>=120) & (SubJet1_mass<180)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [150.0, 185.0, 195.0, 215.0, 300.0]
    }
}

signal_cat_24 = {
    'SubJet1_mass_0_60': {
        'selection': '(SubJet1_mass>=0) & (SubJet1_mass<60)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [60.0, 165.0, 185.0, 265.0]
    },
    'SubJet1_mass_60_70': {
        'selection': '(SubJet1_mass>=60) & (SubJet1_mass<70)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [100.0, 165.0, 175.0, 185.0, 270.0]
    },
    'SubJet1_mass_70_80': {
        'selection': '(SubJet1_mass>=70) & (SubJet1_mass<80)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [105.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 195.0, 300.0]
    },
    'SubJet1_mass_80_90': {
        'selection': '(SubJet1_mass>=80) & (SubJet1_mass<90)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [110.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 205.0, 300.0]
    },
    'SubJet1_mass_90_100': {
        'selection': '(SubJet1_mass>=90) & (SubJet1_mass<100)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [125.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 290.0]
    },
    'SubJet1_mass_100_110': {
        'selection': '(SubJet1_mass>=100) & (SubJet1_mass<110)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [135.0, 170.0, 180.0, 185.0, 190.0, 195.0, 205.0, 295.0]
    },
    'SubJet1_mass_110_120': {
        'selection': '(SubJet1_mass>=110) & (SubJet1_mass<120)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [145.0, 170.0, 185.0, 195.0, 205.0, 285.0]
    },
    'SubJet1_mass_120_130': {
        'selection': '(SubJet1_mass>=120) & (SubJet1_mass<130)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [150.0, 185.0, 200.0, 290.0]
    },
    'SubJet1_mass_130_180': {
        'selection': '(SubJet1_mass>=130) & (SubJet1_mass<180)',
        'discriminator': 'CombinedSubJets_mass',
        'bins': [165.0, 190.0, 205.0, 300.0]
    }
}

background_bins_run2 = [50.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0, 225.0, 245.0, 265.0, 300.0]

background_bins_2223 = [50.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 215.0, 245.0, 300.0]

background_bins_24 = [50.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 235.0, 245.0, 265.0, 300.0]



category_definitions = {}

for key, val in {"run2": signal_cat_run2, "2223": signal_cat_2223, "24": signal_cat_24}.items():
    category_definitions[key] = {}
    for cat_key, cat_val in val.items():
        signal_cat_key = f"{cat_key}_Signal"
        category_definitions[key][signal_cat_key] = cat_val
        category_definitions[key][signal_cat_key]['selection'] += ' & (btm_merged_classifier_pass > 0.5)'
    category_definitions[key][f"{cat_key}_Background"] = {
        'selection': 'btm_merged_classifier_pass <= 0.5',
        'discriminator': 'CombinedSubJets_mass',
        'bins': background_bins_run2 if key == "run2" else background_bins_2223 if key == "2223" else background_bins_24
    }



era_definitions = {
  "run2": f"((year_ind>=0) & (year_ind<=3))",
  "2223": f"((year_ind>=4) & (year_ind<=7))",
  "24" : f"(year_ind==8)",
}

categories = {}
categories_per_era = {}
binned_fit_inputs = {}
for era, era_v in category_definitions.items():
  for cat, cat_v in era_v.items():
    total_cat = f"{era}_{cat}"
    categories[total_cat] = f"(({era_definitions[era]}) & ({cat_v['selection']}))"
    if era not in categories_per_era.keys():
      categories_per_era[era] = []
    categories_per_era[era].append(total_cat)
    binned_fit_inputs[total_cat] = {
      "variable" : cat_v['discriminator'],
      "binning" : cat_v['bins']
    }


binned_fit_input = {
  "files" : {
    "ttbar": [{"file": f"base_ttbar_{era}", "categories": categories_per_era[era]} for era in categories_per_era.keys()],
    "other": [{"file": f"base_other_{era}", "categories": categories_per_era[era]} for era in categories_per_era.keys()]
  },
  "shape_poi_values" : [170.0, 170.5, 171.0, 171.5, 172.0, 172.5, 173.0, 173.5, 174.0, 174.5, 175.0],
  "input" : binned_fit_inputs
}

category_keys = list(categories.keys())
for cat in category_keys:
  categories[cat] = f"(({categories[cat]}) & ({binned_fit_input['input'][cat]['variable']} >= {min(binned_fit_input['input'][cat]['binning'])}) & ({binned_fit_input['input'][cat]['variable']} < {max(binned_fit_input['input'][cat]['binning'])}))"
config = make_common_config(run_name, categories, categories_per_era, add_classifier=True, data_loc=data_loc)
config["name"] = run_name
config["inference"]["binned_fit"] = binned_fit_input
