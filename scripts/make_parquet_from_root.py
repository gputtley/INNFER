import uproot
import pandas as pd
import numpy as np
import re
import copy
import pyarrow as pa
import pyarrow.parquet as pq

defaults = {
  "Tree_Name" : "AnalysisTree",
  #"Selection" : "passed_measurement_rec",
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
}

input = {
  "data/topmass_ttbar_mass_v1.parquet" : {
    "data/top_mc/ttbar_1665_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 166.5}
    },
    "data/top_mc/ttbar_1665_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 166.5}
    },
    "data/top_mc/ttbar_1665_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 166.5}
    },
    "data/top_mc/ttbar_1695_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 169.5}
    },
    "data/top_mc/ttbar_1695_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 169.5}
    },
    "data/top_mc/ttbar_1695_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 169.5}
    },
    "data/top_mc/ttbar_1715_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 171.5}
    },
    "data/top_mc/ttbar_1715_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 171.5}
    },
    "data/top_mc/ttbar_1715_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 171.5}
    },
    "data/top_mc/ttbar_1735_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 173.5}
    },
    "data/top_mc/ttbar_1735_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 173.5}
    },
    "data/top_mc/ttbar_1735_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 173.5}
    },
    "data/top_mc/ttbar_1755_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 175.5}
    },
    "data/top_mc/ttbar_1755_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 175.5}
    },
    "data/top_mc/ttbar_1755_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 175.5}
    },
    "data/top_mc/ttbar_1785_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 178.5}
    },
    "data/top_mc/ttbar_1785_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 178.5}
    },
    "data/top_mc/ttbar_1785_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {"mass" : 178.5}
    },
  },
  "data/topmass_other_mass_v1.parquet" : {
    "data/top_mc/st_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {}
    },
    "data/top_mc/st_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {}
    },
    "data/top_mc/st_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {}
    },
    "data/top_mc/wjets_2016_muon.root" : {
      **defaults,
      "Extra_Columns" : {}
    },
    "data/top_mc/wjets_2017_muon.root" : {
      **defaults,
      "Extra_Columns" : {}
    },
    "data/top_mc/wjets_2018_muon.root" : {
      **defaults,
      "Extra_Columns" : {}
    },
  }
}


for output_file, input_files in input.items():

  print(f"- Making {output_file}")

  for ind, (input_file, parameters) in enumerate(input_files.items()):

    print(f"  - Processing {input_file}")

    file = uproot.open(input_file)
    tree = file[parameters["Tree_Name"]]

    branch_for_weight = re.findall(r'\b\w+\b', parameters["Weight"])

    get_branches = parameters["Columns"] + branch_for_weight
    final_branches = parameters["Columns"] + ["wt"]

    arrays = tree.arrays(get_branches, cut=parameters["Selection"])

    df = pd.DataFrame(np.array(arrays))

    df.eval("wt = " + parameters["Weight"], inplace=True)

    df = df.loc[:,final_branches]

    for k, v in parameters["Extra_Columns"].items():
      df.loc[:,k] = v

    if ind == 0:
      total_df = copy.deepcopy(df)
    else:
      total_df = pd.concat([total_df, df], ignore_index=True)


  total_df = total_df[total_df.loc[:,"wt"] > 0] # tmp
  total_df = total_df.sample(frac=1, random_state=42).reset_index(drop=True)

  total_df.loc[:,"total_E"] = total_df.loc[:,"sub1_E_rec"] + total_df.loc[:,"sub2_E_rec"] + total_df.loc[:,"sub3_E_rec"]
  total_df.loc[:,"total_px"] = total_df.loc[:,"sub1_px_rec"] + total_df.loc[:,"sub2_px_rec"] + total_df.loc[:,"sub3_px_rec"]
  total_df.loc[:,"total_py"] = total_df.loc[:,"sub1_py_rec"] + total_df.loc[:,"sub2_py_rec"] + total_df.loc[:,"sub3_py_rec"]
  total_df.loc[:,"total_pz"] = total_df.loc[:,"sub1_pz_rec"] + total_df.loc[:,"sub2_pz_rec"] + total_df.loc[:,"sub3_pz_rec"]
  total_df.loc[:,"total_mass"] = np.sqrt(total_df.loc[:,"total_E"]**2 - total_df.loc[:,"total_px"]**2 - total_df.loc[:,"total_py"]**2 - total_df.loc[:,"total_pz"]**2)

  print(total_df)
  nan_rows = total_df[total_df.isna().any(axis=1)]
  print("Rows with NaN values:")
  print(nan_rows)
  total_df = total_df.dropna()

  table = pa.Table.from_pandas(total_df)
  pq.write_table(table, output_file)