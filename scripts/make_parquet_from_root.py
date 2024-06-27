import uproot
import pandas as pd
import numpy as np
import re
import copy
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import subprocess
import ast
import yaml
import warnings
import os
import gc

warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cfg', help= 'Config for running.',  default=None)
parser.add_argument('--remove-negative-weights', help= 'Remove entries with negative weights.',  action='store_true')
parser.add_argument('--remove-nans', help= 'Remove nan entries.',  action='store_true')
args = parser.parse_args()

if args.cfg is None:
  raise ValueError("The --cfg is required.")

if os.path.exists(f"configs/root_to_parquet/{args.cfg}"):
  args.cfg = f"configs/root_to_parquet/{args.cfg}"

if ".py" in args.cfg:
  input = ast.literal_eval(subprocess.getoutput(f"echo $(python3 {args.cfg})"))
elif ".yaml" in args.cfg:
  with open(args.cfg, 'r') as yaml_file:
    input = yaml.load(yaml_file, Loader=yaml.FullLoader)
else:
  raise ValueError("The --cfg must be a .py or a .yaml file.")

for output_file, input_files in input.items():

  print(f"- Making {output_file}")

  first_loop = True
  for input_file, parameters in input_files.items():

    print(f"  - Processing {input_file}")

    file = uproot.open(input_file)
    tree = file[parameters["Tree_Name"]]

    branch_for_weight = re.findall(r'\b\w+\b', parameters["Weight"])
    branch_for_weight_shifts = []
    final_branch_for_weight_shifts = []
    if "Weight_Shifts" in parameters.keys():
      for col_name, extra_weights in parameters["Weight_Shifts"].items():
        final_branch_for_weight_shifts.append(col_name)
        for col_val, extra_weight in extra_weights.items():
          branch_for_weight_shifts += list(re.findall(r'\b\w+\b', extra_weight))
      branch_for_weight_shifts = [string for string in branch_for_weight_shifts if not string.isdigit()]

    get_branches = list(set(parameters["Columns"] + branch_for_weight + branch_for_weight_shifts))
    #final_branches = parameters["Columns"] + ["wt"] + final_branch_for_weight_shifts

    arrays = tree.arrays(get_branches, cut=parameters["Selection"])

    df = pd.DataFrame(np.array(arrays))

    if len(df) == 0:
      continue

    df.eval("wt = " + parameters["Weight"], inplace=True)

    #df = df.loc[:,final_branches]

    if "Extra_Columns" in parameters.keys():
      for k, v in parameters["Extra_Columns"].items():
        df.loc[:,k] = v

    if "Calculate_Extra_Columns" in parameters.keys():
      for new_col, expression in parameters["Calculate_Extra_Columns"].items():
        df.loc[:,new_col] = df.eval(expression)

    if first_loop:
      total_df = copy.deepcopy(df)
      first_loop = False
    else:
      total_df = pd.concat([total_df, df], ignore_index=True)

  if "Weight_Shifts" in parameters.keys():
    for col_name in parameters["Weight_Shifts"].keys():
      total_df.loc[:,col_name] = 0.0
    weight_shift_first_loop = True
    for col_name, extra_weights in parameters["Weight_Shifts"].items():
      for col_val, extra_weight in extra_weights.items():
        shifted_total_df = copy.deepcopy(total_df)
        shifted_total_df.loc[:,col_name] = col_val
        shifted_total_df.eval(f"wt = wt*({extra_weight})", inplace=True)
        if weight_shift_first_loop:
          weight_shift_total_df = copy.deepcopy(shifted_total_df)
          weight_shift_first_loop = False
        else:
          weight_shift_total_df = pd.concat([weight_shift_total_df, shifted_total_df], ignore_index=True)
        del shifted_total_df
        gc.collect()
    total_df = copy.deepcopy(weight_shift_total_df)
    del weight_shift_first_loop
    gc.collect()


  neg_weight_rows = (total_df.loc[:,"wt"] < 0)
  print(f"Total negative weights: {len(total_df[neg_weight_rows])}/{len(total_df)} = {round(len(total_df[neg_weight_rows])/len(total_df),4)}")
  if args.remove_negative_weights:
    print("Removing negative weights")
    total_df = total_df[~neg_weight_rows]

  nan_rows = total_df[total_df.isna().any(axis=1)]
  print(f"Total nans: {len(nan_rows)}/{len(total_df)} = {round(len(nan_rows)/len(total_df),4)}")
  if args.remove_nans:
    print("Removing nans")
    total_df = total_df.dropna()

  total_df = total_df.sample(frac=1, random_state=42).reset_index(drop=True)
  
  print(total_df)
  print("Columns:")
  print(list(total_df.columns))

  table = pa.Table.from_pandas(total_df)
  pq.write_table(table, output_file)