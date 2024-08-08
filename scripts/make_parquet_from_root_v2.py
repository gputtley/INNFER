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
import itertools

warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', help='The batch size to load in and perform actions', type=int, default=10**5)
parser.add_argument('-c','--cfg', help= 'Config for running.',  default=None)
parser.add_argument('--remove-negative-weights', help= 'Remove entries with negative weights.',  action='store_true')
parser.add_argument('--remove-nans', help= 'Remove nan entries.',  action='store_true')
parser.add_argument('--verbose', help= 'Verbose.',  action='store_true')
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

    # Open root tree
    file = uproot.open(input_file)
    tree = file[parameters["Tree_Name"]]
    
    # Find the leaves I need to process
    branch_for_weight = re.findall(r'\b\w+\b', parameters["Weight"])
    branch_for_weight_shifts = []
    final_branch_for_weight_shifts = []
    if "Uncorrelated_Weight_Shifts" in parameters.keys():
      for col_name, extra_weights in parameters["Uncorrelated_Weight_Shifts"].items():
        final_branch_for_weight_shifts.append(col_name)
        for col_val, extra_weight in extra_weights.items():
          branch_for_weight_shifts += list(re.findall(r'\b\w+\b', extra_weight))
    if "Correlated_Weight_Shifts" in parameters.keys():
      for col_name, extra_weights in parameters["Correlated_Weight_Shifts"].items():
        final_branch_for_weight_shifts.append(col_name)
        for col_val, extra_weight in extra_weights.items():
          branch_for_weight_shifts += list(re.findall(r'\b\w+\b', extra_weight)) 
    branch_for_weight_shifts = [string for string in branch_for_weight_shifts if not string.isdigit()]
    get_branches = list(set(parameters["Columns"] + branch_for_weight + branch_for_weight_shifts))

    # Load in batches into pandas
    total_entries = tree.num_entries
    for start in range(0, total_entries, args.batch_size):
      stop = min(start + args.batch_size, total_entries)
      
      # Read batch to dataframe
      arrays = tree.arrays(
        get_branches,
        cut=parameters["Selection"],
        entry_start=start,
        entry_stop=stop
      )    
      df = pd.DataFrame(np.array(arrays))

      # Skip if dataframe is empty
      if len(df) == 0:
        continue

      # Make nominal weight
      df.eval("wt = " + parameters["Weight"], inplace=True)

      # Add extra columns
      if "Extra_Columns" in parameters.keys():
        for k, v in parameters["Extra_Columns"].items():
          df.loc[:,k] = v
      if "Calculate_Extra_Columns" in parameters.keys():
        for new_col, expression in parameters["Calculate_Extra_Columns"].items():
          df.loc[:,new_col] = df.eval(expression)

      # Perform uncorrelated weight shifts
      if "Uncorrelated_Weight_Shifts" in parameters.keys():
        for col_name in parameters["Uncorrelated_Weight_Shifts"].keys():
          df.loc[:,col_name] = 0.0
        weight_shift_df = copy.deepcopy(df)
        for col_name, extra_weights in parameters["Uncorrelated_Weight_Shifts"].items():
          for col_val, extra_weight in extra_weights.items():
            shifted_df = copy.deepcopy(df)
            shifted_df.loc[:,col_name] = col_val
            shifted_df.eval(f"wt = wt*({extra_weight})", inplace=True)
            weight_shift_df = pd.concat([weight_shift_df, shifted_df], ignore_index=True)
            del shifted_df
            gc.collect()
        df = copy.deepcopy(weight_shift_df)
        gc.collect()

      # Perform correlated weight shifts
      if "Correlated_Weight_Shifts" in parameters.keys():
        combinations = itertools.product(*[list(weight_dict.keys()) for weight_dict in parameters["Correlated_Weight_Shifts"].values()])
        keys = list(parameters["Correlated_Weight_Shifts"].keys())
        first_correlated_loop = True
        for combination in combinations:
          combination_dict = dict(zip(keys, combination))          
          shifted_df = copy.deepcopy(df)
          for weight_name, weight_val in combination_dict.items():
            shifted_df.loc[:,weight_name] = weight_val
            shifted_df.eval(f"wt = wt*({parameters['Correlated_Weight_Shifts'][weight_name][weight_val]})", inplace=True)
          if first_correlated_loop:
            weight_shift_df = copy.deepcopy(shifted_df)
            first_correlated_loop = False
          else:
            weight_shift_df = pd.concat([weight_shift_df, shifted_df], ignore_index=True)
          del shifted_df
          gc.collect()
        df = copy.deepcopy(weight_shift_df)
        
      # Negative weights actions
      neg_weight_rows = (df.loc[:,"wt"] < 0)
      if args.verbose: print(f"Total negative weights: {len(df[neg_weight_rows])}/{len(df)} = {round(len(df[neg_weight_rows])/len(df),4)}")
      if args.remove_negative_weights:
        if args.verbose: print("Removing negative weights")
        df = df[~neg_weight_rows]

      # Nan rows actions
      nan_rows = df[df.isna().any(axis=1)]
      if args.verbose: print(f"Total nans: {len(nan_rows)}/{len(df)} = {round(len(nan_rows)/len(df),4)}")
      if args.remove_nans:
        if args.verbose: print("Removing nans")
        df = df.dropna()

      # Shuffle
      df = df.sample(frac=1, random_state=42).reset_index(drop=True)
      
      # Print dataframe
      if args.verbose: 
        print(df)
        print("Columns:")
        print(list(df.columns))

      # Write to parquet
      table = pa.Table.from_pandas(df, preserve_index=False)
      if first_loop:
        if os.path.isfile(output_file):
          os.system(f"rm {output_file}")
        pq.write_table(table, output_file, compression='snappy')
        first_loop = False
      else:
        combined_table = pa.concat_tables([pq.read_table(output_file), table])
        pq.write_table(combined_table, output_file, compression='snappy')
        # Clear memory
        del combined_table
      
      # Clear memory
      del df, table
      gc.collect()
      
      print(f"    - Events done {stop}/{total_entries}")