# python3 scripts/make_parquet_from_root_tree.py --file-location="/vols/cms/dw515/outputs/MSSM/mssm_2018/" --file-names="SUSYGluGluToHToTauTau_M-300_powheg_tt_2018.root,SUSYGluGluToHToTauTau_M-400_powheg_tt_2018.root" --extra-variable-for-files="mass:300,400;other:1,2" --variables="m_vis,pt_1,pt_2" --weight-scale="1,2" --shifted-weights="tau_id:1.0|wt_tau_id_dm0_up,-1.0|wt_tau_id_dm0_down;other_id:1.0|wt_tau_id_dm0_up,-1.0|wt_tau_id_dm0_down"

import argparse
import uproot
import pandas as pd
import copy
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--output-name', help= 'Output dataset name',  default="./output.parquet")
parser.add_argument('--file-location', help= 'Location of files, not including the file name',  default="./")
parser.add_argument('--file-names', help= 'Comma separated list of file names',  default="")
parser.add_argument('--file-extension', help= 'Name at end of file',  default="")
parser.add_argument('--extra-variable-for-files', help= 'Comma separated list of colon  separated lists of the extra variable name and its value. If not empty must be the same number of commas as file-names',  default="")
parser.add_argument('--tree-name', help= 'Name of root tree',  default="ntuple")
parser.add_argument('--variables', help= 'Comma separated list of variables to store',  default="")
parser.add_argument('--weight-name', help= 'Name of weight variable',  default="wt")
parser.add_argument('--weight-scale', help= 'Scale for each weight. Can be empty for no scale, no commas for a single scale, or same number of commas as file-name to be different for each file',  default="")
parser.add_argument('--shifted-weights', help= 'Comma separated list of colon separated lists of the shift names, the shift values and the weights.',  default="")
parser.add_argument('--shifted-trees', help= 'Comma separated list of colon separated lists of the shift names, the shift values and the file names.',  default="")
args = parser.parse_args()

def CustomSplit(input, option):
  if input != "":
    return input.split(option)
  else:
    return []

def ConvertWeight(input):
  if "*" in input:
    factor, weight = input.split("*")
    factor = float(factor)
  else:
    factor = 1.0
    weight = input
  return factor, weight

file_names = CustomSplit(args.file_names,",")
variables = CustomSplit(args.variables,",")

shifted_weights = [[j[0],[CustomSplit(k,"|") for k in CustomSplit(j[1],",")]] for j in [CustomSplit(i,":") for i in CustomSplit(args.shifted_weights,";")]]
shifted_trees = [[j[0],[CustomSplit(k,"|") for k in CustomSplit(j[1],",")]] for j in [CustomSplit(i,":") for i in CustomSplit(args.shifted_trees,";")]]
extra_variable_for_files = [[CustomSplit(i,":")[0],CustomSplit(CustomSplit(i,":")[1],",")] for i in CustomSplit(args.extra_variable_for_files,";")]
weight_scale = CustomSplit(args.weight_scale,",")

for extra_variable in extra_variable_for_files:
  if len(extra_variable[1]) != len(file_names):
    raise ValueError("number of files for extra variables must have length equivalent to file names.")

if len(weight_scale) == 0:
  weight_scale = [1.0]*len(file_names)
elif len(weight_scale) == 1:
  weight_scale = weight_scale*len(file_names)
elif len(weight_scale) != len(file_names):
  raise ValueError("weight-scale must have zero length, one length or length equivalent to file-names.")

print("---------------------------------------------------")
print("file_names:", file_names)
print("variables:", variables)
print("shifted_weights", shifted_weights)
print("shifted_trees", shifted_trees)
print("extra_variable_for_files:", extra_variable_for_files)
print("weight_scale:", weight_scale)
print("---------------------------------------------------")

extra_weights = []
extra_columns = []
for k in shifted_weights:
  extra_columns.append(k[0])
  for shift in k[1]:
    weight = ConvertWeight(shift[1])[1]
    if weight not in extra_weights:
      extra_weights.append(weight)

for k in extra_variable_for_files:
  extra_columns.append(k[0])

leaves_to_get = variables + [args.weight_name] + extra_weights
leaves_to_finish = variables + ["wt"] + extra_columns

first_loop = True
for ind, file_name in enumerate(file_names):
  print(file_name)
  root_file = uproot.open(f"{args.file_location}/{file_name}{args.file_extension}")
  if args.tree_name not in root_file: continue
  tree = root_file[args.tree_name]
  print(tree)
  df = tree.arrays(library="pd")
  df = df.loc[:,leaves_to_get]
  df = df.rename(columns={'wt': args.weight_name})
  df.loc[:,"wt"] *= float(weight_scale[ind])
  for extra_variable in extra_variable_for_files: df.loc[:,extra_variable[0]] = float(extra_variable[1][ind])
  for shifted_weight in shifted_weights: df.loc[:,shifted_weight[0]] = 0.0
  for shifted_tree in shifted_trees: df.loc[:,shifted_tree[0]] = 0.0

  if first_loop:
    total_df = copy.deepcopy(df)
    first_loop = False
  else:
    total_df = pd.concat([total_df, df], ignore_index=True)

  for shifted_weight in shifted_weights:
    for shift in shifted_weight[1]:
      shifted_df = copy.deepcopy(df)
      shifted_df.loc[:,shifted_weight[0]] = float(shift[0])
      factor, weight = ConvertWeight(shift[1])
      shifted_df.loc[:,"wt"] *= factor * shifted_df.loc[:,weight]
      total_df = pd.concat([total_df, shifted_df], ignore_index=True)

  for shifted_tree in shifted_trees:
    for shift in shifted_tree[1]:
      root_file = uproot.open(f"{args.file_location}/{shift[1]}{args.file_extension}")
      if args.tree_name not in root_file: continue
      tree = root_file[args.tree_name]
      shifted_df = tree.arrays(library="pd")
      shifted_df = df.loc[:,leaves_to_get]
      shifted_df = df.rename(columns={'wt': args.weight_name})
      shifted_df.loc[:,"wt"] *= float(weight_scale[ind])
      for extra_variable in extra_variable_for_files: shifted_df.loc[:,extra_variable[0]] = float(extra_variable[1][ind])
      for shifted_weight in shifted_weights: shifted_df.loc[:,shifted_weight[0]] = 0.0
      for shifted_tree in shifted_trees: shifted_df.loc[:,shifted_tree[0]] = 0.0      
      shifted_df.loc[:,shifted_tree[0]] = float(shift[0])      
      total_df = pd.concat([total_df, shifted_df], ignore_index=True)

total_df = total_df.loc[:,leaves_to_finish]
total_df = total_df[total_df.loc[:,"wt"] > 0] # tmp
total_df = total_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(total_df)
print(f"Sum of weights: {total_df.loc[:,'wt'].sum()}")

table = pa.Table.from_pandas(total_df)
pq.write_table(table, args.output_name)