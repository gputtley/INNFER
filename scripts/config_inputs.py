import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--files', '-f', help='Files to add to inputs, can be parsed with a wildcard', default="*.parquet")
parser.add_argument('--add-column', '-ac', help='Option should be repeated. Given as ColumnName,FindStr,Value', action="append", default=[])
parser.add_argument('--default-column', '-dc', help='Option should be repeated. Given as ColumnName,DefaultVal', action="append", default=[])
args = parser.parse_args()


# Get files
input_files = args.files.split(",")
files = []
for f in input_files:
  if "*" in f:
    files += glob.glob(f)
  else:
    files += [f]
files = sorted(list(set(files)))

# Format columns to add and defaults
defaults = {k:float(v) for k,v in [x.split(",") for x in args.default_column]}
add_columns = []
search_str = []
values = []
for col in args.add_column:
  parts = col.split(",")
  add_columns.append(parts[0])
  search_str.append(parts[1])
  values.append(float(parts[2]))
  if parts[0] not in defaults:
    defaults[parts[0]] = -999.0

# Loop through files
add_columns_output = {} 
for ind, fn in enumerate(files):

  f = fn.split("/")[-1]

  # Set defaults
  for col, default_val in defaults.items():
    if col not in add_columns_output:
      add_columns_output[col] = []
    add_columns_output[col].append(defaults[col])

  # Set values for columns
  for col, find_str, value in zip(add_columns, search_str, values):
    if find_str in f:
      add_columns_output[col][-1] = value

# print output
print('    inputs:')
for f in files:
  print(f'      - "${{data_loc}}/{f.split("/")[-1]}"')
if len(add_columns_output) > 0:
  print('    add_columns:')
  for col, vals in add_columns_output.items():
    print(f'      {col}:')
    for val in vals:
      print(f'        - {val}')