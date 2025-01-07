import copy
import os
import yaml

from functools import partial
import pyarrow as pa
import pyarrow.parquet as pq

from useful_functions import MakeDirectories

class SplitValidationFiles():

  def __init__(self):
    """
    A template class.
    """
    self.parameters = None
    self.data_splits = ["val","test_inf"]

    self.data_output = None
    self.val_loop = None
    self.verbose = True

  def _DoWriteDatasets(self, df, X_columns=[], Y_columns=[], val_ind=0, data_split="val", extra_columns=[]):

    loop = {"X":X_columns, "Y":Y_columns, "wt":["wt"]}
    if len(extra_columns) > 0:
      loop["Extra"] = extra_columns
    for data_type, columns in loop.items():
      file_path = f"{self.data_output}/val_ind_{val_ind}/{data_type}_{data_split}.parquet"
      table = pa.Table.from_pandas(df.loc[:, sorted(columns)], preserve_index=False)
      if os.path.isfile(file_path):
        combined_table = pa.concat_tables([pq.read_table(file_path), table])
        pq.write_table(combined_table, file_path, compression='snappy')
      else:
        pq.write_table(table, file_path, compression='snappy')
    return df

  def Configure(self, options):
    """
    Configure the class settings.

    Args:
        options (dict): Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def Run(self):
    """
    Run the code utilising the worker classes
    """

    # Open parameters
    if self.verbose:
      print(f"- Loading in the parameters")
    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loop through validation files
    for val_ind, val_info in enumerate(self.val_loop):

      # Make split parameters files
      if self.verbose:
        print(f"- Writing parameters yaml for val_ind={val_ind}")
      copied_parameters = copy.deepcopy(parameters)
      copied_parameters["file_loc"] = f"{self.data_output}/val_ind_{val_ind}"
      param_file_name = f"{self.data_output}/val_ind_{val_ind}/parameters.yaml"
      MakeDirectories(param_file_name)
      with open(param_file_name, 'w') as yaml_file:
        yaml.dump(copied_parameters, yaml_file, default_flow_style=False)  
      print(f"Created {param_file_name}")

    for data_split in self.data_splits:

      file_inputs = [f"{parameters['file_loc']}/X_{data_split}.parquet",f"{parameters['file_loc']}/Y_{data_split}.parquet",f"{parameters['file_loc']}/wt_{data_split}.parquet"]
      if "Extra_columns" in parameters.keys():
        if len(parameters["Extra_columns"]) > 0: 
          file_inputs += [f"{parameters['file_loc']}/Extra_{data_split}.parquet"]

      # Build data processor
      from data_processor import DataProcessor
      dp = DataProcessor(
        [file_inputs],
        "parquet",
        options = {
          "parameters" : parameters,
        }
      )

      # Loop through validation files
      for val_ind, val_info in enumerate(self.val_loop):

        # Make split validation dataset files
        if self.verbose:
          print(f"- Making validation dataset for val_ind={val_ind}")
        for data_type in ["X","Y","wt"]:
          file_name = f"{self.data_output}/val_ind_{val_ind}/{data_type}_{data_split}.parquet"
          if os.path.isfile(file_name):
            os.system(f"rm {file_name}")
        shape_Y_cols = [col for col in val_info['row'].columns if col in parameters["Y_columns"]]
        dp.GetFull(
          method = None,
          extra_sel = " & ".join([f"({col}=={val_info['row'].loc[:,col].iloc[0]})" for col in shape_Y_cols]) if len(shape_Y_cols) > 0 else None,
          functions_to_apply = [
            "untransform",
            "transform",
            partial(self._DoWriteDatasets, X_columns=parameters["X_columns"], Y_columns=parameters["Y_columns"], extra_columns=parameters["Extra_columns"] if "Extra_columns" in parameters.keys() else [], val_ind=val_ind, data_split=data_split),
          ]
        )
        print(f"Created {self.data_output}/val_ind_{val_ind}/X_{data_split}.yaml")
        print(f"Created {self.data_output}/val_ind_{val_ind}/Y_{data_split}.yaml")
        print(f"Created {self.data_output}/val_ind_{val_ind}/wt_{data_split}.yaml")
        if "Extra_columns" in parameters.keys():
          if len(parameters["Extra_columns"]) > 0: 
            print(f"Created {self.data_output}/val_ind_{val_ind}/Extra_{data_split}.yaml")

  def Outputs(self):
    """
    Return a list of outputs given by class
    """
    outputs = []
    for val_ind, val_info in enumerate(self.val_loop):
      for data_type in ["X","Y","wt"]:
        file_name = f"{self.data_output}/val_ind_{val_ind}/{data_type}_val.parquet"

    return outputs

  def Inputs(self):
    """
    Return a list of inputs required by class
    """
    inputs = [
      self.parameters,      
    ]

    with open(self.parameters, 'r') as yaml_file:
      parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
    inputs += [f"{parameters['file_loc']}/X_val.parquet",f"{parameters['file_loc']}/Y_val.parquet",f"{parameters['file_loc']}/wt_val.parquet"]

    return inputs
