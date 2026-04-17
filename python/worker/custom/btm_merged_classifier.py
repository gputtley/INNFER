
import yaml
import numpy as np

import xgboost as xgb

from useful_functions import GetDictionaryEntryFromYaml

def btm_merged_classifier(
    df, 
    model_path="data/merged_ttbar_bdt/ttbar_bdt_model.json",
    features_path="data/merged_ttbar_bdt/ttbar_bdt_features.yaml",
    working_point_path="data/merged_ttbar_bdt/ttbar_bdt_working_points.yaml",
    working_point = ["signal_efficiency",0.7,"bdt_threshold"]
  ):

  if len(df) == 0:
    return df
  
  # Load model and features
  model = xgb.XGBClassifier()
  model.load_model(model_path)

  # Load features  
  with open(features_path, "r") as f:
    features = yaml.safe_load(f)["features"]

  # Evaluate model
  X = df[features]
  df["btm_merged_classifier"] = model.predict_proba(X)[:, 1]

  # Load working points
  wp = GetDictionaryEntryFromYaml(working_point_path, working_point)

  # Get classification based on working point
  df["btm_merged_classifier_pass"] = (df["btm_merged_classifier"] >= wp).astype(int)

  return df