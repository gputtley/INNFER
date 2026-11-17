
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


def btm_merged_classifier_by_era(
    df,
    classifiers,
    working_point = ["signal_efficiency",0.7,"bdt_threshold"]
  ):
  """Apply each era's classifier, feature list and working point to its data rows."""

  if len(df) == 0:
    return df
  if "year_ind" not in df.columns:
    raise ValueError("year_ind is required for era-specific BTM classification")

  masks = {}
  covered = np.zeros(len(df), dtype=bool)
  for era, settings in classifiers.items():
    mask = df["year_ind"].isin(settings["year_inds"]).to_numpy()
    if np.any(covered & mask):
      raise ValueError(f"Overlapping year indices in BTM classifier configuration for {era}")
    masks[era] = mask
    covered |= mask
  if not np.all(covered):
    missing_years = df.loc[~covered,"year_ind"].unique().tolist()
    raise ValueError(f"No BTM classifier configured for year indices {missing_years}")

  scores = np.empty(len(df), dtype=float)
  passed = np.empty(len(df), dtype=int)
  for era, settings in classifiers.items():
    mask = masks[era]
    if not np.any(mask):
      continue
    prediction = btm_merged_classifier(
      df.iloc[np.flatnonzero(mask)].copy(),
      model_path=settings["model_path"],
      features_path=settings["features_path"],
      working_point_path=settings["working_point_path"],
      working_point=working_point,
    )
    scores[mask] = prediction["btm_merged_classifier"].to_numpy()
    passed[mask] = prediction["btm_merged_classifier_pass"].to_numpy()

  df["btm_merged_classifier"] = scores
  df["btm_merged_classifier_pass"] = passed
  return df
