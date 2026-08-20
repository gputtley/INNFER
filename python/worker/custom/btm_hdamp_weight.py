import os

import numpy as np
import onnxruntime as ort

class btm_hdamp_weight:

  def __init__(self):

    self.up_file_loc_13p6 = 'data/other/mymodel12_hdamp_up_13.6TeV.onnx'
    self.down_file_loc_13p6 = 'data/other/mymodel12_hdamp_down_13.6TeV.onnx'
    self.up_file_loc_13 = 'data/other/mymodel12_hdamp_up_13TeV.onnx'
    self.down_file_loc_13 = 'data/other/mymodel12_hdamp_down_13TeV.onnx'

    if not os.path.exists(self.up_file_loc_13p6):
      #raise FileNotFoundError(f"File not found: {self.up_file_loc_13p6}")
      os.system(f"curl -fL -o {self.up_file_loc_13p6} https://twiki.cern.ch/twiki/pub/CMS/MLReweighting/mymodel12_hdamp_up_13.6TeV.onnx")
    if not os.path.exists(self.down_file_loc_13p6):
      #raise FileNotFoundError(f"File not found: {self.down_file_loc_13p6}")
      os.system(f"curl -fL -o {self.down_file_loc_13p6} https://twiki.cern.ch/twiki/pub/CMS/MLReweighting/mymodel12_hdamp_down_13.6TeV.onnx")
    if not os.path.exists(self.up_file_loc_13):
      #raise FileNotFoundError(f"File not found: {self.up_file_loc_13}")
      os.system(f"curl -fL -o {self.up_file_loc_13} https://twiki.cern.ch/twiki/pub/CMS/MLReweighting/mymodel12_hdamp_up_13TeV.onnx")
    if not os.path.exists(self.down_file_loc_13):
      #raise FileNotFoundError(f"File not found: {self.down_file_loc_13}")
      os.system(f"curl -fL -o {self.down_file_loc_13} https://twiki.cern.ch/twiki/pub/CMS/MLReweighting/mymodel12_hdamp_down_13TeV.onnx")

    self.ort_sess_up = None
    self.ort_sess_down = None

  def __call__(self, df, run=2):

    if self.ort_sess_up is None or self.ort_sess_down is None:
      if run == 3:
        self.ort_sess_up = ort.InferenceSession(self.up_file_loc_13p6)
        self.ort_sess_down = ort.InferenceSession(self.down_file_loc_13p6)
      elif run == 2:
        self.ort_sess_up = ort.InferenceSession(self.up_file_loc_13)
        self.ort_sess_down = ort.InferenceSession(self.down_file_loc_13)
      else:
        raise ValueError(f"Unknown run value: {run}")
       
    input_name = self.ort_sess_up.get_inputs()[0].name
    output_name = self.ort_sess_up.get_outputs()[0].name

    # Expected values
    assert input_name == "input"
    assert output_name == "activation_6"

    hdamp_nominal = 1.379
    max_mass = 243.95

    # Top-quark variables
    top_index = 1
    antitop_index = 2

    pt_top = df[f"GenTop{top_index}_pt"].to_numpy(dtype=np.float64)
    eta_top = df[f"GenTop{top_index}_eta"].to_numpy(dtype=np.float64)
    phi_top = df[f"GenTop{top_index}_phi"].to_numpy(dtype=np.float64)
    mass_top = df[f"GenTop{top_index}_mass"].to_numpy(dtype=np.float64)

    # Antitop-quark variables
    pt_antitop = df[f"GenTop{antitop_index}_pt"].to_numpy(dtype=np.float64)
    eta_antitop = df[f"GenTop{antitop_index}_eta"].to_numpy(dtype=np.float64)
    phi_antitop = df[f"GenTop{antitop_index}_phi"].to_numpy(dtype=np.float64)
    mass_antitop = df[f"GenTop{antitop_index}_mass"].to_numpy(dtype=np.float64)

    # Convert pseudorapidity to rapidity
    mt_top = np.sqrt(mass_top**2 + pt_top**2)
    mt_antitop = np.sqrt(mass_antitop**2 + pt_antitop**2)

    y_top = np.arcsinh(pt_top * np.sinh(eta_top) / mt_top)
    y_antitop = np.arcsinh(
        pt_antitop * np.sinh(eta_antitop) / mt_antitop
    )

    # Normalised PDG IDs:
    #   -6 -> 0.0
    #   +6 -> 0.2
    pid_top = (6.0 / 60.0) + 0.1
    pid_antitop = (-6.0 / 60.0) + 0.1


    n_events = len(df)

    top_features = np.column_stack([
        np.log10(pt_top),
        y_top,
        phi_top,
        mass_top / max_mass,
        np.full(n_events, pid_top),
        np.full(n_events, hdamp_nominal),
    ])

    antitop_features = np.column_stack([
        np.log10(pt_antitop),
        y_antitop,
        phi_antitop,
        mass_antitop / max_mass,
        np.full(n_events, pid_antitop),
        np.full(n_events, hdamp_nominal),
    ])

    # Shape: (number of events, 2 particles, 6 features)
    model_input = np.stack(
        [top_features, antitop_features],
        axis=1,
    ).astype(np.float32)

    assert model_input.shape == (n_events, 2, 6)
    assert np.all(np.isfinite(model_input))

    # Run the network. Most versions have a dynamic batch dimension.
    input_shape = self.ort_sess_up.get_inputs()[0].shape

    if input_shape[0] in (None, "None", -1) or not isinstance(input_shape[0], int):
        prediction_up = self.ort_sess_up.run(
            [output_name],
            {input_name: model_input},
        )[0]
        prediction_down = self.ort_sess_down.run(
            [output_name],
            {input_name: model_input},
        )[0]
    else:
        # Use one event at a time if the model has a fixed (1, 2, 6) input.
        prediction_up = np.concatenate([
            self.ort_sess_up.run(
                [output_name],
                {input_name: model_input[i:i + 1]},
            )[0]
            for i in range(n_events)
        ])
        prediction_down = np.concatenate([
            self.ort_sess_down.run(
                [output_name],
                {input_name: model_input[i:i + 1]},
            )[0]
            for i in range(n_events)
        ])

    prediction_up = np.asarray(prediction_up)
    prediction_down = np.asarray(prediction_down)
    prob_nominal_up = np.clip(prediction_up[:, 0], 1e-7, None)
    prob_target_up = prediction_up[:, 1]
    prob_nominal_down = np.clip(prediction_down[:, 0], 1e-7, None)
    prob_target_down = prediction_down[:, 1]

    weights_up = prob_target_up / prob_nominal_up
    weights_down = prob_target_down / prob_nominal_down

    # Calculate pT of the ttbar system
    px_ttbar = (
        pt_top * np.cos(phi_top)
        + pt_antitop * np.cos(phi_antitop)
    )
    py_ttbar = (
        pt_top * np.sin(phi_top)
        + pt_antitop * np.sin(phi_antitop)
    )
    pt_ttbar = np.hypot(px_ttbar, py_ttbar)

    # The recommended prescription above 1 TeV
    weights_up[pt_ttbar > 1000.0] = 1.0
    weights_down[pt_ttbar > 1000.0] = 1.0

    df["hdamp_weight_up"] = weights_up
    df["hdamp_weight_down"] = weights_down

    return df