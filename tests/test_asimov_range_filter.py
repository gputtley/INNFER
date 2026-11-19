"""Range cuts must be opt-in and preserve requested Asimov count and yield."""
import ast
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('EVENTS_PER_BATCH', '10000')

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python/worker'), str(ROOT/'python/runner')]
from make_asimov import MakeAsimov
from useful_functions import InitiateDensityModel


class AsimovRangeFilterTests(unittest.TestCase):
  def test_real_flow_tails_are_retained_by_default_and_rejected_when_requested(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      params = {'X_columns': ['x'], 'Y_columns': [],
        'standardisation': {'x': {'mean': 5., 'std': 2.}},
        'minmax': {'x': {'min': -1., 'max': 1.}},
        'initial_minmax': {'x': {'min': 4., 'max': 6.}}}
      for split in ['train', 'test']:
        for label, frame in [('X', pd.DataFrame({'x': [0., .1]})),
                             ('Y', pd.DataFrame(index=range(2))),
                             ('wt', pd.DataFrame({'wt': [1., 1.]}))]:
          frame.to_parquet(root/f'{label}_{split}.parquet', index=False)
      architecture = {'type': 'KerasFlow', 'num_coupling_layers': 2, 'hidden_units': [8]}
      model = root/'models'/'example_density'
      model.mkdir(parents=True)
      network = InitiateDensityModel(architecture, str(root), options={'data_parameters': params})
      network.BuildModel()
      network.Save(str(model/'example.h5'))
      (model/'example_architecture.yaml').write_text(yaml.safe_dump(architecture))
      metadata = root/'parameters.yaml'
      metadata.write_text(yaml.safe_dump({'file_name': 'example', 'density': params,
                                          'yields': {'nominal': 100., 'lnN': {}}}))
      outputs = {}
      for filtered in [False, True]:
        runner = MakeAsimov()
        self.assertFalse(runner.apply_range_filter)
        options = {'cfg': 'config.py', 'file_name': 'example',
          'density_model': {'name': 'example_density', 'file_loc': str(root), 'category': 'inclusive'},
          'parameters': str(metadata), 'model_input': str(root/'models'),
          'data_output': str(root/str(filtered)), 'only_density': True,
          'n_asimov_events': 1000, 'scale_up': 5., 'verbose': False}
        if filtered:
          options['apply_range_filter'] = True
        runner.Configure(options)
        with patch('make_asimov.LoadConfig', return_value={'models': {'example': {}}}), \
             patch('make_asimov.GetDefaultsInModel', return_value={}):
          runner.Run()
        frame = pd.read_parquet(root/str(filtered)/'asimov.parquet')
        self.assertEqual(len(frame), 1000)
        self.assertAlmostEqual(frame.wt.sum(), 100.)
        self.assertTrue(np.isfinite(frame.to_numpy()).all())
        outputs[filtered] = frame
      self.assertTrue(((outputs[True].x > 4.) & (outputs[True].x < 6.)).all())
      self.assertTrue(((outputs[False].x < 3.) | (outputs[False].x > 7.)).any())

  def test_cli_flag_is_forwarded_to_all_asimov_and_density_metric_dispatches(self):
    tree = ast.parse((ROOT/'scripts/innfer.py').read_text())
    checked = 0
    for call in ast.walk(tree):
      if not isinstance(call, ast.Call):
        continue
      keywords = {key.arg: key.value for key in call.keywords}
      name = keywords.get('module_name')
      if not isinstance(name, ast.Constant) or name.value not in ['make_asimov', 'density_performance_metrics', 'hyperparameter_scan', 'bayesian_hyperparameter_tuning']:
        continue
      config = keywords['config']
      fields = {key.value: value for key, value in zip(config.keys, config.values) if isinstance(key, ast.Constant)}
      self.assertEqual(ast.unparse(fields['apply_range_filter']), 'args.asimov_apply_range_filter')
      checked += 1
    self.assertEqual(checked, 11)


if __name__ == '__main__':
  unittest.main()
