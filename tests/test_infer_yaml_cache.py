"""Inference metadata caching preserves file updates and pruning decisions."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python/worker'), str(ROOT/'python/runner')]
from infer import Infer


class InferYamlCacheTests(unittest.TestCase):

  def test_repeated_reads_parse_once_and_return_independent_metadata(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'parameters.yaml'
      path.write_text('classifier:\n  nuisance:\n    X_columns: [mass]\n')
      runner = Infer()
      with patch('infer.yaml.load', wraps=yaml.load) as load:
        first = runner._LoadYaml(path)
        first['classifier']['nuisance']['X_columns'].append('btag')
        second = runner._LoadYaml(path.parent/'.'/path.name)
      self.assertEqual(load.call_count, 1)
      self.assertEqual(second['classifier']['nuisance']['X_columns'], ['mass'])

  def test_changed_replaced_and_deleted_files_do_not_return_stale_metadata(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'metrics.yaml'
      path.write_text('q75: 1.00\n')
      runner = Infer()
      self.assertEqual(runner._LoadYaml(path)['q75'], 1.)
      previous = path.stat()
      path.write_text('q75: 1.02\n')
      os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 10**9))
      self.assertEqual(runner._LoadYaml(path)['q75'], 1.02)
      replacement = path.with_suffix('.new')
      replacement.write_text('q75: 1.04\n')
      os.utime(replacement, ns=(previous.st_atime_ns, path.stat().st_mtime_ns))
      replacement.replace(path)
      self.assertEqual(runner._LoadYaml(path)['q75'], 1.04)
      path.unlink()
      with self.assertRaises(FileNotFoundError):
        runner._LoadYaml(path)

  def test_full_loader_semantics_and_python_fallback(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)/'parameters.yaml'
      path.write_text('binning: !!python/tuple [1, 2]\n')
      expected = yaml.load(path.read_text(), Loader=yaml.FullLoader)
      self.assertEqual(Infer()._LoadYaml(path), expected)
      if hasattr(yaml, 'CFullLoader'):
        with patch.dict(yaml.__dict__):
          del yaml.CFullLoader
          self.assertEqual(Infer()._LoadYaml(path), expected)

  def test_classifier_loading_reuses_metadata_and_preserves_pruning(self):
    with tempfile.TemporaryDirectory() as directory:
      folder = Path(directory)
      metrics = {
        'at_threshold': {'q75': 1.01, 'q90': 1.05},
        'above_q75': {'q75': 1.02, 'q90': 1.00},
        'above_q90': {'q75': 1.00, 'q90': 1.06},
        'below_threshold': {'q75': 1.00, 'q90': 1.00},
      }
      parameters = folder/'parameters.yaml'
      parameters.write_text(yaml.safe_dump({
        'file_name': 'ttbar', 'yields': {'lnN': {}},
        'classifier': {key: {'X_columns': ['mass']} for key in metrics},
      }))
      runner = Infer()
      runner.verbose = False
      runner.model_input = str(folder)
      runner.prune_classifier_models = {'q75': 1.01, 'q90': 1.05}
      runner.classifier_models = {'2223': {'ttbar': []}}
      runner.classifier_pruning_files = {'2223': {'ttbar': {}}}
      runner.Y_columns = ['above_q75', 'above_q90']
      runner.initial_best_fit_guess = pd.DataFrame([{key: 0. for key in runner.Y_columns}])
      for parameter, values in metrics.items():
        pruning_path = folder/f'{parameter}.yaml'
        pruning_path.write_text(yaml.safe_dump(values))
        runner.classifier_pruning_files['2223']['ttbar'][parameter] = str(pruning_path)
        runner.classifier_models['2223']['ttbar'].append({
          'parameter': parameter, 'parameters': str(parameters),
          'name': parameter, 'file_loc': 'unused',
        })
        # Pruned checkpoints need not be read or constructed.
        if parameter in runner.Y_columns:
          model_folder = folder/parameter
          model_folder.mkdir()
          (model_folder/'ttbar_architecture.yaml').write_text('type: FCNN\n')

      with patch('infer.InitiateClassifierModel', side_effect=lambda *args, **kwargs: Mock()) as initiate, \
        patch('infer.yaml.load', wraps=yaml.load) as load:
        networks, splines = runner._BuildClassifierModels()
        runner._BuildClassifierModels()
      self.assertEqual(set(networks['2223']['ttbar']), set(runner.Y_columns))
      self.assertEqual(splines, {'2223': {'ttbar': {}}})
      self.assertEqual(initiate.call_count, 4)
      self.assertEqual(load.call_count, len(metrics) + 1 + len(runner.Y_columns))
      self.assertEqual(runner.freeze, {})
      for parameter, network in networks['2223']['ttbar'].items():
        network.Load.assert_called_once_with(name=f'{folder}/{parameter}/ttbar.h5')

  def test_disabled_pruning_does_not_read_pruning_files(self):
    runner = Infer()
    runner.Y_columns = ['nuisance']
    runner.verbose = False
    runner.model_input = 'models'
    runner.classifier_models = {'2223': {'ttbar': [{
      'parameter': 'nuisance', 'parameters': 'parameters.yaml',
      'name': 'classifier', 'file_loc': 'unused',
    }]}}
    runner.classifier_pruning_files = {'2223': {'ttbar': {'nuisance': 'missing.yaml'}}}
    metadata = {'file_name': 'ttbar', 'classifier': {'nuisance': {}}, 'yields': {'lnN': {}}}
    with patch.object(runner, '_LoadYaml', side_effect=[metadata, {'type': 'FCNN'}]) as load, \
      patch('infer.InitiateClassifierModel', return_value=Mock()):
      networks, _ = runner._BuildClassifierModels()
    self.assertEqual([call.args[0] for call in load.call_args_list],
      ['parameters.yaml', 'models/classifier/ttbar_architecture.yaml'])
    self.assertIn('nuisance', networks['2223']['ttbar'])


if __name__ == '__main__':
  unittest.main()
