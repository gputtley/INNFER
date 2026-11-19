"""Overlapping checkpoint evaluations must not overwrite or delete each other's samples."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python/worker'), str(ROOT/'python/runner'), str(ROOT/'python/BayesFlow')]
from density_performance_metrics import DensityPerformanceMetrics


class DensityCheckpointSamplesTests(unittest.TestCase):
  def runner(self, folder, checkpoint, **options):
    runner = DensityPerformanceMetrics()
    runner.Configure({'cfg': 'config.py', 'parameters': str(folder/'parameters.yaml'),
      'file_name': 'ttbar', 'category': '24', 'data_output': str(folder/'metrics'),
      'val_file_loc': 'validation', 'save_extra_name': checkpoint,
      'do_loss': False, 'do_inference': False, 'do_histogram_metrics': True,
      'do_multidimensional_dataset_metrics': False, 'histogram_datasets': ['val'],
      'multidimensional_datasets': [], 'inference_datasets': ['val'],
      'verbose': False, 'make_metrics_table': False, 'tidy_up_asimov': True, **options})
    return runner

  def test_interleaved_generation_and_cleanup_preserve_each_checkpoint(self):
    with tempfile.TemporaryDirectory() as directory:
      folder = Path(directory)
      (folder/'parameters.yaml').write_text(yaml.safe_dump({'density': {'Y_columns': ['bw_mass']}}))
      first = self.runner(folder, '_epoch_0')
      second = self.runner(folder, '_epoch_1')
      paths = []

      class Samples:
        def Configure(self, options):
          self.options = options

        def Run(self):
          output = Path(self.options['data_output'])/'asimov.parquet'
          output.parent.mkdir(parents=True, exist_ok=True)
          output.write_text(self.options['model_extra_name'])
          paths.append(output)

      def measure_second():
        _, files = second._GetFiles(0, 'val')
        second.metrics['sample_checkpoint'] = Path(files[0]).read_text()

      def measure_first():
        # Interleave a complete second evaluation after first has generated its
        # sample. Its cleanup must not remove the first checkpoint's input.
        second.Run()
        _, files = first._GetFiles(0, 'val')
        first.metrics['sample_checkpoint'] = Path(files[0]).read_text()

      with patch('density_performance_metrics.LoadConfig', return_value={}), \
        patch('density_performance_metrics.GetValidationLoop', return_value=[{'bw_mass': 172.5}]), \
        patch('density_performance_metrics.GetModelLoop', return_value=[{}]), \
        patch('density_performance_metrics.SkipNonDensity', return_value=False), \
        patch('density_performance_metrics.SkipEmptyDataset', return_value=False), \
        patch('density_performance_metrics.MakeAsimov', Samples), \
        patch.object(first, 'DoHistogramMetrics', side_effect=measure_first), \
        patch.object(second, 'DoHistogramMetrics', side_effect=measure_second):
        first.Run()

      self.assertEqual(first.metrics['sample_checkpoint'], '_epoch_0')
      self.assertEqual(second.metrics['sample_checkpoint'], '_epoch_1')
      self.assertEqual(len(set(paths)), 2)
      self.assertTrue(all(not path.exists() for path in paths))

  def test_declared_outputs_and_toy_inputs_match_checkpoint_sample_paths(self):
    with tempfile.TemporaryDirectory() as directory:
      runner = self.runner(Path(directory), '_epoch_4', metrics_save_extra_name='_toy_3',
        synth_vs_synth=True, tidy_up_asimov=False)
      runner.alternative_asimov_seed = 45
      with patch('density_performance_metrics.LoadConfig', return_value={}), \
        patch('density_performance_metrics.GetValidationLoop', return_value=[{'bw_mass': 172.5}]), \
        patch('density_performance_metrics.SkipNonDensity', return_value=False), \
        patch('density_performance_metrics.SkipEmptyDataset', return_value=False):
        outputs = runner.Outputs()
      alternative, reference = runner._GetFiles(0, 'val')
      self.assertIn(reference[0], outputs)
      self.assertNotEqual(reference[0], alternative[0])
      self.assertIn('_epoch_4_toy_3_', reference[0])
      self.assertIn('_epoch_4_toy_3_', alternative[0])
      # Non-checkpoint PValue workflows keep their existing directory names.
      runner.save_extra_name = ''
      self.assertEqual(runner._GetAsimovDirectory(0, 'val', 42),
        str(Path(directory)/'metrics/val_ind_0_toy_3_seed_42_for_val'))


if __name__ == '__main__':
  unittest.main()
