"""Binned workflow groups must agree across BW mixtures, normalization and fits."""
import ast
import copy
import fnmatch
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('PREP_DATA_DIR', 'data')
os.environ.setdefault('PLOTS_DIR', 'plots')
os.environ.setdefault('EVENTS_PER_BATCH_FOR_PREPROCESS', '1000')

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python/worker'), str(ROOT/'python/worker/custom'), str(ROOT/'python/runner/custom_module')]
import top_bw_fractions as bw_module
from top_bw_fractions import top_bw_fractions


def workflow_functions():
  tree = ast.parse((ROOT/'python/worker/useful_functions.py').read_text())
  functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
    and node.name in ['GetCategoryLoop', 'GetSnakeMakeStepLoop']]
  namespace = {'copy': copy, 'yaml': yaml, 'fnmatch': fnmatch}
  exec(compile(ast.Module(body=functions, type_ignores=[]), '<workflow helpers>', 'exec'), namespace)
  return namespace


class BinnedWorkflowTests(unittest.TestCase):
  def test_full_workflow_uses_valid_groups_and_isolated_fit_comparisons(self):
    functions = workflow_functions()
    directory = Path.cwd()
    try:
      os.chdir(ROOT)
      steps = functions['GetSnakeMakeStepLoop'](yaml.safe_load((ROOT/'configs/snakemake/btm_binned.yaml').read_text()))
    finally:
      os.chdir(directory)
    self.assertEqual(steps[0]['step'], 'LoadData')
    custom = [step['run_options'] for step in steps if step['step'] == 'Custom']
    options = {item.split(':', 1)[0]:item.split(':', 1)[1] for item in custom[0]['custom_options'].split(';')}
    self.assertEqual(options['group_by_base_file'], 'True')
    self.assertEqual(custom[1]['custom_options'], 'parameters:CR1,CR2,ERDOn,ue,hdamp')
    self.assertEqual([s['step'] for s in steps].count('PreProcessParallelMerge'), 1)
    factors = next(s for s in steps if s['step'] == 'SimToDataFactors')
    self.assertEqual(factors['run_options']['sim_to_data_norm_groups'], 'run2_*;2223_*;24_*')
    fits = [s for s in steps if s['step'] == 'InitialFit']
    self.assertEqual(len(fits), 8)
    comparisons = [s for s in steps if s['step'] == 'ScanPlot' and s.get('run_options', {}).get('scan_plot_breakdown')]
    self.assertEqual(len(comparisons), 4)
    for era in [None, 'run2', '2223', '24']:
      pattern = None if era is None else era+'_*'
      selected = [s for s in fits if s['run_options'].get('specific_category') == pattern]
      self.assertEqual(len(selected), 2)
      self.assertEqual({s['run_options']['extra_dir_name'] for s in selected}, {'Asimov','AsimovStatOnly'})
      comparison = next(s for s in comparisons if s['run_options'].get('specific_category') == pattern)
      suffix = '' if era is None else '_'+era
      self.assertEqual(comparison['run_options']['other_input'], f'Stat. Only:ScanCollectAsimovStatOnly{suffix}:scan_collect')
      self.assertEqual(comparison['run_options'].get('extra_extra_dir_name', ''), suffix)
      self.assertFalse(comparison['run_options'].get('add_specific_category_to_dir_name', False))

  def test_new_config_loads_with_expanded_channels(self):
    module = 'btm_common_cfg_inc_24_fsr_decorrelation_smaller_range'
    with patch(module+'.os.path.exists', return_value=True), \
        patch(module+'.glob.glob', side_effect=lambda pattern:[pattern.replace('*','part0')]), \
        patch(module+'.socket.getfqdn', return_value='test.local'):
      cfg = runpy.run_path(str(ROOT/'configs/run/btm_full_171126_binned.py'))['config']
    counts = [len(workflow_functions()['GetCategoryLoop'](cfg, era+'_*')) for era in ['run2','2223','24']]
    self.assertEqual(counts, [10,8,10])
    self.assertEqual(len(cfg['files']), 12)
    self.assertTrue(cfg['data_file'])


class BWGroupingTests(unittest.TestCase):
  def config(self):
    cfg = {'name': 'toy', 'variables': ['x'], 'categories': {
      'r_Signal': '(year_ind==0) & (x>1)', 'r_Background': '(year_ind==0) & (x<=1)',
      'b_Signal': '(year_ind==1) & (x>0)'},
      'inference': {'binned_fit': {'files': {'ttbar': [
        {'file': 'base_ttbar_r', 'categories': ['r_Signal','r_Background']},
        {'file': 'base_ttbar_b', 'categories': ['b_Signal']}]}}}, 'files': {}}
    for era, expression, column, post in [('r', 'x+10', 'derived_r', 'x<3'), ('b', 'x*2', 'derived_b', 'x>0')]:
      cfg['files'][f'base_ttbar_{era}'] = {'weight': 'wt', 'calculate': {column: expression},
        'post_calculate_selection': post,
        'weight_shifts': {'bw_fractions': {'args': {'category': era}}}}
    return cfg

  def runner(self, cfg, **options):
    runner = top_bw_fractions()
    with patch.object(bw_module, 'LoadConfig', return_value=cfg):
      runner.Configure({'cfg': 'toy.py', 'options': {'group_by_base_file': 'True', **options}})
    return runner

  def test_inputs_outputs_and_selection_union_match_bw_consumers(self):
    cfg = self.config()
    runner = self.runner(cfg, plot='True', plot_dist='True')
    with patch.object(bw_module, 'LoadConfig', return_value=cfg):
      inputs, outputs = runner.Inputs(), runner.Outputs()
    self.assertEqual(inputs, ['toy.py', 'data/toy/LoadData/base_ttbar_r.parquet', 'data/toy/LoadData/base_ttbar_b.parquet'])
    self.assertIn('plots/toy/top_bw_fractions/ttbar/fractions_spline_r.pdf', outputs)
    self.assertIn('plots/toy/top_bw_fractions/ttbar/bw_reweighted_x_b.pdf', outputs)
    setup = runner._GetCategorySetup(cfg)
    frame = pd.DataFrame({'year_ind': [0,0,0,1,1], 'x': [0,1,2,0,2]})
    np.testing.assert_array_equal(frame.eval(setup['r']['selection']), [1,1,1,0,0])
    np.testing.assert_array_equal(frame.eval(setup['b']['selection']), [0,0,0,0,1])

  def test_grouped_run_writes_era_keys_and_plot_only_reuses_them(self):
    cfg = self.config()
    with tempfile.TemporaryDirectory() as directory, \
        patch.object(bw_module, 'data_dir', directory), \
        patch.object(bw_module, 'LoadConfig', return_value=cfg), \
        patch.object(bw_module, 'GetDefaults', return_value={'bw_mass': 172.5}):
      runner = self.runner(cfg)
      calls = []
      def fractions(base, weight, **kwargs):
        calls.append((base, kwargs))
        return {172.5:{172.5:1.0}}, {172.5:np.poly1d([1.0])}
      with patch.object(runner, '_CalculateOptimalFractions', side_effect=fractions):
        runner.Run()
      locations = Path(directory)/'toy/top_bw_fractions/top_bw_fraction_locations.yaml'
      mapping = yaml.safe_load(locations.read_text())
      self.assertEqual(set(mapping), {'r','b'})
      self.assertTrue(all(Path(path).is_file() for group in mapping.values() for path in group.values()))
      self.assertTrue(all(Path(path).is_file() for path in runner.Outputs()))
      # Stored calculation callbacks must retain their own column and selection.
      for (_, call), column, selected_x in zip(calls, ['derived_r','derived_b'], [[0,1,2],[1,2,3]]):
        frame = pd.DataFrame({'x':[0,1,2,3], 'wt':[1.0]*4})
        for function in call['functions']:
          frame = function(frame)
        self.assertIn(column, frame)
        self.assertEqual(frame.x.tolist(), selected_x)
      plotter = self.runner(cfg, only_plot='True', plot='True')
      with patch.object(plotter, '_CalculateOptimalFractions', side_effect=AssertionError('must reuse splines')), \
          patch.object(plotter, '_PlotFractions') as plot:
        plotter.Run()
      self.assertEqual([call.args[1] for call in plot.call_args_list], ['r','b'])

  def test_default_mode_is_preserved_and_invalid_grouping_fails(self):
    cfg = self.config()
    legacy = self.runner(cfg, group_by_base_file='False')
    self.assertEqual(set(legacy._GetCategorySetup(cfg)), set(cfg['categories']))
    self.assertEqual(legacy._GetCategorySetup(cfg)['r_Signal']['selection'], cfg['categories']['r_Signal'])
    missing = copy.deepcopy(cfg)
    del missing['inference']['binned_fit']['files']['ttbar'][1]
    with self.assertRaisesRegex(ValueError, 'no BW base-file mapping'):
      self.runner(missing)._GetCategorySetup(missing)
    overlap = copy.deepcopy(cfg)
    overlap['inference']['binned_fit']['files']['ttbar'][1]['categories'].append('r_Signal')
    with self.assertRaisesRegex(ValueError, 'multiple BW base-file groups'):
      self.runner(overlap)._GetCategorySetup(overlap)

  def test_reweighting_plots_calculate_classifier_before_category_selection(self):
    cfg = self.config()
    runner = self.runner(cfg)
    with tempfile.TemporaryDirectory() as directory:
      runner.plot_dir = directory
      runner.batch_size = 4
      path = Path(directory)/'raw.parquet'
      # The rejected source mass must never reach the fraction lookup or plots.
      pd.DataFrame({'x': np.arange(10.),
        'sim_mass': [169.5, 169.5, 172.5, 173.5, 172.5, 173.5, 172.5, 173.5, 172.5, 173.5],
        'wt': np.ones(10)}).to_parquet(path, index=False)

      def calculate(df):
        df['btm_merged_classifier_pass'] = (df.x > 1).astype(float)
        df['GenTop1_mass'] = 170.+df.x
        df['GenTop2_mass'] = 171.+df.x
        return df

      original = runner._custom_info_apply_wt
      selected = []
      def capture(tmp, out, options={}):
        selected.extend(tmp.x.tolist())
        return original(tmp, out, options)

      fractions = {mass: {target: .5 for target in [172.5, 173.5]}
                   for mass in [172.5, 173.5]}
      with patch.object(bw_module, 'LoadConfig', return_value=cfg), \
          patch.object(bw_module, 'plot_histograms') as plot, \
          patch.object(bw_module, 'plot_histograms_with_ratio') as ratio_plot, \
          patch.object(runner, '_ApplyBWReweight', side_effect=lambda df, **kwargs: pd.Series(1., index=df.index)), \
          patch.object(runner, '_custom_info_apply_wt', side_effect=capture):
        runner._PlotReweighting(fractions, str(path), 'wt',
          selection='btm_merged_classifier_pass > 0.5', functions=[calculate], category='r_Signal')

      self.assertEqual(sorted(selected), list(np.arange(2.,10.)))
      self.assertEqual(plot.call_count, 3)
      self.assertEqual(ratio_plot.call_count, 3)


if __name__ == '__main__':
  unittest.main()
