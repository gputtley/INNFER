"""Asimov components must use distinct, reproducible random streams."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python/worker'), str(ROOT/'python/runner')]
from make_asimov import MakeAsimov
from useful_functions import GetCategoryLoop, GetModelFileLoop


class AsimovSeedTests(unittest.TestCase):
  cfg = {'models': {'ttbar': {}, 'other': {}},
         'categories': {'run2': '', '2223': '', '24': ''}}

  def runner(self, file_name='ttbar', category='run2', val_ind=0, seed=42):
    runner = MakeAsimov()
    runner.Configure({'file_name': file_name, 'density_model': {'category': category},
                      'val_ind': val_ind, 'seed': seed})
    return runner

  def test_reproducible_but_distinct_across_components_hypotheses_and_base_seeds(self):
    seeds = []
    for base_seed in [0, 1, 42, 43, 100]:
      for file_name in GetModelFileLoop(self.cfg):
        for category in GetCategoryLoop(self.cfg):
          for val_ind in range(5):
            runner = self.runner(file_name, category, val_ind, base_seed)
            seed = runner._GetSeed(self.cfg)
            self.assertEqual(seed, self.runner(file_name, category, val_ind, base_seed)._GetSeed(self.cfg))
            seeds.append(seed)
    self.assertEqual(len(set(seeds)), len(seeds))

  def test_filtering_does_not_renumber_selected_components(self):
    seeds = {(file_name, category): self.runner(file_name, category, 2)._GetSeed(self.cfg)
             for file_name in GetModelFileLoop(self.cfg)
             for category in GetCategoryLoop(self.cfg)}
    for file_name in GetModelFileLoop(self.cfg, specific_file_name=['other']):
      for category in GetCategoryLoop(self.cfg, specific_category='24'):
        self.assertEqual(self.runner(file_name, category, 2)._GetSeed(self.cfg), seeds[file_name, category])

  def test_large_paired_indices_fit_numpy_seed_range(self):
    seed = self.runner(val_ind=10**6, seed=10**30)._GetSeed(self.cfg)
    self.assertGreaterEqual(seed, 0)
    self.assertLess(seed, 2**32)
    first = np.random.RandomState(seed).normal(size=10)
    np.testing.assert_array_equal(first, np.random.RandomState(seed).normal(size=10))

  def test_inclusive_category_and_default_validation_index(self):
    cfg = {'models': {'ttbar': {}}}
    runner = self.runner(category='inclusive')
    self.assertEqual(runner.val_ind, MakeAsimov().val_ind)
    self.assertIsInstance(runner._GetSeed(cfg), int)
    for field in ['seed', 'val_ind']:
      with self.subTest(field=field):
        invalid = self.runner(category='inclusive')
        setattr(invalid, field, -1)
        with self.assertRaisesRegex(ValueError, 'non-negative'):
          invalid._GetSeed(cfg)


if __name__ == '__main__':
  unittest.main()
