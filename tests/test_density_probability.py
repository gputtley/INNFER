"""Physical density conversion must include each component's spline Jacobian."""
import ast
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault('EVENTS_PER_BATCH', '1000')
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python/worker'))
from density_probability import UnTransformDensityProbability


class PhysicalDensityTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.path = Path(self.temp.name)/'cdf.pkl'
    self.forward = PchipInterpolator([0., .25, .5, .75, 1.], [0., .1, .3, .6, 1.])
    with self.path.open('wb') as file:
      pickle.dump(self.forward, file)
    self.parameters = {'X_columns': ['x', 'v'], 'Y_columns': ['mass'],
      'standardisation': {'x': {'mean': 0., 'std': 1.}, 'v': {'mean': 2., 'std': 3.},
                          'mass': {'mean': 172.5, 'std': .5}},
      'spline_to_gaussian': {'forward': {'x': str(self.path)}, 'inverse': {}}}
    self.X = pd.DataFrame({'x': [.12, .38, .81], 'v': [-1., 2., 5.]}, index=[8, 2, 8])

  def tearDown(self):
    self.temp.cleanup()

  def gaussian(self, X):
    z = norm.ppf(self.forward(X.x.to_numpy()))
    v = (X.v.to_numpy()-2.)/3.
    return pd.DataFrame({'log_prob': norm.logpdf(z)+norm.logpdf(v)})

  def expected(self, X):
    return np.log(self.forward.derivative()(X.x.to_numpy()))+norm.logpdf((X.v.to_numpy()-2.)/3.)-np.log(3.)

  def test_change_of_variables_for_both_backend_converters(self):
    for filename in ['bayes_flow_network.py', 'keras_flow_network.py']:
      tree = ast.parse((ROOT/'python/worker'/filename).read_text())
      cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ['BayesFlowNetwork','KerasFlowNetwork'])
      method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_UnTransformProbability')
      namespace = {'UnTransformDensityProbability': UnTransformDensityProbability}
      exec(compile(ast.Module(body=[method], type_ignores=[]), filename, 'exec'), namespace)
      for cached in [False, True]:
        instance = SimpleNamespace(data_parameters=self.parameters, cache_observable_transforms=cached,
          probability_transform_cache={}, max_cached_transform_batches=2)
        actual = namespace['_UnTransformProbability'](instance, self.gaussian(self.X), X=self.X)
        np.testing.assert_allclose(actual.log_prob, self.expected(self.X), atol=1e-12)
      public = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'Probability')
      calls = [n for n in ast.walk(public) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == '_UnTransformProbability']
      self.assertTrue(calls)
      self.assertTrue(all(any(k.arg == 'X' for k in n.keywords) for n in calls))

  def test_density_integrates_to_one_in_physical_coordinates(self):
    X = pd.DataFrame({'x': np.linspace(1e-7, 1.-1e-7, 10001), 'v': 2.})
    actual = UnTransformDensityProbability(self.gaussian(X), X, self.parameters).log_prob.to_numpy()
    # Divide out the independent v density, leaving the normalized x density.
    integral = np.trapz(np.exp(actual)/(norm.pdf(0.)/3.), X.x.to_numpy())
    self.assertAlmostEqual(integral, 1., places=5)

  def test_cache_is_keyed_by_values_order_and_metadata_and_is_bounded(self):
    cache = {}
    for X in [self.X, self.X.iloc[::-1], self.X.assign(x=[.2,.4,.6]), self.X]:
      actual = UnTransformDensityProbability(self.gaussian(X), X, self.parameters, cache=cache, max_cached_batches=2)
      np.testing.assert_allclose(actual.log_prob, self.expected(X), atol=1e-12)
      self.assertLessEqual(len(cache['batches']), 2)
    self.parameters['standardisation']['v']['std'] = 6.
    actual = UnTransformDensityProbability(self.gaussian(self.X), self.X, self.parameters, cache=cache)
    np.testing.assert_allclose(actual.log_prob, self.expected(self.X)-np.log(2.), atol=1e-12)
    self.assertEqual(len(cache['batches']), 1)

  def test_condition_derivatives_receive_only_condition_scaling(self):
    frame = pd.DataFrame({'d_log_prob_by_d_mass': [1., -2.], 'd2_log_prob_by_d_mass_and_mass': [3., 4.]})
    for cache in [None, {}]:
      actual = UnTransformDensityProbability(frame, None, self.parameters, cache=cache)
      np.testing.assert_allclose(actual.iloc[:,0], frame.iloc[:,0]/.5)
      np.testing.assert_allclose(actual.iloc[:,1], frame.iloc[:,1]/.25)

  def test_training_coordinates_are_not_given_a_physical_jacobian(self):
    frame = self.gaussian(self.X)
    pd.testing.assert_frame_equal(UnTransformDensityProbability(frame, None, self.parameters, transform_X=False, cache={}), frame)

  def test_no_spline_preserves_linear_conversion(self):
    parameters = {**self.parameters, 'spline_to_gaussian': {'forward': {}, 'inverse': {}}}
    for cache in [None, {}]:
      actual = UnTransformDensityProbability(self.gaussian(self.X), self.X, parameters, cache=cache)
      np.testing.assert_allclose(actual.log_prob, self.gaussian(self.X).log_prob-np.log(3.))

  def test_outside_cdf_support_has_zero_density(self):
    X = pd.DataFrame({'x': [-1., 2.], 'v': [2., 2.]})
    for cache in [None, {}]:
      actual = UnTransformDensityProbability(pd.DataFrame({'log_prob': [np.nan, np.nan]}), X,
                                            self.parameters, cache=cache)
      self.assertTrue(np.all(np.isneginf(actual.log_prob)))


if __name__ == '__main__':
  unittest.main()
