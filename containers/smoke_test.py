"""Check an image without external data, model weights or the original checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run_checked(command, **options):
  """Keep successful checks quiet, but show the tool's actual failure output."""
  result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, **options)
  if result.returncode:
    print(result.stdout, file=sys.stderr, flush=True)
    result.check_returncode()
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--require-gpu', action='store_true')
  args = parser.parse_args()
  if not args.require_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  os.environ['TF_NUM_INTEROP_THREADS'] = '2'
  os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
  os.environ['OMP_NUM_THREADS'] = '2'

  import bayesflow
  import numpy as np
  import pandas as pd
  import pyarrow
  import tensorflow as tf
  import tensorflow_probability as tfp
  from useful_functions import InitiateDensityModel

  root = Path('/opt/innfer/app')
  assert Path(sys.executable).is_relative_to('/opt/innfer/env'), sys.executable
  assert Path(bayesflow.__file__).is_relative_to(root/'python/BayesFlow'), bayesflow.__file__
  manifest = json.loads((root/'container-manifest.json').read_text())
  if args.require_gpu and not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('The image cannot see a GPU; run with --nv on a GPU worker')

  with tempfile.TemporaryDirectory(prefix='innfer-smoke-') as folder:
    folder = Path(folder)
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(16, 2)), columns=['x0', 'x1'])
    Y = pd.DataFrame({'mass': rng.normal(size=16)})
    for split in ['train', 'test']:
      for label, frame in [('X', X), ('Y', Y), ('wt', pd.DataFrame({'wt': np.ones(16)}))]:
        frame.to_parquet(folder/f'{label}_{split}.parquet', index=False)
    parameters = {'X_columns': list(X.columns), 'Y_columns': ['mass']}
    for kind in ['BayesFlow', 'KerasFlow']:
      architecture = {'type': kind, 'coupling_design': 'affine', 'num_coupling_layers': 2,
        'permutation': 'fixed' if kind == 'BayesFlow' else 'none',
        'affine_num_dense_layers': 1, 'affine_units_per_dense_layer': 8,
        'affine_dropout': False, 'affine_mc_dropout': False,
        'hidden_units': [8], 'l2_lambda': 0., 'use_summary_network': False}
      network = InitiateDensityModel(architecture, str(folder),
        options={'data_parameters': parameters, 'batch_size': 8})
      network.BuildModel()
      if kind == 'BayesFlow':
        network.inference_net(X.to_numpy(np.float32), Y.to_numpy(np.float32))
        network._SaveWeights(str(folder/f'{kind}.h5'))
      else:
        network.Save(str(folder/f'{kind}.h5'))
      network.Load(str(folder/f'{kind}.h5'))
      logp = network.Probability(X, Y, transform_X=False, transform_Y=False)
      samples = network.Sample(Y.iloc[:1], 8, seed=42, transform_X=False, transform_Y=False)
      assert np.isfinite(logp).all() and np.isfinite(samples.to_numpy()).all(), kind
      print(f'{kind}: checkpoint round trip, density and sampling passed', flush=True)

    tex = folder/'table.tex'
    tex.write_text(r'\documentclass{article}\usepackage{booktabs,longtable,array,geometry}'
                   r'\begin{document}\begin{tabular}{lr}\toprule Metric & Value\\'
                   r'\midrule Check & 1\\\bottomrule\end{tabular}\end{document}')
    run_checked(['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
                 '-halt-on-error', tex.name], cwd=folder)
    assert tex.with_suffix('.pdf').stat().st_size > 0

  run_checked([sys.executable, str(root/'scripts/innfer.py'), '--help'])
  print(json.dumps({'source_sha256': manifest['source_sha256'],
    'python': sys.version.split()[0], 'tensorflow': tf.__version__,
    'tensorflow_probability': tfp.__version__, 'pyarrow': pyarrow.__version__,
    'bayesflow_source': bayesflow.__file__,
    'gpu_tested': args.require_gpu, 'status': 'passed'}, indent=2))


if __name__ == '__main__':
  main()
