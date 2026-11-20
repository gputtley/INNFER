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


def check_package_requirements(packages):
  """Validate metadata, retaining setup.sh's exact tested wrapt workaround."""
  from importlib.metadata import requires, version
  from packaging.requirements import Requirement
  setup_versions = {'tensorflow': '2.15.1', 'snakemake': '9.9.0',
    'snakemake-interface-storage-plugins': '4.2.3', 'wrapt': '1.14.1'}
  accepted = []
  for package in packages:
    for entry in requires(package) or []:
      requirement = Requirement(entry)
      if requirement.marker is not None and not requirement.marker.evaluate():
        continue
      installed = version(requirement.name)
      if installed in requirement.specifier:
        continue
      if (package == 'snakemake-interface-storage-plugins' and requirement.name == 'wrapt'
          and all(version(name) == expected for name, expected in setup_versions.items())):
        accepted.append(f'{package}: {requirement} with wrapt=={installed} (setup.sh workaround)')
        continue
      raise RuntimeError(f'{package} requires {requirement}; installed {requirement.name}=={installed}')
  return accepted


def check_runtime_dependencies():
  """Exercise CPU ONNX inference and numerical derivatives without external files."""
  from importlib.metadata import version
  accepted = check_package_requirements(['onnxruntime', 'numdifftools', 'tensorflow',
    'snakemake', 'snakemake-interface-storage-plugins'])

  import numpy as np
  import numdifftools as nd
  import onnxruntime as ort

  # ONNX Identity graph: IR 8, opset 13, two float32 values (x -> y).
  # Store the tiny protobuf directly so the ONNX model-building package is not needed.
  model = (b'\x08\x08\x3a\x3b\x0a\x10\x0a\x01x\x12\x01y\x22\x08Identity'
    b'\x12\x05smoke\x5a\x0f\x0a\x01x\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x02'
    b'\x62\x0f\x0a\x01y\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x02\x42\x02\x10\x0d')
  options = ort.SessionOptions()
  options.intra_op_num_threads = 1
  options.inter_op_num_threads = 1
  session = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
  values = np.array([1., -2.], dtype=np.float32)
  np.testing.assert_array_equal(session.run(None, {'x': values})[0], values)
  np.testing.assert_allclose(nd.Derivative(np.sin)(0.), 1., rtol=1e-6)
  print('CPU ONNX inference and numerical derivatives passed', flush=True)
  return {'onnxruntime': ort.__version__, 'numdifftools': nd.__version__,
    'protobuf': version('protobuf'), 'wrapt': version('wrapt'),
    'snakemake': version('snakemake'), 'setup_dependency_exceptions': accepted}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--require-gpu', action='store_true')
  parser.add_argument('--dependencies-only', action='store_true',
    help='Check runtime packages before packing the host environment')
  args = parser.parse_args()
  if not args.require_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  os.environ['TF_NUM_INTEROP_THREADS'] = '2'
  os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
  os.environ['OMP_NUM_THREADS'] = '2'

  dependencies = check_runtime_dependencies()
  if args.dependencies_only:
    print(json.dumps(dependencies, indent=2))
    return

  import bayesflow
  import numpy as np
  import pandas as pd
  import pyarrow
  import tensorflow as tf
  import tensorflow_probability as tfp
  from useful_functions import InitiateDensityModel
  from python.worker import asym_log_normal
  from python.worker.useful_functions import GetDefaultsInModel

  root = Path('/opt/innfer/app')
  assert Path(sys.executable).is_relative_to('/opt/innfer/env'), sys.executable
  assert Path(bayesflow.__file__).is_relative_to(root/'python/BayesFlow'), bayesflow.__file__
  assert Path(asym_log_normal.__file__).is_relative_to(root), asym_log_normal.__file__
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

    snakefile = folder/'Snakefile'
    snakefile.write_text('rule all:\n  input: "scheduler-check.txt"\n\n'
      'rule check:\n  output: "scheduler-check.txt"\n'
      '  shell: "printf passed > {output}"\n')
    run_checked([sys.executable, '-m', 'snakemake', '--snakefile', str(snakefile),
      '--cores', '1', '--scheduler', 'greedy', '--profile', 'none',
      '--workflow-profile', 'none'], cwd=folder)
    assert (folder/'scheduler-check.txt').read_text() == 'passed'
    print('Snakemake: local workflow execution passed', flush=True)

    tex = folder/'table.tex'
    tex.write_text(r'\documentclass{article}\usepackage{booktabs,longtable,array,geometry}'
                   r'\begin{document}\begin{tabular}{lr}\toprule Metric & Value\\'
                   r'\midrule Check & 1\\\bottomrule\end{tabular}\end{document}')
    run_checked(['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
                 '-halt-on-error', tex.name], cwd=folder)
    assert tex.with_suffix('.pdf').stat().st_size > 0

  run_checked([sys.executable, str(root/'scripts/innfer.py'), '--help'])
  print(json.dumps({**dependencies, 'source_sha256': manifest['source_sha256'],
    'python': sys.version.split()[0], 'tensorflow': tf.__version__,
    'tensorflow_probability': tfp.__version__, 'pyarrow': pyarrow.__version__,
    'bayesflow_source': bayesflow.__file__,
    'gpu_tested': args.require_gpu, 'status': 'passed'}, indent=2))


if __name__ == '__main__':
  main()
