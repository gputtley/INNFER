"""EMA evaluation contexts shared by the density training backends."""
import os
from contextlib import contextmanager


def RawDensityModelPath(path):
  """Keep non-averaged comparison models in a separate, ordinary model directory."""
  directory, filename = os.path.split(os.fspath(path))
  return os.path.join(f"{directory}_raw", filename)


@contextmanager
def UseEMAWeights(optimizer, variables):
  """Temporarily evaluate/save averages, restoring optimizer weights even on error."""
  if not getattr(optimizer, "use_ema", False):
    yield
    return
  variables = list(variables)
  if not variables:
    raise ValueError("Build the density model before evaluating EMA weights")
  # Seed averages from the actual starting weights and register all variables,
  # including any coupling layers initially frozen during training.
  optimizer.build(variables)
  original = [variable.read_value() for variable in variables]
  try:
    optimizer.finalize_variable_values(variables)
    yield
  finally:
    for variable, value in zip(variables, original):
      variable.assign(value)
