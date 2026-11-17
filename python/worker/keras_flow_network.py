"""Conditional coupling flows implemented directly with TensorFlow/Keras."""
import hashlib
import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf

from data_loader import DataLoader
from data_processor import DataProcessor
from optimizer import Optimizer
from density_ema import RawDensityModelPath, UseEMAWeights
from useful_functions import MakeDirectories


class CouplingConditioner(tf.keras.Model):
  """One configurable MLP predicts all transform parameters for a coupling."""

  def __init__(self, units, output_dim, activation, dropout, residual, l2_lambda):
    super().__init__()
    regularizer = tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None
    self.hidden = [tf.keras.layers.Dense(width, activation=activation, kernel_regularizer=regularizer) for width in units]
    self.dropouts = [tf.keras.layers.Dropout(dropout) for _ in units]
    self.residual = residual
    self.output_layer = tf.keras.layers.Dense(output_dim, kernel_initializer="zeros", bias_initializer="zeros", kernel_regularizer=regularizer)

  def call(self, inputs, training=False):
    hidden = inputs
    for dense, dropout in zip(self.hidden, self.dropouts):
      updated = dropout(dense(hidden), training=training)
      hidden = hidden + updated if self.residual and hidden.shape[-1] == updated.shape[-1] else updated
    return self.output_layer(hidden)


class ConditionalCoupling(tf.keras.layers.Layer):
  """Affine or rational-quadratic spline coupling, including fixed mixing."""

  def __init__(self, dimension, permutation, settings, l2_lambda):
    super().__init__()
    self.order = tf.constant(permutation, tf.int32)
    self.reverse_order = tf.constant(np.argsort(permutation), tf.int32)
    self.split = dimension // 2
    self.transformed_dim = dimension - self.split
    self.design = settings["design"]
    self.scale_clip = settings["scale_clip"]
    self.bins = settings["spline_bins"]
    self.bound = settings["spline_bound"]
    self.min_bin_size = settings["min_bin_size"]
    self.min_slope = settings["min_slope"]
    count = 2 if self.design == "affine" else 3 * self.bins - 1
    self.conditioner = CouplingConditioner(settings["hidden_units"], self.transformed_dim * count,
      settings["activation"], settings["dropout"], settings["residual"], l2_lambda)

  def _GetParameters(self, fixed, conditions, training=False):
    inputs = tf.concat([fixed, conditions], axis=-1)
    # A constant input permits a trainable unconditional one-dimensional flow.
    if inputs.shape[-1] == 0:
      inputs = tf.ones([tf.shape(fixed)[0], 1], tf.float32)
    return self.conditioner(inputs, training=training)

  def _GetSpline(self, parameters):
    # Imported lazily: affine models do not need TensorFlow Probability.
    import tensorflow_probability as tfp
    parameters = tf.reshape(parameters, [-1, self.transformed_dim, 3 * self.bins - 1])
    widths, heights, slopes = tf.split(parameters, [self.bins, self.bins, self.bins - 1], axis=-1)
    size = 2.0 * self.bound - self.bins * self.min_bin_size
    widths = self.min_bin_size + size * tf.nn.softmax(widths, axis=-1)
    heights = self.min_bin_size + size * tf.nn.softmax(heights, axis=-1)
    # Zero conditioner output gives the identity map, including unit slopes.
    slopes = self.min_slope + tf.nn.softplus(slopes + np.log(np.expm1(1.0 - self.min_slope)))
    return tfp.bijectors.RationalQuadraticSpline(widths, heights, slopes, range_min=-self.bound)

  def call(self, inputs, conditions, training=False):
    permuted = tf.gather(inputs, self.order, axis=-1)
    fixed, changed = permuted[:, :self.split], permuted[:, self.split:]
    parameters = self._GetParameters(fixed, conditions, training=training)
    if self.design == "affine":
      shift, raw_scale = tf.split(parameters, 2, axis=-1)
      log_scale = self.scale_clip * tf.tanh(raw_scale / self.scale_clip)
      changed = changed * tf.exp(log_scale) + shift
      log_det = tf.reduce_sum(log_scale, axis=-1)
    else:
      spline = self._GetSpline(parameters)
      log_det = spline.forward_log_det_jacobian(changed, event_ndims=1)
      changed = spline.forward(changed)
    return tf.concat([fixed, changed], axis=-1), log_det

  def Inverse(self, inputs, conditions):
    fixed, changed = inputs[:, :self.split], inputs[:, self.split:]
    parameters = self._GetParameters(fixed, conditions, training=False)
    if self.design == "affine":
      shift, raw_scale = tf.split(parameters, 2, axis=-1)
      log_scale = self.scale_clip * tf.tanh(raw_scale / self.scale_clip)
      changed = (changed - shift) * tf.exp(-log_scale)
    else:
      changed = self._GetSpline(parameters).inverse(changed)
    return tf.gather(tf.concat([fixed, changed], axis=-1), self.reverse_order, axis=-1)


class ConditionalFlow(tf.keras.Model):
  """A normalized conditional density with an isotropic Gaussian base."""

  def __init__(self, dimension, couplings, condition_hidden_units, condition_dim, activation, l2_lambda):
    super().__init__()
    self.dimension = dimension
    self.couplings = couplings
    self.condition_encoder = None
    if condition_dim is not None:
      regularizer = tf.keras.regularizers.l2(l2_lambda) if l2_lambda else None
      self.condition_encoder = tf.keras.Sequential([
        *[tf.keras.layers.Dense(width, activation=activation, kernel_regularizer=regularizer) for width in condition_hidden_units],
        tf.keras.layers.Dense(condition_dim, kernel_regularizer=regularizer),
      ])

  def _Encode(self, conditions, training=False):
    return self.condition_encoder(conditions, training=training) if self.condition_encoder is not None else conditions

  def call(self, inputs, training=False):
    values, conditions = inputs
    conditions = self._Encode(conditions, training=training)
    log_det = tf.zeros(tf.shape(values)[0], tf.float32)
    for coupling in self.couplings:
      values, contribution = coupling(values, conditions, training=training)
      log_det += contribution
    return -0.5 * tf.reduce_sum(tf.square(values) + np.log(2.0 * np.pi), axis=-1) + log_det

  def Inverse(self, latent, conditions):
    conditions = self._Encode(conditions, training=False)
    for coupling in reversed(self.couplings):
      latent = coupling.Inverse(latent, conditions)
    return latent


class KerasFlowNetwork():
  """Repository density interface without a BayesFlow model or trainer.

  Training parquet tables are already transformed. Probability and Sample use
  the same DataProcessor transformations as the other density implementations.
  """

  def __init__(self, X_train=None, Y_train=None, wt_train=None, X_test=None, Y_test=None, wt_test=None, options=None):
    self.num_coupling_layers = 8
    self.coupling_design = "affine"
    self.hidden_units = [128, 128]
    self.activation = "elu"
    self.dropout = 0.0
    self.residual = False
    self.scale_clip = 2.0
    self.spline_bins = 8
    self.spline_bound = 5.0
    self.min_bin_size = 0.001
    self.min_slope = 0.001
    self.permutation = "random"
    self.coupling_layers = None
    self.condition_hidden_units = [32, 32]
    self.condition_dim = None
    self.seed = 42
    self.jit_compile = False
    self.epochs = 30
    self.batch_size = 2048
    self.learning_rate = 0.001
    self.optimizer_name = "AdamW"
    self.use_ema = False
    self.ema_momentum = 0.999
    self.lr_scheduler_name = "CosineDecayWithWarmUp"
    self.lr_scheduler_options = {}
    self.gradient_clipping_norm = None
    self.l2_lambda = 0.0001
    self.log_l2_lambda = None
    self.clip_negative_weight_loss = True
    self.shuffle_training = False
    self.shuffle_buffer_size = 65536
    self.shuffle_seed = 42
    self.early_stopping = False
    self.patience = 5
    self.tolerance = 0.0
    self.wait_till = 5
    self.disable_tqdm = False
    self.use_wandb = False
    self.save_model_per_epoch = False
    self.plot_loss = True
    self.plot_lr = True
    self.plot_dir = "plots"
    self.data_parameters = {}
    self.cache_observable_transforms = False
    self.max_cached_transform_batches = 1024
    self.prepared_X_cache = OrderedDict()
    self.probability_transform_cache = {}
    self.transform_metadata_key = None
    self.use_gaussian_cache = False
    for key, value in (options or {}).items():
      setattr(self, key, value)
    if self.log_l2_lambda is not None:
      self.l2_lambda = 10.0 ** float(self.log_l2_lambda)
    # These BayesFlow-specific training procedures have no equivalent here.
    for key in ["active_learning", "resample", "trainable_cl_per_epoch", "turn_conditions_on_at_epoch"]:
      if (options or {}).get(key):
        raise ValueError(f"{key} is not supported by KerasFlow")
    self._ValidateOptions()
    for label, path in zip(["X_train", "Y_train", "wt_train", "X_test", "Y_test", "wt_test"],
                           [X_train, Y_train, wt_train, X_test, Y_test, wt_test]):
      setattr(self, label, DataLoader(path, batch_size=self.batch_size) if path is not None else None)
    self.model = None
    self.optimizer = None
    self.loss_history = {"train": [], "test": [], "learning_rate": []}

  def _ValidateOptions(self):
    if self.batch_size < 1 or self.epochs < 0 or self.num_coupling_layers < 1:
      raise ValueError("batch_size and num_coupling_layers must be positive; epochs must be nonnegative")
    if self.l2_lambda < 0 or not np.isfinite(self.l2_lambda):
      raise ValueError("l2_lambda must be finite and nonnegative")
    if self.shuffle_training and (self.shuffle_buffer_size < self.batch_size or self.shuffle_buffer_size % self.batch_size):
      raise ValueError("shuffle_buffer_size must be a positive multiple of batch_size")
    if self.permutation not in ["random", "reverse", "none"]:
      raise ValueError("permutation must be random, reverse or none")
    if self.condition_dim is not None and (not isinstance(self.condition_dim, int) or self.condition_dim < 1):
      raise ValueError("condition_dim must be a positive integer or null")
    if self.early_stopping and (self.patience < 1 or self.tolerance < 0):
      raise ValueError("Early stopping requires positive patience and nonnegative tolerance")

  def BuildModel(self):
    """Build couplings and compile dynamic-batch inference functions."""
    tf.keras.utils.set_random_seed(self.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
      try:
        tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError:
        pass # Device already initialized by another repository model.
    self.x_columns = self.data_parameters.get("X_columns", self.X_train.columns if self.X_train else [])
    self.y_columns = self.data_parameters.get("Y_columns", self.Y_train.columns if self.Y_train else [])
    dimension = len(self.x_columns)
    if not dimension:
      raise ValueError("KerasFlow requires at least one observable")
    if self.condition_dim is not None and not self.y_columns:
      raise ValueError("condition_dim requires a conditional dataset")
    defaults = {key: getattr(self, key) for key in ["hidden_units", "activation", "dropout", "residual", "scale_clip", "spline_bins", "spline_bound", "min_bin_size", "min_slope"]}
    specs = self.coupling_layers
    if specs is None:
      specs = [{"design": ("affine" if i % 2 == 0 else "spline") if self.coupling_design == "interleaved" else self.coupling_design}
               for i in range(self.num_coupling_layers)]
    if not specs:
      raise ValueError("coupling_layers cannot be empty")
    rng = np.random.default_rng(self.seed)
    couplings = []
    for spec in specs:
      settings = {**defaults, **spec}
      if settings.get("design") not in ["affine", "spline"]:
        raise ValueError("Each coupling design must be affine or spline")
      if not 0 <= settings["dropout"] < 1 or settings["scale_clip"] <= 0:
        raise ValueError("dropout must be in [0,1); scale_clip must be positive")
      if any(width < 1 for width in settings["hidden_units"]):
        raise ValueError("hidden_units must contain positive widths")
      if settings["spline_bins"] < 2 or settings["min_bin_size"] <= 0 or settings["spline_bins"] * settings["min_bin_size"] >= 2 * settings["spline_bound"] or not 0 < settings["min_slope"] < 1:
        raise ValueError("Invalid spline bins, bound, minimum bin size or slope")
      order = rng.permutation(dimension) if self.permutation == "random" else np.arange(dimension)[::-1] if self.permutation == "reverse" else np.arange(dimension)
      couplings.append(ConditionalCoupling(dimension, order, settings, self.l2_lambda))
    self.model = ConditionalFlow(dimension, couplings, self.condition_hidden_units, self.condition_dim, self.activation, self.l2_lambda)
    self.model((tf.zeros([1, dimension]), tf.zeros([1, len(self.y_columns)])))
    signature = [tf.TensorSpec([None, dimension], tf.float32), tf.TensorSpec([None, len(self.y_columns)], tf.float32)]
    # Recreate graphs on every build/load so they never retain old variables.
    self._ComputeLogProb = tf.function(self._ForwardLogProb, input_signature=signature, jit_compile=self.jit_compile)
    self.inverse_fast = tf.function(self.model.Inverse, input_signature=signature, jit_compile=self.jit_compile)
    self._ComputeGradient = tf.function(self._Gradient, reduce_retracing=True, jit_compile=self.jit_compile)
    self.optimizer = None

  def _ForwardLogProb(self, parameters, conditions):
    return self.model((parameters, conditions), training=False)

  def _Gradient(self, parameters, conditions, differentiate_parameters=False):
    target = parameters if differentiate_parameters else conditions
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(target)
      prediction = self._ForwardLogProb(parameters, conditions)
    return prediction, tape.gradient(prediction, target, unconnected_gradients=tf.UnconnectedGradients.ZERO)

  def BuildTrainer(self):
    """Use the repository optimizer and schedule with a compiled update."""
    if self.model is None:
      self.BuildModel()
    if self.X_train is None or self.X_train.num_rows == 0:
      raise ValueError("A nonempty training dataset is required")
    self.optimizer, self.lr_scheduler, self.adaptive_lr_scheduler = Optimizer().GetOptimizer(
      self.X_train.num_rows, self.batch_size, self.epochs, optimizer_name=self.optimizer_name,
      lr_scheduler_name=self.lr_scheduler_name, lr_scheduler_options=self.lr_scheduler_options,
      default_lr=self.learning_rate, gradient_clipping_norm=self.gradient_clipping_norm,
      use_ema=self.use_ema, ema_momentum=self.ema_momentum)
    self._TrainStep = tf.function(self._UpdateWeights, reduce_retracing=True, jit_compile=self.jit_compile)

  def _UpdateWeights(self, X, Y, weights):
    with tf.GradientTape() as tape:
      nll = -self.model((X, Y), training=True)
      if self.clip_negative_weight_loss:
        positive = tf.where(weights > 0, nll, tf.constant(-np.inf, tf.float32))
        cutoff = tf.reduce_max(positive)
        nll = tf.where((weights < 0) & (nll > cutoff), tf.zeros_like(nll), nll)
      total = tf.reduce_sum(weights)
      tf.debugging.assert_positive(total, "Training batch must have positive net weight")
      loss = tf.reduce_sum(nll * weights) / total
      if self.model.losses:
        loss += tf.add_n(self.model.losses)
      tf.debugging.assert_all_finite(loss, "Nonfinite flow training loss")
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients([(g, v) for g, v in zip(gradients, self.model.trainable_variables) if g is not None])
    return loss

  def _GetBatches(self, X, Y, wt, shuffle_rng=None, batch_size=None):
    """Read aligned buffers once, optionally shuffle, then yield all rows."""
    size = batch_size or self.batch_size
    buffer_size = self.shuffle_buffer_size if shuffle_rng is not None else size
    loaders = [DataLoader(path, batch_size=buffer_size) for path in [X, Y, wt]]
    if loaders[0].num_rows != loaders[2].num_rows or (loaders[1].num_columns and loaders[0].num_rows != loaders[1].num_rows):
      raise ValueError("Features, conditions and weights must have matching row counts")
    try:
      for _ in range(loaders[0].num_batches):
        values, conditions, weights = [loader.LoadNextBatch().to_numpy(np.float32) for loader in loaders]
        if not loaders[1].num_columns:
          conditions = np.empty((len(values), 0), np.float32)
        if weights.shape[1] != 1:
          raise ValueError("The weight table must have exactly one column")
        indices = shuffle_rng.permutation(len(values)) if shuffle_rng is not None else np.arange(len(values))
        for start in range(0, len(indices), size):
          selected = indices[start:start + size]
          yield values[selected], conditions[selected], weights[selected, 0]
    finally:
      for loader in loaders:
        if loader.parquet_file is not None:
          loader.parquet_file.close()

  def Loss(self, X, Y, wt):
    """Signed weighted NLL in training space, without L2 or clipping."""
    if self.model is None:
      self.BuildModel()
    numerator, denominator = 0.0, 0.0
    size = int(os.getenv("EVENTS_PER_BATCH", self.batch_size))
    for values, conditions, weights in self._GetBatches(X, Y, wt, batch_size=size):
      nll = -self._ComputeLogProb(values, conditions).numpy().astype(np.float64)
      numerator += np.sum(nll * weights.astype(np.float64))
      denominator += np.sum(weights, dtype=np.float64)
    if denominator <= 0 or not np.isfinite(numerator):
      raise ValueError("Loss requires finite predictions and positive total weight")
    return float(numerator / denominator)

  def Train(self, name="model.h5"):
    """Train, save the best evaluation checkpoint and optionally each epoch."""
    from tqdm.autonotebook import tqdm
    if self.optimizer is None:
      self.BuildTrainer()
    train_paths = [loader.parquet_file_name for loader in [self.X_train, self.Y_train, self.wt_train]]
    test_paths = [loader.parquet_file_name for loader in [self.X_test, self.Y_test, self.wt_test]] if self.X_test is not None else train_paths
    rng = np.random.default_rng(self.shuffle_seed)
    self.loss_history = {"train": [], "test": [], "learning_rate": []}
    best, patience_best, stale = np.inf, np.inf, 0
    for epoch in range(self.epochs + 1):
      if epoch:
        batches = self._GetBatches(*train_paths, shuffle_rng=rng if self.shuffle_training else None)
        for values, conditions, weights in tqdm(batches, total=self.X_train.num_batches, disable=self.disable_tqdm, desc=f"Training epoch {epoch}"):
          loss = self._TrainStep(values, conditions, weights)
          if self.adaptive_lr_scheduler is not None and getattr(self.adaptive_lr_scheduler, "update_on", "batch") == "batch":
            self.optimizer.learning_rate.assign(self.adaptive_lr_scheduler.update(float(self.optimizer.learning_rate.numpy()), float(loss)))
      saved_best = False
      with UseEMAWeights(self.optimizer, self.model.trainable_variables):
        train_loss, test_loss = self.Loss(*train_paths), self.Loss(*test_paths)
        lr = float(self.optimizer.learning_rate.numpy())
        for key, value in zip(self.loss_history, [train_loss, test_loss, lr]):
          self.loss_history[key].append(value)
        print(f"Epoch {epoch}: train_loss={train_loss:.8f}, test_loss={test_loss:.8f}, learning_rate={lr:.6g}")
        if test_loss < best:
          best = test_loss
          self.Save(name)
          saved_best = True
        if self.save_model_per_epoch:
          self.Save(name.replace(".h5", f"_epoch_{epoch}.h5"))
      if self.use_ema:
        if saved_best:
          self.Save(RawDensityModelPath(name))
        if self.save_model_per_epoch:
          self.Save(RawDensityModelPath(name.replace(".h5", f"_epoch_{epoch}.h5")))
      if self.use_wandb:
        import wandb
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": test_loss, "lr": lr})
      if self.adaptive_lr_scheduler is not None and getattr(self.adaptive_lr_scheduler, "update_on", "batch") == "epoch":
        self.optimizer.learning_rate.assign(self.adaptive_lr_scheduler.update(lr, test_loss))
      if test_loss < patience_best - self.tolerance:
        patience_best, stale = test_loss, 0
      elif epoch >= self.wait_till:
        stale += 1
      if self.early_stopping and epoch >= self.wait_till and stale >= self.patience:
        break
    if self.plot_loss or self.plot_lr:
      from plotting import plot_histograms
      for enabled, filename, values, labels, ylabel, first in [
          (self.plot_loss, "loss", [self.loss_history["train"], self.loss_history["test"]], ["Train", "Test"], "Loss", 0),
          (self.plot_loss, "loss_no_zero", [self.loss_history["train"][1:], self.loss_history["test"][1:]], ["Train", "Test"], "Loss", 1),
          (self.plot_lr, "learning_rate", [self.loss_history["learning_rate"]], [self.lr_scheduler_name], "Learning Rate", 0)]:
        if enabled and values[0]:
          plot_histograms(range(first, first + len(values[0])), values, labels, title_right="", name=f"{self.plot_dir}/{filename}", x_label="Epochs", y_label=ylabel)

  def Save(self, name="model.h5"):
    """Save native Keras weights using the repository checkpoint filename."""
    MakeDirectories(name)
    self.model.save_weights(name)

  def Load(self, name="model.h5", seed=42):
    """Rebuild the configured topology and load its weights."""
    self.BuildModel()
    self.model.load_weights(name)

  def Probability(self, X, Y, return_log_prob=True, transform_X=True, transform_Y=True, no_fix=False,
                  order=0, column_1=None, column_2=None, grad_of="direct_conditions"):
    """Physical density and its first/second log-density derivatives."""
    if self.model is None:
      self.BuildModel()
    values = tf.convert_to_tensor(self._PrepareProbabilityX(X[self.x_columns], transform_X), tf.float32)
    conditions = tf.convert_to_tensor(self.PrepareY(X, Y[self.y_columns], transform_Y).to_numpy(np.float32))
    if len(values) != len(conditions):
      raise ValueError("One condition row or one row per observable is required")
    orders = order if isinstance(order, list) else [order]
    if any(item not in [0, 1, 2] for item in orders):
      raise ValueError("order must contain only 0, 1 and 2")
    if grad_of not in ["parameters", "direct_conditions", "summary_conditions"]:
      raise ValueError("grad_of must be parameters or direct_conditions")
    columns = self.x_columns if grad_of == "parameters" else self.y_columns
    first = [column_1] if isinstance(column_1, str) else columns if column_1 is None else column_1
    second = [column_2] if isinstance(column_2, str) else first if column_2 is None else column_2
    if 2 in orders and (len(first) != 1 or len(second) != 1):
      raise ValueError("Second derivatives require one column_1 and one column_2")
    target = values if grad_of == "parameters" else conditions
    if 2 in orders:
      with tf.GradientTape(watch_accessed_variables=False) as outer:
        outer.watch(target)
        with tf.GradientTape(watch_accessed_variables=False) as inner:
          inner.watch(target)
          predictions = self._ForwardLogProb(values, conditions)
        gradient = inner.gradient(predictions, target, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        selected = gradient[:, columns.index(first[0])] if first[0] in columns else tf.zeros(tf.shape(values)[0])
      hessian_row = outer.gradient(selected, target, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    elif 1 in orders:
      predictions, gradient = self._ComputeGradient(values, conditions, grad_of == "parameters")
    else:
      predictions = self._ComputeLogProb(values, conditions)
    output = []
    for derivative_order in orders:
      if derivative_order == 0:
        frame = pd.DataFrame({"log_prob": predictions.numpy().astype(np.float64)})
      elif derivative_order == 1:
        array = gradient.numpy()
        frame = pd.DataFrame({f"d_log_prob_by_d_{col}": array[:, columns.index(col)] if col in columns else np.zeros(len(X)) for col in first})
      else:
        array = hessian_row.numpy()
        derivative = array[:, columns.index(second[0])] if first[0] in columns and second[0] in columns else np.zeros(len(X))
        frame = pd.DataFrame({f"d2_log_prob_by_d_{first[0]}_and_{second[0]}": derivative})
      # Density Jacobian concerns X; gradients concern the differentiated input.
      convert = transform_X if derivative_order == 0 or grad_of == "parameters" else transform_Y
      frame = self._UnTransformProbability(frame, transform_X=convert)
      array = frame.to_numpy(np.float64)
      output.append(array if return_log_prob else np.exp(array))
    return output if isinstance(order, list) else output[0]

  def Sample(self, Y, n_events, seed=None, batch_number=None, batch_size=None, transform_Y=True, transform_X=True, apply_range_filter=True):
    """Generate conditional samples, optionally in transformed training space."""
    if self.model is None:
      self.BuildModel()
    copies = len(Y) <= 1
    if not copies:
      n_events = len(Y)
    conditions = self.PrepareY(pd.DataFrame(index=range(n_events)), Y[self.y_columns], transform_Y).to_numpy(np.float32)
    if batch_number is None:
      latent = tf.random.normal([n_events, len(self.x_columns)]) if seed is None else tf.random.stateless_normal([n_events, len(self.x_columns)], [seed, 456])
    else:
      start = batch_number * (batch_size if batch_size is not None else n_events)
      master = tf.constant([0 if seed is None else seed, 456], tf.int32)
      latent = tf.vectorized_map(lambda index: tf.random.stateless_normal([len(self.x_columns)], tf.random.experimental.stateless_fold_in(master, index)), tf.range(start, start + n_events))
    synth = pd.DataFrame(self.inverse_fast(latent, conditions).numpy(), columns=self.x_columns, dtype=np.float64)
    if not transform_X:
      return synth
    functions = ["remove_outside_minmax_from_transformed", "untransform", "remove_outside_minmax_from_untransformed"] if copies and apply_range_filter else ["untransform"]
    return DataProcessor([[synth]], "dataset", options={"parameters": self.data_parameters}).GetFull(method="dataset", functions_to_apply=functions)

  def ProbabilityIntegral(self, Y, **kwargs):
    """The full, untruncated flow density is normalized analytically."""
    return 1.0


  def _PrepareProbabilityX(self, X, transform_X=True):

    if not self.cache_observable_transforms or not transform_X:
      return self.PrepareX(X, transform_X=transform_X).to_numpy(np.float32)

    if self.max_cached_transform_batches < 1:
      raise ValueError("max_cached_transform_batches must be at least 1")
    metadata_key = hashlib.blake2b(pickle.dumps(self.data_parameters), digest_size=16).digest()
    if metadata_key != self.transform_metadata_key:
      self.prepared_X_cache.clear()
      self.probability_transform_cache.clear()
      self.transform_metadata_key = metadata_key
    # Content-based keys remain valid for freshly read batches, changed
    # selections, reordered rows, and integration samples at different masses.
    values = np.ascontiguousarray(X.to_numpy())
    key = (values.shape, values.dtype.str, hashlib.blake2b(values.tobytes(), digest_size=16).digest())
    if key not in self.prepared_X_cache:
      prepared = self.PrepareX(X, transform_X=True).to_numpy(np.float32)
      prepared.setflags(write=False)
      self.prepared_X_cache[key] = prepared
      while len(self.prepared_X_cache) > self.max_cached_transform_batches:
        self.prepared_X_cache.popitem(last=False)
    else:
      self.prepared_X_cache.move_to_end(key)
    return self.prepared_X_cache[key]


  def _UnTransformProbability(self, log_probs, transform_X=True):

    columns = list(log_probs.columns)
    if self.cache_observable_transforms and transform_X:
      result = log_probs.copy()
      for column in columns:
        if column not in self.probability_transform_cache:
          # Preserve the existing physical-density and derivative conversion.
          # These output-only conversions depend on preprocessing parameters.
          dp = DataProcessor([[pd.DataFrame({column: [0., 1.]})]], "dataset", options={"parameters": self.data_parameters})
          converted = dp.GetFull(method="dataset", functions_to_apply=["untransform"])[column].to_numpy()
          self.probability_transform_cache[column] = (converted[0], converted[1])
        zero, one = self.probability_transform_cache[column]
        if column == "log_prob":
          result[column] += zero
        else:
          result[column] *= one
      return result

    dp = DataProcessor([[log_probs]], "dataset", options={"parameters": self.data_parameters})
    return dp.GetFull(method="dataset", functions_to_apply=["untransform"] if transform_X else []).loc[:, columns]


  def PrepareX(self, X, transform_X=True):

    # Transform X
    X_dp = DataProcessor(
      [[X]],
      "dataset",
      options = {
        "parameters" : self.data_parameters,
      }
    )
    X = X_dp.GetFull(
      method="dataset",
      functions_to_apply = ["transform"] if transform_X else []
    )
    return X[self.data_parameters["X_columns"]]


  def PrepareY(self, X, Y, transform_Y=True):

    # Set up Y correctly
    if len(Y) == 1:
      Y = pd.DataFrame(np.tile(Y.to_numpy().flatten(), (len(X), 1)), columns=Y.columns, dtype=np.float64)
    elif len(Y) == 0:
      Y = pd.DataFrame(np.tile(np.array([]), (len(X), 1)), columns=Y.columns, dtype=np.float64)

    # Transform Y input
    Y_dp = DataProcessor(
      [[Y]],
      "dataset",
      options = {
        "parameters" : self.data_parameters,
      }
    )
    if len(Y.columns) != 0:
      Y = Y_dp.GetFull(
        method="dataset",
        functions_to_apply = ["transform"] if transform_Y else []
      )
    return Y[self.data_parameters["Y_columns"]]
