import numpy as np
import tensorflow as tf


class FullDatasetGradient():
  """Accumulate a signed classifier objective before one optimizer update."""

  def __init__(self, network, total_weight):
    if not np.isfinite(total_weight) or total_weight <= 0:
      raise ValueError("The full dataset must have a finite positive signed weight sum")
    self.network = network
    self.variables = network.model.trainable_variables
    self.total_weight = tf.constant(total_weight, dtype=tf.float64)
    self.buffers = [tf.Variable(tf.zeros(v.shape, tf.float64), trainable=False) for v in self.variables]
    self.numerator = tf.Variable(0.0, dtype=tf.float64, trainable=False)

  def _GetNumerator(self, X, y, wt, training):
    logits = self.network.model(X, training=training)
    probabilities = tf.clip_by_value(tf.nn.softmax(logits, axis=-1), 1e-7, 1-1e-7)
    labels = tf.one_hot(tf.reshape(y, [-1]), depth=self.network.num_classes, dtype=probabilities.dtype)
    per_sample_loss = -tf.reduce_sum(labels * tf.math.log(probabilities), axis=-1)
    return tf.reduce_sum(tf.cast(per_sample_loss, tf.float64) * tf.cast(tf.reshape(wt, [-1]), tf.float64))

  @tf.function(reduce_retracing=True)
  def Reset(self):
    self.numerator.assign(0.0)
    for buffer in self.buffers:
      buffer.assign(tf.zeros_like(buffer))

  @tf.function(reduce_retracing=True)
  def Accumulate(self, X, y, wt):
    # Normalize by the population, never by a chunk's possibly zero or negative sum.
    with tf.GradientTape() as tape:
      numerator = self._GetNumerator(X, y, wt, training=True)
      loss = numerator / self.total_weight
    gradients = tape.gradient(loss, self.variables)
    self.numerator.assign_add(numerator)
    for buffer, gradient in zip(self.buffers, gradients):
      if gradient is None:
        raise ValueError("Disconnected trainable parameter in full-dataset gradient")
      buffer.assign_add(tf.cast(gradient, tf.float64))

  @tf.function(reduce_retracing=True)
  def Apply(self):
    # Explicit regularization, clipping and AdamW decay act once per optimizer update.
    with tf.GradientTape() as tape:
      regularization = tf.add_n(self.network.model.losses) if self.network.model.losses else tf.constant(0.0)
    regularization_gradients = tape.gradient(regularization, self.variables)
    gradients = []
    for buffer, variable, regularization_gradient in zip(self.buffers, self.variables, regularization_gradients):
      gradient = tf.cast(buffer, variable.dtype)
      if regularization_gradient is not None:
        gradient += regularization_gradient
      tf.debugging.assert_all_finite(gradient, "Non-finite full-dataset gradient")
      gradients.append(gradient)
    self.network.optimizer.apply_gradients(zip(gradients, self.variables))
    return tf.linalg.global_norm(gradients)

  @tf.function(reduce_retracing=True)
  def EvaluateNumerator(self, X, y, wt):
    return self._GetNumerator(X, y, wt, training=False)
