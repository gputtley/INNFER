import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

class Optimizer():

  def __init__(self):
    pass

  def _lr_scheduler(self, name, options, default_lr, num_rows, batch_size, epochs):

    adapt_lr_scheduler = None

    if name == "ExponentialDecay":

      lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        default_lr, 
        decay_rate=options["decay_rate"] if "decay_rate" in options.keys() else 0.9,
        decay_steps=int(num_rows/batch_size)
      )    

    elif name == "ExponentialDecayWithConstant":

      class ExponentialDecayWithConstant(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_learning_rate, decay_rate, decay_steps, minimum_learning_rate):
          super(ExponentialDecayWithConstant, self).__init__()
          self.initial_learning_rate = initial_learning_rate
          self.decay_rate = decay_rate
          self.decay_steps = decay_steps
          self.minimum_learning_rate = minimum_learning_rate
        def __call__(self, step):
          learning_rate = ((self.initial_learning_rate-self.minimum_learning_rate) * (self.decay_rate**(step/self.decay_steps))) + self.minimum_learning_rate
          return learning_rate
      lr_scheduler = ExponentialDecayWithConstant(
          initial_learning_rate=default_lr,
          decay_rate=options["decay_rate"] if "decay_rate" in options.keys() else 0.9,
          decay_steps=int(np.ceil(num_rows/batch_size)),
          minimum_learning_rate=options["min_lr"] if "min_lr" in options.keys() else 0.0001
      )

    elif name == "PolynomialDecay":

      lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
          initial_learning_rate=default_lr,
          decay_steps=options["decay_steps"] if "decay_steps" in options.keys() else epochs*int(np.ceil(num_rows/batch_size)),
          end_learning_rate=options["end_lr"] if "end_lr" in options.keys() else 0.0,
          power=options["power"] if "power" in options.keys() else 1.0,
      )

    elif name == "CosineDecay":

      lr_scheduler = tf.keras.experimental.CosineDecay(
          initial_learning_rate=default_lr,
          decay_steps=options["decay_steps"] if "decay_steps" in options.keys() else epochs*int(np.ceil(num_rows/batch_size)),
          alpha=options["alpha"] if "alpha" in options.keys() else 0.0
      )

    elif name == "NestedCosineDecay":

      class NestedCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_learning_rate, outer_period, inner_period):
          super(NestedCosineDecay, self).__init__()
          self.initial_learning_rate = initial_learning_rate
          self.outer_period = outer_period
          self.inner_period = inner_period
        def __call__(self, step):
          outer_amplitude = 0.5 * (
              1 + tf.math.cos((step / self.outer_period) * np.pi)
          )
          inner_amplitude = 0.5 * (
              1 + tf.math.cos((step % self.inner_period / self.inner_period) * np.pi)
          )
          learning_rate = self.initial_learning_rate * outer_amplitude * inner_amplitude
          return learning_rate
      lr_scheduler = NestedCosineDecay(
        initial_learning_rate=default_lr,
        outer_period=options["outer_period"]*int(np.ceil(num_rows/batch_size)) if "outer_period" in options.keys() else epochs*int(np.ceil(num_rows/batch_size)),
        inner_period=options["inner_period"]*int(np.ceil(num_rows/batch_size)) if "inner_period" in options.keys() else 10*int(np.ceil(num_rows/batch_size)),
      )

    elif name == "Cyclic":

      class Cyclic(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, max_lr, min_lr, step_size):
          super(Cyclic, self).__init__()
          self.max_lr = max_lr
          self.min_lr = min_lr
          self.step_size = step_size
          self.np = np
        def __call__(self, step):
          learning_rate = (((self.max_lr-self.min_lr)/2)*(tf.math.cos(2*self.np.pi*tf.cast(step, tf.float32)/self.step_size) + 1)) + self.min_lr
          return learning_rate
      lr_scheduler = Cyclic(
        max_lr=default_lr,
        min_lr=options["min_lr"] if "min_lr" in options.keys() else 0.0,
        step_size=int(np.ceil(num_rows/batch_size))/options["cycles_per_epoch"] if "cycles_per_epoch" in options.keys() else int(np.ceil(num_rows/batch_size))/2,
      )

    elif name == "CyclicWithExponential":
      class CyclicWithExponential(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, max_lr, lr_shift, step_size_for_cycle, decay_rate, step_size_for_decay, offset):
          super(CyclicWithExponential, self).__init__()
          self.max_lr = max_lr
          self.lr_shift = lr_shift
          self.step_size_for_cycle = step_size_for_cycle
          self.decay_rate = decay_rate
          self.step_size_for_decay = step_size_for_decay
          self.offset = offset
          self.np = np
        def __call__(self, step):
          decayed_lr = ((self.max_lr-self.offset) * (self.decay_rate**(tf.cast(step, tf.float32)/float(self.step_size_for_decay)))) + self.offset
          cycled_lr = self.lr_shift * tf.math.cos(2*self.np.pi*tf.cast(step, tf.float32)/self.step_size_for_cycle)
          learning_rate = decayed_lr + cycled_lr
          return learning_rate
      lr_scheduler = CyclicWithExponential(
          max_lr = default_lr,
          lr_shift = options["lr_shift"] if "lr_shift" in options.keys() else 0.00035,
          step_size_for_cycle = int(np.ceil(num_rows/batch_size))/options["cycles_per_epoch"] if "cycles_per_epoch" in options.keys() else int(np.ceil(num_rows/batch_size))/0.5,
          decay_rate = options["decay_rate"] if "decay_rate" in options.keys() else 0.5,
          step_size_for_decay = options["decay_steps"] if "decay_steps" in options.keys() else int(np.ceil(num_rows/batch_size)),
          offset = options["offset"] if "offset" in options.keys() else 0.0004,
      )

    elif name == "AdaptiveExponential":

      lr_scheduler = default_lr
      class adaptive_lr_scheduler():
        def __init__(self, num_its=10, decay_rate=0.99):
          self.num_its = num_its
          self.decay_rate = decay_rate
          self.loss_stores = []
        def update(self, lr, loss):
          self.loss_stores.append(loss)
          if len(self.loss_stores) < self.num_its + 1:
            return lr
          else:
            rolling_ave_before = np.sum(self.loss_stores[:1])
            self.loss_stores = self.loss_stores[-1:]
            rolling_ave_after = np.sum(self.loss_stores)
            if rolling_ave_before < rolling_ave_after:
              return lr*self.decay_rate
            else:
              return lr
            
      adapt_lr_scheduler = adaptive_lr_scheduler(
        num_its=options["num_its"] if "num_its" in options.keys() else 10,
        decay_rate=options["decay_rate"] if "decay_rate" in options.keys() else 0.99,
      )

    elif name == "AdaptiveGradient":

      lr_scheduler = default_lr
      class adaptive_lr_scheduler():
        def __init__(self, num_its=60, scaling=0.001, max_lr=10**(-3), min_lr=10**(-9), max_shift_fraction=0.05, grad_change=1.0):
          self.num_its = num_its
          self.scaling = scaling
          self.max_lr = max_lr
          self.min_lr = min_lr
          self.max_shift_fraction = max_shift_fraction
          self.grad_change = grad_change
          self.loss_stores = []
        def update(self, lr, loss):
          self.loss_stores.append(loss)
          if len(self.loss_stores) <= 3*self.num_its:
            return lr
          else:
            self.loss_stores = self.loss_stores[1:]
            rolling_ave_before = np.sum(self.loss_stores[:self.num_its])
            rolling_ave_middle = np.sum(self.loss_stores[self.num_its:2*self.num_its])
            rolling_ave_after = np.sum(self.loss_stores[2*self.num_its:])
            grad_before = rolling_ave_middle-rolling_ave_before
            grad_after = rolling_ave_after-rolling_ave_middle
            shift = lr*self.scaling*((grad_after/grad_before)-self.grad_change)
            min_max_shift = max(min(shift,lr*self.max_shift_fraction),-self.max_shift_fraction*self.max_shift_fraction)
            lr = min_max_shift + lr
            return max(min(lr,self.max_lr),self.min_lr)
          
      adapt_lr_scheduler = adaptive_lr_scheduler(
        num_its=options["num_its"] if "num_its" in options.keys() else 40,
        scaling=options["scaling"] if "scaling" in options.keys() else 0.0002,
        max_lr=default_lr,
        min_lr=options["min_lr"] if "min_lr" in options.keys() else 10**(-9),
        max_shift_fraction=options["max_shift_fraction"] if "max_shift_fraction" in options.keys() else 0.01,
        grad_change=options["grad_change"] if "grad_change" in options.keys() else 0.99,
      )

    elif name == "GaussianNoise":

      lr_scheduler = default_lr
      class noise_lr_scheduler():
        def __init__(self, initial_learning_rate, std_percentage=0.0001, num_its=50):
          self.initial_learning_rate = initial_learning_rate
          self.num_its = num_its
          self.std_percentage = std_percentage
          self.loss_stores = []
          self.lr_stores = []
        def update(self, lr, loss):
          self.loss_stores.append(loss)
          self.lr_stores.append(lr)
          if len(self.lr_stores) > self.num_its:
            self.lr_stores = self.lr_stores[1:]
            if len(self.loss_stores) > self.num_its + 1:
              self.loss_stores = self.loss_stores[1:]
            loss_grads = [self.loss_stores[ind+1] - self.loss_stores[ind] for ind in range(len(self.loss_stores)-1)]

            #min_ind = loss_grads.index(max(loss_grads)) # maybe make it some weighted average instead
            #lr = self.lr_stores[min_ind]

            max_loss_grad = max(loss_grads)
            weights = [-(loss_grad-max_loss_grad) for loss_grad in loss_grads]

            #weights = [-loss_grad if loss_grad<0 else 0 for loss_grad in loss_grads]

            product = np.sum([weights[ind]*self.lr_stores[ind] for ind in range(len(weights))])
            sum_weights = np.sum(weights)
            lr = product/sum_weights

          lr += np.random.normal(loc=0.0, scale=lr*self.std_percentage)
          return lr
          
      adapt_lr_scheduler = noise_lr_scheduler(
        initial_learning_rate=default_lr,
        std_percentage=options["std_percentage"] if "std_percentage" in options.keys() else 0.05,
        num_its=options["num_its"] if "num_its" in options.keys() else 50,
      )

    elif name == "CosineAndExponential":

      class CosineWithExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_learning_rate, batches_per_epoch, cosine_decay_length, exponential_switch):
          super(CosineWithExponentialDecay, self).__init__()
          self.initial_learning_rate = tf.convert_to_tensor(initial_learning_rate, dtype=tf.float32)
          self.batches_per_epoch = tf.convert_to_tensor(batches_per_epoch, dtype=tf.float32)
          self.cosine_decay_length = cosine_decay_length * batches_per_epoch
          self.exponential_switch = exponential_switch * batches_per_epoch
        def __call__(self, step):
          step = tf.cast(step, tf.float32)
          # Calculate learning rate in cosine decay phase
          cosine_decay = self.initial_learning_rate * 0.5 * (
              1 + tf.math.cos(
                  tf.constant(np.pi) * (step % self.cosine_decay_length) / self.cosine_decay_length
              )
          )
          # Learning rate at the switch step
          lr_at_switch = self.initial_learning_rate * 0.5 * (
              1 + tf.math.cos(
                  tf.constant(np.pi) * (self.exponential_switch % self.cosine_decay_length) / self.cosine_decay_length
              )
          )
          # Gradient of cosine at switch step
          gradient_at_switch = -0.5 * tf.constant(np.pi) * tf.math.sin(
              tf.constant(np.pi) * (self.exponential_switch % self.cosine_decay_length) / self.cosine_decay_length
          )
          # Per-epoch decay factor based on gradient at the switch
          decay_factor_per_epoch = -1 / gradient_at_switch
          # Convert per-epoch decay to per-batch decay
          decay_factor_per_batch = tf.pow(
              (decay_factor_per_epoch), 1.0 / self.batches_per_epoch
          )
          # Exponential decay learning rate
          exponential_decay = lr_at_switch * tf.pow(
              decay_factor_per_batch, (step - self.exponential_switch)
          )
          # Conditional selection between cosine and exponential decay
          learning_rate = tf.where(step < self.exponential_switch, cosine_decay, exponential_decay)
          return learning_rate

      lr_scheduler = CosineWithExponentialDecay(
        initial_learning_rate=default_lr,
        batches_per_epoch=int(np.ceil(num_rows/batch_size)),
        cosine_decay_length=options["cosine_decay_length"] if "cosine_decay_length" in options.keys() else 10,
        exponential_switch=options["exponential_switch"] if "exponential_switch" in options.keys() else 5,
      )

    elif name == "HalfGaussian":
      
      class HalfGaussianDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_learning_rate, stddev):
          super(HalfGaussianDecay, self).__init__()
          self.initial_learning_rate = tf.convert_to_tensor(initial_learning_rate, dtype=tf.float32)
          self.stddev = tf.convert_to_tensor(stddev, dtype=tf.float32)
        def __call__(self, step):
          learning_rate = self.initial_learning_rate * tf.math.exp(
              -0.5 * tf.math.pow(tf.cast(step, tf.float32) / self.stddev, 2)
          )
          return learning_rate

      lr_scheduler = HalfGaussianDecay(
        initial_learning_rate=default_lr,
        stddev=options["stddev_epochs"] if "stddev_epochs" in options.keys() else epochs*int(np.ceil(num_rows/batch_size))/4, # Travels 4 standard deviations
      )

    else:
      print("ERROR: lr_schedule not valid.")

    return lr_scheduler, adapt_lr_scheduler

  def _optimizer(self, lr_scheduler, optimizer_name, gradient_clipping_norm):

    if optimizer_name == "Adam":
      optimizer = tf.keras.optimizers.Adam(lr_scheduler, clipnorm=gradient_clipping_norm)
    elif optimizer_name == "AdamW":
      optimizer = tf.keras.optimizers.AdamW(lr_scheduler, clipnorm=gradient_clipping_norm)
    elif optimizer_name == "SGD":
      optimizer = tf.keras.optimizers.SGD(lr_scheduler, clipnorm=gradient_clipping_norm)
    elif optimizer_name == "RMSprop":
      optimizer = tf.keras.optimizers.RMSprop(lr_scheduler, clipnorm=gradient_clipping_norm)
    elif optimizer_name == "Adadelta":
      optimizer = tf.keras.optimizers.Adadelta(lr_scheduler, clipnorm=gradient_clipping_norm)
    else:
      print("ERROR: optimizer not valid.")

    return optimizer
  
  def GetOptimizer(
      self, 
      num_rows,
      batch_size,
      epochs,
      optimizer_name="AdamW", 
      lr_scheduler_name="CosineDecay",
      lr_scheduler_options={},
      default_lr=0.001,
      gradient_clipping_norm=None,
      ):
    
    lr_scheduler, adapt_lr_scheduler = self._lr_scheduler(
      lr_scheduler_name,
      lr_scheduler_options,
      default_lr,
      num_rows,
      batch_size,
      epochs,
    )

    optimizer = self._optimizer(
      lr_scheduler,
      optimizer_name,
      gradient_clipping_norm,
    )

    return optimizer, lr_scheduler, adapt_lr_scheduler

