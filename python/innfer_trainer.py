import os
import random
import logging
import warnings
import copy
import wandb
import numpy as np
import bayesflow as bf
import tensorflow as tf
from bayesflow.helper_functions import backprop_step, extract_current_lr, format_loss_string, loss_to_string
from bayesflow.helper_classes import EarlyStopper
from tqdm.autonotebook import tqdm
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

seed = 42
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)

class InnferTrainer(bf.trainers.Trainer):
   """
   Adapted trainer for conditional invertible neural networks.
   """
   def train_innfer(
      self,
      X_train,
      Y_train,
      wt_train,
      X_test,
      Y_test,
      wt_test,
      epochs,
      batch_size,
      save_checkpoint=True,
      optimizer=None,
      reuse_optimizer=False,
      early_stopping=False,
      use_autograph=True,
      disable_tqdm=False,
      fix_1d_spline=False,
      use_wandb = False,
      adaptive_lr_scheduler = False,
      **kwargs,
   ):
      """
      Train the Bayesian neural network.

      Args:
         X_train (DataLoader): DataLoader for training features.
         Y_train (DataLoader): DataLoader for training target variables.
         wt_train (DataLoader): DataLoader for training weights.
         X_test (DataLoader): DataLoader for test features.
         Y_test (DataLoader): DataLoader for test target variables.
         wt_test (DataLoader): DataLoader for test weights.
         epochs (int): Number of training epochs.
         batch_size (int): Batch size for training.
         save_checkpoint (bool): Flag to save checkpoints during training.
         optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
         reuse_optimizer (bool): Flag to reuse optimizer.
         early_stopping (bool): Flag to enable early stopping.
         use_autograph (bool): Flag to use Autograph for faster execution.
         disable_tqdm (bool): Flag to disable tqdm progress bar.
         fix_1d_spline (bool): Flag to fix 1D spline if conditions are met.
         **kwargs: Additional keyword arguments.

      Returns:
         dict: Dictionary of plottable loss history.
      """

      self.fix_1d_spline = fix_1d_spline

      # Compile update function, if specified
      if use_autograph:
         _backprop_step = tf.function(backprop_step, reduce_retracing=True)
      else:
         _backprop_step = backprop_step

      self._setup_optimizer(optimizer, epochs, X_train.num_batches)
      self.loss_history.start_new_run()
      self.loss_history._total_train_loss = []
      loss = self._get_epoch_loss(X_train, Y_train, wt_train, **kwargs)
      self.loss_history._total_train_loss.append(float(loss))
      val_loss = self._get_epoch_loss(X_test, Y_test, wt_test, **kwargs)
      self.loss_history.add_val_entry(0, val_loss)

      if use_wandb:
         metrics = {
            "train_loss": loss,
            "val_loss": val_loss,
            "epoch": 0,
            "lr": self._convert_lr(extract_current_lr(self.optimizer)),
         }
         wandb.log(metrics)

      # Create early stopper, if conditions met, otherwise None returned
      early_stopper = self._config_early_stopping(early_stopping, **kwargs)

      # Loop through epochs
      for ep in range(1, epochs + 1):
         with tqdm(total=X_train.num_batches, desc="Training epoch {}".format(ep), disable=disable_tqdm) as p_bar:
            #Loop through dataset
            for bi in range(1,X_train.num_batches+1):
               # Perform one training step and obtain current loss value
               input_dict = self._load_resampled_batch(X_train, Y_train, wt_train)

               loss = self._train_step(batch_size, _backprop_step, input_dict, **kwargs)

               if adaptive_lr_scheduler is not None:
                  self.optimizer.learning_rate.assign(adaptive_lr_scheduler.update(self.optimizer.learning_rate.numpy(), float(loss)))

               self.loss_history.add_entry(ep, loss)
               avg_dict = self.loss_history.get_running_losses(ep)
               lr = self._convert_lr(extract_current_lr(self.optimizer))
               disp_str = format_loss_string(ep, bi, loss, avg_dict, lr=lr, it_str="Batch")
               p_bar.set_postfix_str(disp_str)
               p_bar.update(1)

         # Store and compute validation loss, if specified
         self._save_trainer(save_checkpoint)
         loss = self._get_epoch_loss(X_train, Y_train, wt_train, **kwargs)
         self.loss_history._total_train_loss.append(float(loss))
         val_loss = self._validation(ep, X_test, Y_test, wt_test, **kwargs)

         if use_wandb:
            metrics = {
               "train_loss": loss,
               "val_loss": val_loss,
               "epoch": ep,
               "lr": lr,
            }
            wandb.log(metrics)


         # Check early stopping, if specified
         if self._check_early_stopping(early_stopper):
               break

      # Remove optimizer reference, if not set as persistent
      if not reuse_optimizer:
         self.optimizer = None

      return self.loss_history.get_plottable()
   
   def _get_epoch_loss(self, X, Y, wt, batch_size=1000000, **kwargs):        
      """
      Helper method to compute the average epoch loss(es).

      Args:
         X (DataLoader): DataLoader for features.
         Y (DataLoader): DataLoader for target variables.
         wt (DataLoader): DataLoader for weights.
         batch_size (int): Batch size for computing average epoch loss.
         **kwargs: Additional keyword arguments.

      Returns:
         tf.Tensor: Average epoch loss.
      """
      X_old_batch_size = copy.deepcopy(X.batch_size)
      Y_old_batch_size = copy.deepcopy(Y.batch_size)
      wt_old_batch_size = copy.deepcopy(wt.batch_size)
      X.ChangeBatchSize(batch_size)
      Y.ChangeBatchSize(batch_size)
      wt.ChangeBatchSize(batch_size)
      sum_loss = 0
      for _ in range(X.num_batches):
         conf = self._load_resampled_batch(X, Y, wt)
         sum_loss += float(self.amortizer.compute_loss(conf, **kwargs.pop("net_args", {})))
      loss = tf.constant(sum_loss/X.num_batches)
      X.ChangeBatchSize(X_old_batch_size)
      Y.ChangeBatchSize(Y_old_batch_size)
      wt.ChangeBatchSize(wt_old_batch_size)
      return loss     

   def _validation(self, ep, X_test, Y_test, wt_test, **kwargs):
      """
      Helper method to compute the validation loss(es).

      Args:
         ep (int): Current epoch.
         X_test (DataLoader): DataLoader for test features.
         Y_test (DataLoader): DataLoader for test target variables.
         wt_test (DataLoader): DataLoader for test weights.
         **kwargs: Additional keyword arguments.
      """
      val_loss = self._get_epoch_loss(X_test, Y_test, wt_test, **kwargs)
      self.loss_history.add_val_entry(ep, val_loss)
      val_loss_str = loss_to_string(ep, val_loss)
      logger = logging.getLogger()
      logger.info(val_loss_str)
      return val_loss

   def _config_early_stopping(self, early_stopping, **kwargs):
      """
      Helper method to configure early stopping or warn user for.

      Args:
         early_stopping (bool): Flag to enable early stopping.
         **kwargs: Additional keyword arguments.

      Returns:
         EarlyStopper or None: EarlyStopper instance if early stopping is enabled, else None.
      """
      if early_stopping:
         early_stopper = EarlyStopper(**kwargs.pop("early_stopping_args", {}))
         return early_stopper
      return None
   
   def _load_resampled_batch(self, X, Y, wt):
      """
      Helper method to resample batches based off weights.

      Args:
         X (DataLoader): DataLoader for features.
         Y (DataLoader): DataLoader for target variables.
         wt (DataLoader): DataLoader for weights.

      Returns:
         dict: Dictionary containing resampled batch data.
      """
      X_data = X.LoadNextBatch().to_numpy()
      Y_data = Y.LoadNextBatch().to_numpy()
      wt_data = wt.LoadNextBatch().to_numpy().flatten()
      if len(wt_data[wt_data < 0]) > 0:
         print("WARNING: Ignoring negative weights.")
         wt_data[wt_data < 0] = 0
      if self.fix_1d_spline:
         X_data = np.column_stack((X_data.flatten(), np.random.normal(0.0, 1.0, (len(X_data),))))
      probs = wt_data / np.sum(wt_data)
      resampled_indices = np.random.choice(len(X_data), size=len(X_data), p=probs)
      X_data = X_data[resampled_indices]
      Y_data = Y_data[resampled_indices]
      return {"parameters" : X_data, "direct_conditions" : Y_data}
      #return {"parameters" : X_data}
   
   def _convert_lr(self, lr):
      return lr if not isinstance(lr, np.ndarray) else lr[0]