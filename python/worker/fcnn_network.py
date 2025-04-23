import copy
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import wandb
import warnings

from bayesflow.helper_functions import extract_current_lr
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, losses
from tqdm.autonotebook import tqdm

from data_loader import DataLoader
from data_processor import DataProcessor
from optimizer import Optimizer
from plotting import plot_histograms
from useful_functions import MakeDirectories

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  print("INFO: Using GPUs for FCNNNetwork")
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class FCNNNetwork():
  """
  Network class for building and training a BayesFlow normalising flow.
  """
  def __init__(self, X_train=None, y_train=None, wt_train=None, X_test=None, y_test=None, wt_test=None, options={}):

    # Architecture parameters
    self.dropout = 0.05
    self.dense_layers = [128,256,128]
    self.activation = "relu"
    self.l2_lambda = 0.001

    # Training parameters
    self.early_stopping = False
    self.patience = 3
    self.epochs = 15
    self.batch_size = 2**6
    self.learning_rate = 1e-3
    self.optimizer_name = "Adam" 
    self.lr_scheduler_name = "ExponentialDecay"
    self.lr_scheduler_options = {} 
    self.gradient_clipping_norm = None

    # Other
    self.disable_tqdm = False
    self.use_wandb = False
    self.save_model_per_epoch = False
    self.only_X_columns = None

    # Running parameters
    self.plot_loss = True
    self.plot_lr = True
    self.plot_dir = "plots"

    # Data parameters
    self.data_parameters = {}

    self._SetOptions(options)

    # Data parquet files
    if X_train is not None:
      self.X_train = DataLoader(X_train, batch_size=self.batch_size)
    else:
      self.X_train = None
    if y_train is not None:
      self.y_train = DataLoader(y_train, batch_size=self.batch_size)
    else:
      self.y_train = None
    if wt_train is not None:
      self.wt_train = DataLoader(wt_train, batch_size=self.batch_size)
    else:
      self.wt_train = None
    if X_test is not None:
      self.X_test = DataLoader(X_test, batch_size=self.batch_size)
    else:
      self.X_test = None
    if y_test is not None:
      self.y_test = DataLoader(y_test, batch_size=self.batch_size)
    else:
      self.y_test = None
    if wt_test is not None:
      self.wt_test = DataLoader(wt_test, batch_size=self.batch_size)
    else:
      self.wt_test = None

  def _SetOptions(self, options):
    """
    Set options for the Network instance.

    Parameters
    ----------
    options : dict
        Dictionary of options to set.
    """
    for key, value in options.items():
      setattr(self, key, value)

  def _GetEpochLoss(self, data_type="train"):
    batch_size = int(os.getenv("EVENTS_PER_BATCH"))

    if data_type == "train":
      X_old_batch_size = copy.deepcopy(self.X_train.batch_size)
      Y_old_batch_size = copy.deepcopy(self.y_train.batch_size)
      wt_old_batch_size = copy.deepcopy(self.wt_train.batch_size)
      self.X_train.ChangeBatchSize(batch_size)
      self.y_train.ChangeBatchSize(batch_size)
      self.wt_train.ChangeBatchSize(batch_size)
    elif data_type == "test":
      X_old_batch_size = copy.deepcopy(self.X_test.batch_size)
      Y_old_batch_size = copy.deepcopy(self.y_test.batch_size)
      wt_old_batch_size = copy.deepcopy(self.wt_test.batch_size)
      self.X_test.ChangeBatchSize(batch_size)
      self.y_test.ChangeBatchSize(batch_size)
      self.wt_test.ChangeBatchSize(batch_size)

    sum_loss = 0
    sum_wts = 0
    for _ in range(self.X_train.num_batches):

      if data_type == "train":
        if self.only_X_columns is not None:
          X_data = self.X_train.LoadNextBatch().loc[:,self.only_X_columns].to_numpy()
        else:
          X_data = self.X_train.LoadNextBatch().loc[:,self.data_parameters["X_columns"]].to_numpy()
        y_data = self.y_train.LoadNextBatch().to_numpy()
        wt_data = self.wt_train.LoadNextBatch().to_numpy()
      elif data_type == "test":
        if self.only_X_columns is not None:
          X_data = self.X_test.LoadNextBatch().loc[:,self.only_X_columns].to_numpy()
        else:
          X_data = self.X_test.LoadNextBatch().loc[:,self.data_parameters["X_columns"]].to_numpy()        
        y_data = self.y_test.LoadNextBatch().to_numpy()
        wt_data = self.wt_test.LoadNextBatch().to_numpy()    

      predictions = self._GraphPredict(X_data)
      sum_loss += (tf.reduce_sum(tf.cast(wt_data, predictions.dtype))* self._MSE(predictions, y_data, wt_data))  # Compute loss
      sum_wts += tf.reduce_sum(tf.cast(wt_data, predictions.dtype))

    loss = tf.constant(sum_loss/sum_wts)

    if data_type == "train":
      self.X_train.ChangeBatchSize(X_old_batch_size)
      self.y_train.ChangeBatchSize(Y_old_batch_size)
      self.wt_train.ChangeBatchSize(wt_old_batch_size)
    elif data_type == "test":
      self.X_test.ChangeBatchSize(X_old_batch_size)
      self.y_test.ChangeBatchSize(Y_old_batch_size)
      self.wt_test.ChangeBatchSize(wt_old_batch_size)      

    return loss  


  def _FormatLossString(self, ep, it, loss, avg_dict, slope=None, lr=None, ep_str="Epoch", it_str="Iter", scalar_loss_str="Loss"):

      # Prepare info part
      disp_str = f"{ep_str}: {ep}, {it_str}: {it}"
      if type(loss) is dict:
          for k, v in loss.items():
              disp_str += f",{k}: {v.numpy():.3g}"
      else:
          disp_str += f",{scalar_loss_str}: {loss.numpy():.3g}"
      # Add running
      if avg_dict is not None:
          for k, v in avg_dict.items():
              disp_str += f",{k}: {v:.3g}"
      if slope is not None:
          disp_str += f",L.Slope: {slope:.3f}"
      if lr is not None:
          disp_str += f",LR: {lr:.2E}"
      return disp_str

  def _MSE(self, predictions, y_data, wt_data=None):
    if wt_data is None:
      return tf.reduce_mean(tf.square(y_data - predictions))
    else:
      return tf.reduce_sum(tf.square(y_data - predictions) * wt_data)/tf.reduce_sum(tf.cast(wt_data, predictions.dtype))

  def _MAE(self, predictions, y_data, wt_data=None):
    if wt_data is None:
      return tf.reduce_mean(tf.math.abs(y_data - predictions))
    else:
      return tf.reduce_sum(tf.math.abs(y_data - predictions) * wt_data)/tf.reduce_sum(tf.cast(wt_data, predictions.dtype))


  def BuildModel(self):

    if self.only_X_columns is not None:
      input_dim = len(self.only_X_columns)
    else:
      input_dim = self.X_train.num_columns
    output_dim = self.y_train.num_columns

    nn_layers = [layers.Input(shape=input_dim),]
    for layer in self.dense_layers:
      nn_layers.append(layers.Dense(layer, activation=self.activation, kernel_regularizer=regularizers.L2(self.l2_lambda)))
      nn_layers.append(layers.Dropout(self.dropout))
    nn_layers.append(layers.Dense(output_dim))

    self.model = models.Sequential(nn_layers)


  def BuildTrainer(self):

    optim = Optimizer()
    self.optimizer, _, _ = optim.GetOptimizer(
      self.X_train.num_rows,
      self.batch_size,
      self.epochs,
      optimizer_name=self.optimizer_name, 
      lr_scheduler_name=self.lr_scheduler_name,
      lr_scheduler_options=self.lr_scheduler_options,
      default_lr=self.learning_rate,
      gradient_clipping_norm=self.gradient_clipping_norm,
    )
    
    self.model.compile(
        optimizer=self.optimizer,
        loss=losses.MeanSquaredError(),
    )


  @tf.function
  def _train_step(self, X_data, y_data, wt_data):
    with tf.GradientTape() as tape:
      predictions = self.model(X_data, training=True)  # Forward pass
      loss = self._MSE(predictions, y_data, wt_data)  # Compute loss
    
    # Compute gradients and apply them
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return loss



  def Train(self, name="model.h5"):

    # Get loss before training
    epoch_train_loss = self._GetEpochLoss(data_type="train")
    epoch_test_loss = self._GetEpochLoss(data_type="test")
    print(f"Train, Epoch: 0, Loss: {epoch_train_loss: .3g}")
    print(f"Test, Epoch: 0, Loss: {epoch_test_loss: .3g}")
    self.epoch_train_losses = [epoch_train_loss]
    self.epoch_test_losses = [epoch_test_loss]
    lr = extract_current_lr(self.model.optimizer)
    lr = lr if not isinstance(lr, np.ndarray) else lr[0]
    self.epoch_lrs = [lr]
    best_epoch_test_loss = None

    # Write metrics to wandb
    if self.use_wandb:
        metrics = {
          "train_loss": self.epoch_train_losses[0],
          "val_loss": self.epoch_test_losses[0],
          "epoch": 0,
          "lr": lr,
        }
        wandb.log(metrics)

    for ep in range(1, self.epochs + 1):

      ep_losses = []

      with tqdm(total=self.X_train.num_batches, desc="Training epoch {}".format(ep), disable=self.disable_tqdm) as p_bar:

        #Loop through dataset
        for bi in range(1,self.X_train.num_batches+1):

          if self.only_X_columns is not None:
            X_data = self.X_train.LoadNextBatch().loc[:,self.only_X_columns].to_numpy().astype("float32")
          else:
            X_data = self.X_train.LoadNextBatch().loc[:,self.data_parameters["X_columns"]].to_numpy().astype("float32")

          y_data = self.y_train.LoadNextBatch().to_numpy().astype("float32")
          wt_data = self.wt_train.LoadNextBatch().to_numpy().astype("float32")

          loss = self._train_step(X_data, y_data, wt_data)

          ep_losses.append(loss.numpy())
          lr = extract_current_lr(self.model.optimizer)
          lr = lr if not isinstance(lr, np.ndarray) else lr[0]
          avg_loss = np.sum(ep_losses)/float(len(ep_losses))
          disp_str = self._FormatLossString(ep, bi, loss, {"Avg.Loss":avg_loss}, lr=lr, it_str="Batch")
          p_bar.set_postfix_str(disp_str)
          p_bar.update(1)

      epoch_train_loss = self._GetEpochLoss(data_type="train")
      epoch_test_loss = self._GetEpochLoss(data_type="test")
      print(f"Train, Epoch: {ep}, Loss: {epoch_train_loss}")
      print(f"Test, Epoch: {ep}, Loss: {epoch_test_loss}")
      #print(f"Train, Epoch: {ep}, Loss: {epoch_train_loss: .3g}")
      #print(f"Test, Epoch: {ep}, Loss: {epoch_test_loss: .3g}")

      self.epoch_train_losses.append(epoch_train_loss)
      self.epoch_test_losses.append(epoch_test_loss)
      self.epoch_lrs.append(lr)

      # Save model per epoch
      if self.save_model_per_epoch:
        self.model.save_weights(name.replace(".h5",f"_epoch_{ep}.h5"))

      # Save best model
      if (best_epoch_test_loss is None) or (epoch_test_loss < best_epoch_test_loss):
        patience_counter = 0
        best_epoch_test_loss = 1.0*epoch_test_loss
        MakeDirectories(name)
        self.model.save_weights(name)
      else:
        patience_counter += 1

      if self.use_wandb:
        metrics = {
            "train_loss": epoch_train_loss,
            "val_loss": epoch_test_loss,
            "epoch": ep,
            "lr": lr,
        }
        wandb.log(metrics)

      if self.early_stopping and patience_counter >= self.patience:
        print(f"Early stopping triggered.")
        break

    if self.plot_loss:
      plot_histograms(
        range(len(self.epoch_train_losses)),
        [self.epoch_train_losses, self.epoch_test_losses],
        ["Train", "Test"],
        title_right = "",
        name = f"{self.plot_dir}/loss",
        x_label = "Epochs",
        y_label = "Loss"
      )

      plot_histograms(
        range(1,len(self.epoch_train_losses)),
        [self.epoch_train_losses[1:], self.epoch_test_losses[1:]],
        ["Train", "Test"],
        title_right = "",
        name = f"{self.plot_dir}/loss_no_zero",
        x_label = "Epochs",
        y_label = "Loss"
      )

    if self.plot_lr:
      plot_histograms(
        range(len(self.epoch_lrs)),
        [self.epoch_lrs],
        [self.lr_scheduler_name],
        title_right = "",
        name = f"{self.plot_dir}/learning_rate",
        x_label = "Epochs",
        y_label = "Learning Rate"
      )

  def Load(self, name="model.h5"):
    self.BuildModel()
    if self.only_X_columns is not None:
      X_train_batch = self.X_train.LoadNextBatch().loc[:,self.only_X_columns].to_numpy()
    else:
      X_train_batch = self.X_train.LoadNextBatch().loc[:,self.data_parameters["X_columns"]].to_numpy()
    _ = self.model(X_train_batch)
    self.model.load_weights(name)

  @tf.function
  def _GraphPredict(self, X):
    return self.model(X, training=False)


  def Predict(self, X, transform_X=True, order=0, column_1=None, column_2=None):

    # Preprocess inputs
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
    if self.only_X_columns is not None:
      X = X.loc[:,self.only_X_columns]
    else:
      X = X.loc[:,self.data_parameters["X_columns"]]
    X = X.to_numpy()


    # Check type of gradient
    if isinstance(order,int):
      gradients = [order]
    else:
      gradients = order
    if 2 in gradients and column_1 is None and column_2 is None:
      raise ValueError("You must specifiy the columns required for the second derivative.")
    if isinstance(column_1, str):
      column_1 = [column_1]
    if isinstance(column_2, str):
      column_2 = [column_2]  
    if 2 in gradients and not (len(column_1)==1 and len(column_2)==1):
      raise ValueError("The second derivative must be done one column at a time.")

    # Check columns
    if column_1 is None:
      column_1 = self.data_parameters["X_columns"]
    if column_2 is None:
      column_2 = self.data_parameters["X_columns"]

    # Get indices
    indices_1 = [self.data_parameters["X_columns"].index(col) for col in column_1 if col in self.data_parameters["X_columns"]]
    indices_2 = [self.data_parameters["X_columns"].index(col) for col in column_2 if col in self.data_parameters["X_columns"]]
    column_to_index_1 = {col : indices_1.index(self.data_parameters["X_columns"].index(col)) for col in column_1 if col in self.data_parameters["X_columns"]}


    preds = []
    param_name = self.data_parameters["y_columns"][0]

    if order == 0 or order == [0]:

      pred = self._GraphPredict(X)
      preds += [pd.DataFrame({param_name : pred.numpy().flatten()})]

    elif order == 1 or order == [1] or order == [0,1]:

      X = tf.convert_to_tensor(X, dtype=tf.float32)
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        pred = self._GraphPredict(X)
      if len(indices_1) == 0:
        first_derivative = tf.zeros((len(X), 1), dtype=tf.float32)
      else:
        grad = tape.gradient(pred, X)
        first_derivative = tf.gather(grad, indices_1, axis=1)

      if order == [0,1]:
        preds += [pd.DataFrame({param_name : pred.numpy().flatten()})]
        del pred
        gc.collect()
      if order == 1 or order == [1] or order == [0,1]:
        first_derivative_for_all_columns = np.zeros((len(X),len(column_1)))
        for ind, col in enumerate(column_1):
          if col not in self.data_parameters["X_columns"]: continue
          first_derivative_for_all_columns[:, ind] = first_derivative.numpy()[:, column_to_index_1[col]]
        preds += [pd.DataFrame(first_derivative_for_all_columns, columns=[f"d_{param_name}_by_d_{col}" for col in column_1], dtype=np.float64)]
        del first_derivative, first_derivative_for_all_columns
        gc.collect()

    elif order == 2 or order == [2] or order == [1,2] or order == [0,1,2] or order == [0,2]:

      X = tf.convert_to_tensor(X, dtype=tf.float32)
      with tf.GradientTape() as tape_2:
        tape_2.watch(X)
        with tf.GradientTape() as tape_1:
          tape_1.watch(X)
          pred = self._GraphPredict(X)
        if len(indices_1) == 0 or len(indices_2) == 0:
          first_derivative = tf.zeros((len(X), 1), dtype=tf.float32)
        else:
          grad = tape_1.gradient(pred, X)
          first_derivative = tf.gather(grad, indices_1, axis=1)
      if len(indices_1) == 0 or len(indices_2) == 0:
        second_derivative = tf.zeros((len(X), 1), dtype=tf.float32)
      else:
        grad_of_grad = tape_2.gradient(first_derivative, X)
        second_derivative = tf.gather(grad_of_grad, indices_2, axis=1)

      # Make log_probs array
      if order == [0,1,2] or order == [0,2]:
        preds += [pd.DataFrame({param_name : pred.numpy().flatten()})]
        del pred
        gc.collect()
      if order == [0,1,2] or order == [1,2]:
        first_derivative_for_all_columns = np.zeros((len(X),len(column_1)))
        for ind, col in enumerate(column_1):
          if col not in self.data_parameters["X_columns"]: continue
          first_derivative_for_all_columns[:, ind] = first_derivative.numpy()[:, column_to_index_1[col]]
        preds += [pd.DataFrame(first_derivative_for_all_columns, columns=[f"d_{param_name}_by_d_{col}" for col in column_1], dtype=np.float64)]
        del first_derivative, first_derivative_for_all_columns
        gc.collect()
      if order == 2 or order == [2] or order == [0,1,2] or order == [0,2] or order == [1,2]:
        preds += [pd.DataFrame(second_derivative.numpy(), columns=[f"d2_{param_name}_by_d_{col1}_and_{col2}" for col1 in column_1 for col2 in column_2], dtype=np.float64)]
        del second_derivative
        gc.collect()

    # Post process prediction
    for ind in range(len(preds)):

      pred_dp = DataProcessor(
        [[preds[ind]]],
        "dataset",
        options = {
          "parameters" : self.data_parameters,
        }
      )
      preds[ind] = pred_dp.GetFull(
        method="dataset",
        functions_to_apply = ["untransform"]
      )

      preds[ind] = preds[ind].to_numpy()

    if isinstance(order, list):
      return preds
    else:
      return preds[0]

