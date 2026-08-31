import os
import sys
import time
import csv
import pickle
import signal
import numpy as np
import builtins


# for graphical representation of the camera
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry

import tensorflow as tf
import keras

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

from astropy import units as u

# import the different trigger stages
from triggerkit.Stages.DigitalSum import DigitalSum, DigitalSumChannelList
from triggerkit.Stages.TDSCAN import TDSCAN, getTDSCANNeighbors
# from triggerkit.Stages.DBSCAN import DBSCAN
from triggerkit.Stages.TrainSoftMaxPool2D import TrainSoftMaxPool2D
from triggerkit.Stages.MovingAverage import TemporalMovingAverage
from triggerkit.Stages.Shift import Shift
from triggerkit.Stages.ScoreQuantizer import ScoreQuantizer

from triggerkit.Stages.TrainableThreshold import TrainableThreshold
from triggerkit.Stages.OrMerge import OrMerge
from triggerkit.Stages.FADC import FADC, FADCList
from triggerkit.FileIO.FileOpenerCTAO import FileOpenerCTAO, AsyncFileOpenerProcess, SimTelTFDataset, SimTelTFDatasetConfig

from triggerkit.Loss.RateConstrainedBCE import RateConstrainedBCE

from triggerkit.Metric.NSBRateHz import NSBRateHz
from triggerkit.Metric.TauMetric import TauMetric
from triggerkit.Metric.TDSCANMetric import TDSCANMetric

from triggerkit.Callback import TDSCANController
from triggerkit.Callback.GradNormLogger import GradNormLogger


from triggerkit.Statistics.H5StatsWriter import (
    H5StatsWriter,
    PRE_THRESHOLD_AVAILABLE_ATTR,
    PRE_THRESHOLD_COMPARISON_ATTR,
    PRE_THRESHOLD_REFERENCE_ATTR,
    PRE_THRESHOLD_SCORE_DATASET,
)
from triggerkit.Statistics.metrics import roc_auc_mann_whitney

from triggerkit.Helper.Quantize import FixedPointConstraint


STATS_EVENT_FLOAT_KEYS = (
    "energy",
    "ev_time",
    "azimuth",
    "altitude",
    "h_first_int",
    "xmax",
    "xcore",
    "ycore",
    "tel_pos_x",
    "tel_pos_y",
    "tel_pos_z",
)


class TriggerChain:
  def __init__(self, simtel_path: str, simtel_nsb_path: str = None):
    self.simtel_path = simtel_path
    self.simtel_nsb_path = simtel_nsb_path
    # check if simtel_path is a list or a string
    first_path = simtel_path[0] if isinstance(simtel_path, list) else simtel_path

    # Use as a context manager so the geometry-probe file (and its underlying
    # ctapipe/h5py handle) is always released, even though we only read the
    # first event and `break` out early -- previously this leaked one open
    # file handle per TriggerChain construction (see FileOpenerCTAO's
    # __enter__/__exit__; there is no public close(), only _close()).
    with FileOpenerCTAO(first_path) as src:
        self.camera_name = src.camera_name
        self.geom = src.geom
        self.last_geom = self.geom
        self.tel_ids = src.tel_ids
        self.stages = []
        self.event_stats = []
        self.event_stats_nsb = []
        # from the geometry get the number of pixels
        self.num_pixels = self.geom.n_pixels
        # from a datacube get the number of samples
        for tel_ids_list, wf_list, wf1_list, dl0_list, dl1_list, true_image_list, pedestal_per_sample_list, event_stat_list, _ in src:
            if len(wf_list) > 0:
                # print(wf_list[0].shape)
                try:
                    self.num_samples = wf_list[0].shape[1]
                except Exception as e:
                    self.num_samples = 432
                break
    if not hasattr(self, 'num_samples'):
        raise ValueError("Could not determine number of samples from the simtel file.")
    # open the necessary file depending on the camera type to get the sampling rate
    if self.camera_name == "UNKNOWN-7987PX":
        self.sampling_rate_hz = 1e9  # 1 GHz for LST
    elif self.camera_name == "DigiCam" or self.camera_name == "DigiCam_R0Alpha":
        self.sampling_rate_hz = 250e6  # 250 MHz for sst1m
    else:
        raise ValueError(f"Unknown camera name: {self.camera_name}")
    self.window_size = (1.0 / self.sampling_rate_hz) * self.num_samples  # in seconds
    self.input_layer = tf.keras.Input(shape=(self.num_pixels, self.num_samples), dtype=tf.uint16, name="waveform")
    self.input_baseline = tf.keras.Input(shape=(self.num_pixels,), dtype=tf.int32, name="pedestal")
    self.last_layer = self.input_layer

  def add_stage(self, stage_name, **kwargs):
    if stage_name == "digital_sum":
        mode = kwargs.get('mode', 'patch7')
        neighbors = DigitalSumChannelList(self.camera_name)
        digital_sum_layer = DigitalSum(input_geometry=self.last_geom, neighbors=neighbors, mode=mode)
        self.last_layer = digital_sum_layer(self.last_layer)
        self.last_geom = digital_sum_layer.output_geometry
        return digital_sum_layer
    elif stage_name == "tdscan":
        eps_xy = kwargs.get('eps_xy', 1)
        eps_t = kwargs.get('eps_t', 1)
        filters = kwargs.get('filters', 1)
        share_neighbors = kwargs.get('share_neighbors', True)
        quantize_step = kwargs.get('quantize_step', None)
        overflow_mode = kwargs.get('overflow_mode', 'AP_SAT')
        quantization_mode = kwargs.get('quantization_mode', 'AP_TRN')
        rescale_shift = kwargs.get('rescale_shift', 0)
        pad_value = kwargs.get('pad_value', 0.0)
        fake_quant_accumulators = kwargs.get('fake_quant_accumulators', False)
        mean_penalty_lambda = kwargs.get('mean_penalty_lambda', 0.0)
        mean_penalty_target = kwargs.get('mean_penalty_target', 0.0)
        std_penalty_lambda = kwargs.get('std_penalty_lambda', 0.0)
        std_penalty_target = kwargs.get('std_penalty_target', 0.0)
        if kwargs.get('quantize', None) is not None:
            raise ValueError("TDSCAN no longer accepts 'quantize'. Use quantize_step with an explicit 'input' qspec.")
        if kwargs.get('frac_bits', None) is not None or kwargs.get('word_bits', None) is not None:
            raise ValueError("TDSCAN no longer accepts 'word_bits'/'frac_bits'. Use quantize_step with explicit qspecs.")
        initializer = kwargs.get('initializer', tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.5))
        neighbors = getTDSCANNeighbors(eps_xy=eps_xy, camera_name=self.camera_name, n_pixels=self.num_pixels)
        tdscan_layer = TDSCAN(neighbors, eps_t=eps_t, eps_xy=eps_xy, filters=filters, share_neighbors=share_neighbors, input_geometry=self.last_geom, quantize_step=quantize_step, overflow_mode=overflow_mode, quantization_mode=quantization_mode, rescale_shift=rescale_shift, pad_value=pad_value, fake_quant_accumulators=fake_quant_accumulators, mean_penalty_lambda=mean_penalty_lambda, mean_penalty_target=mean_penalty_target, std_penalty_lambda=std_penalty_lambda, std_penalty_target=std_penalty_target, initializer=initializer)
        self.last_layer = tdscan_layer(self.last_layer)
        self.last_geom = tdscan_layer.output_geometry
        return tdscan_layer
    elif stage_name == "dbscan":
        raise NotImplementedError("DBSCAN stage is not implemented yet in TriggerChain.")
    elif stage_name == "fadc":
        neighbors = FADCList()
        baseline_layer = FADC(input_geometry=self.last_geom,digi_sum_channel_list=neighbors)
        self.last_geom = baseline_layer.output_geometry
        self.last_layer = baseline_layer([self.last_layer, self.input_baseline])
        return baseline_layer
    elif stage_name == "moving_average":
        window_size = kwargs.get('window_size', 3)
        moving_avg_layer = TemporalMovingAverage(input_geometry=self.last_geom, window_size=window_size)
        self.last_layer = moving_avg_layer(self.last_layer)
        self.last_geom = moving_avg_layer.output_geometry
        return moving_avg_layer
    elif stage_name == "shift":
        value = kwargs.get('value', kwargs.get('shift', 0.0))
        quantize_step = kwargs.get('quantize_step', None)
        overflow_mode = kwargs.get('overflow_mode', 'AP_WRAP')
        quantization_mode = kwargs.get('quantization_mode', 'AP_TRN')
        shift_layer = Shift(
            input_geometry=self.last_geom,
            value=value,
            quantize_step=quantize_step,
            overflow_mode=overflow_mode,
            quantization_mode=quantization_mode,
        )
        self.last_layer = shift_layer(self.last_layer)
        self.last_geom = shift_layer.output_geometry
        return shift_layer
    elif stage_name == "score_quantizer":
        edges = kwargs.get('edges', None)
        score_quantizer_layer = ScoreQuantizer(
            input_geometry=self.last_geom,
            edges=edges,
        )
        self.last_layer = score_quantizer_layer(self.last_layer)
        self.last_geom = score_quantizer_layer.output_geometry
        return score_quantizer_layer
    elif stage_name == "threshold":
        init_tau = kwargs.get('init_tau', 10)
        temp = kwargs.get('temp', 1.0)
        binary_output = kwargs.get('binary_output', kwargs.get('binary', kwargs.get('hard', False)))
        comparison = kwargs.get('comparison', 'gt')
        inclusive = kwargs.get('inclusive', None)
        trainable_threshold_layer = TrainableThreshold(
            input_geometry=self.last_geom,
            init_tau=init_tau,
            temp=temp,
            binary_output=binary_output,
            comparison=comparison,
            inclusive=inclusive,
        )
        self.last_layer = trainable_threshold_layer(self.last_layer)
        self.last_geom = trainable_threshold_layer.output_geometry
        return trainable_threshold_layer
    elif stage_name == "global_max_pooling_2d":
        self.last_layer = tf.keras.layers.GlobalMaxPooling2D()(self.last_layer)
        return self.last_layer
    elif stage_name == "rescaling":
        scale = kwargs.get('scale', 1./255)
        self.last_layer = tf.keras.layers.Rescaling(scale)(self.last_layer)
        return self.last_layer
    elif stage_name == "train_softmax_pooling_2d":
        beta = kwargs.get('beta', 5.0)
        reduce_channels = kwargs.get('reduce_channels', False)
        train_softmax_pooling_layer = TrainSoftMaxPool2D(beta=beta, reduce_channels=reduce_channels)
        self.last_layer = train_softmax_pooling_layer(self.last_layer)
        return train_softmax_pooling_layer
    elif stage_name == "or_merge":
        # Combine several parallel branches with a (soft) logical OR. Unlike the
        # other stages this one does not extend the single ``last_layer`` cursor;
        # it consumes a list of branch outputs (the per-branch threshold layers'
        # outputs) and replaces ``last_layer`` with the merged fire signal.
        branches = kwargs.get('branches', None)
        if not branches or len(branches) < 2:
            raise ValueError("or_merge requires 'branches' with at least two branch outputs.")
        or_merge_layer = OrMerge(input_geometry=self.last_geom)
        self.last_layer = or_merge_layer(list(branches))
        # Geometry is event-level past the pooling/threshold; keep whatever the
        # branches carried (they share it) so downstream code has a value.
        self.last_geom = or_merge_layer.output_geometry
        return or_merge_layer
    else:
        raise ValueError(f"Unknown stage name: {stage_name}")

  def branch_point(self):
    """Snapshot the current build cursor so a new branch can start from here.

    Returns an opaque token capturing ``(last_layer, last_geom)``. Build one
    branch with the usual ``add_stage`` calls, then ``restore_branch_point(tok)``
    to rewind the cursor and build the next branch from the same tensor. This is
    the only supported way to build the parallel branches an ``or_merge`` later
    combines; nothing else in the chain branches.
    """
    return (self.last_layer, self.last_geom)

  def restore_branch_point(self, token):
    """Rewind the build cursor to a snapshot from ``branch_point``."""
    self.last_layer, self.last_geom = token
    return self.last_layer

    
  def compile_chain(self,
                    loss=None,
                    optimizer=None,
                    model_path:str=None
                    ):
    self.model_path = model_path
    if self.model_path is not None and os.path.exists(self.model_path):
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path, custom_objects={
            'TDSCAN': TDSCAN,
            'TrainableThreshold': TrainableThreshold,
            'FADC': FADC,
            'DigitalSum': DigitalSum,
            'TemporalMovingAverage': TemporalMovingAverage,
            'Shift': Shift,
            'ScoreQuantizer': ScoreQuantizer,
            'OrMerge': OrMerge,
            'RateConstrainedBCE': RateConstrainedBCE,
            'NSBRateHz': NSBRateHz,
            'TauMetric': TauMetric,
            'TDSCANMetric': TDSCANMetric,
            'FixedPointConstraint': FixedPointConstraint,
        })
        # Saving trained model to trained_models/trigger_chain_stats_baselinesubstractor_tdscan11_threshold0_model.keras...
        # Saving training history to trained_models/trigger_chain_stats_baselinesubstractor_tdscan11_threshold0_history.npy...
        # load the history if exists
        history_path = self.model_path.replace("_model.keras", "_history.npy")
        if os.path.exists(history_path):
            history = np.load(history_path, allow_pickle=True).item()
            self.model.history = tf.keras.callbacks.History()
            self.model.history.history = history
        self.model.summary()
        return
    # if the camera name is Digicam_R0Alpha the pedestal has already been subtracted in the input, so we can set the baseline to 0
    if self.camera_name == "DigiCam_R0Alpha":
        self.model = tf.keras.Model(
            inputs=self.input_layer,
            outputs=self.last_layer
        )
    else:    
        self.model = tf.keras.Model(
            inputs=[self.input_layer, self.input_baseline],
            outputs=self.last_layer
        )

    if loss is None:
        # Plain Binary Cross Entropy; trigger-rate tuning is handled later from
        # the saved pre-threshold score distribution.
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.0)
    
    # pass over each layer and show the weight of tdscan and trainable threhsold taking into account the quantization if enabled
    for layer in self.model.layers:
        if isinstance(layer, TDSCAN):
            print(f"Layer {layer.name} is a TDSCAN layer with quantization {layer.quantize_step}.")
            # get the weights and quantize them if quantization is enabled
            weights = layer.get_flat_weights()
            print(f"Original weights: {weights}")
        elif isinstance(layer, TrainableThreshold):
            print(f"Layer {layer.name} is a TrainableThreshold layer.")
            weights = np.array(layer.get_weights()).flatten()
            print(f"Original tau weight: {weights}")
    
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3
        )
    
    self.model.summary()
    print("Compiling model...")
    self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[tf.keras.metrics.Recall(name="gamma_recall"),
                    tf.keras.metrics.Accuracy(name="accuracy"),
            # NSBRateHz(window_size_ns=self.window_size, name="nsb_hz", hard=True),
            # TauMetric(self.model.get_layer("trainable_threshold"), name="tau"),
            ])

  def _get_last_trainable_threshold_layer(self):
    if not hasattr(self, "model") or self.model is None:
        return None
    threshold_layers = [
        layer for layer in self.model.layers
        if isinstance(layer, TrainableThreshold)
    ]
    if not threshold_layers:
        return None
    return threshold_layers[-1]

  def compile_chain_autoencoder(self, optimizer=None, loss=None, metrics=None):
    # ensure model exists
    if not hasattr(self, "model") or self.model is None:
        if self.camera_name == "DigiCam_R0Alpha":
            self.model = tf.keras.Model(
                inputs=self.input_layer,
                outputs=self.last_layer
            )
        else:
            self.model = tf.keras.Model(
                inputs=[self.input_layer, self.input_baseline],
                outputs=self.last_layer
            )

    # autoencoder loss: sum over time axis to match true_image shape
    def _collapse_pred(y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        if y_pred.shape.rank == 4:
            # (B, N, T, C) -> sum over time -> (B, N, C)
            y_img = tf.reduce_sum(y_pred, axis=2)
            # reduce channels if needed
            if y_img.shape.rank == 3:
                if y_img.shape[-1] == 1:
                    y_img = tf.squeeze(y_img, axis=-1)
                else:
                    y_img = tf.reduce_sum(y_img, axis=-1)
            return y_img
        if y_pred.shape.rank == 3:
            # (B, N, T) -> (B, N)
            return tf.reduce_sum(y_pred, axis=2)
        if y_pred.shape.rank == 2:
            return y_pred
        # fallback: flatten everything but batch
        return tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))

    def reconstruction_mse(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        if y_true_f.shape.rank > 2:
            y_true_f = tf.reshape(y_true_f, (tf.shape(y_true_f)[0], -1))
        y_pred_img = _collapse_pred(y_pred)
        return tf.reduce_mean(tf.square(y_true_f - y_pred_img), axis=-1)

    def recon_mse_metric(y_true, y_pred):
        return reconstruction_mse(y_true, y_pred)

    if loss is None:
        loss = reconstruction_mse
    if metrics is None:
        metrics = [recon_mse_metric]

    if optimizer is None:
        optimizer = (
            self.model.optimizer
            if hasattr(self.model, "optimizer") and self.model.optimizer is not None
            else tf.keras.optimizers.Adam(learning_rate=1e-3)
        )

    print("Compiling autoencoder model...")
    self.model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

  def plot_train_history(self):
    if self.model is None:
        print("Model is not trained yet.")
        return
    # check if history exist
    if not hasattr(self.model, 'history'):
        print("No training history found.")
        return
    
    history = self.model.history
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['gamma_recall'])
    plt.plot(history.history['val_gamma_recall'])
    plt.title('Model Gamma Recall')
    plt.ylabel('Gamma Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplot(2, 1, 2)
    plt.plot(history.history['nsb_hz'])
    plt.plot(history.history['val_nsb_hz'])
    plt.title('Model NSB Rate Hz (hard)')
    plt.ylabel('NSB Rate Hz')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
    # print the loss values
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()

  def train_chain(
        self,
        epochs=100,
        batch_size=4096,
        percent_validation=0.2,
        n_pe_max_train = 350,
        n_pe_max_val = 350,
        n_pe_min_train = None,
        n_pe_min_val = None,
        tel_id_only = 2,
        max_gamma_samples_train=None,
        max_gamma_samples_val=None,
        max_nsb_samples_train=None,
        max_nsb_samples_val=None,
        load_ram=True,
        output_folder="trained_models",
        base_name="trigger_chain",
        callbacks=None,
        verbose=1
    ):

    # check the folder exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cfg_train = SimTelTFDatasetConfig(
        batch_size=batch_size,
        shuffle_samples=True, # True to decorrelate batches and help generalization. 
        sample_shuffle_buffer=10000,
        seed=1337,
        load_ram=load_ram,
        interleave_files=True,
        waveform_level="r0",
        gamma_tel_id_only=tel_id_only,
        # Evaluate on everything
        gamma_n_pe_max=n_pe_max_train, # only train on n_pe with less than 350
        gamma_n_pe_min=n_pe_min_train,
        gamma_skip_if_missing_n_pe=True,
        include_event_features=True,
        event_feature_keys = (
            "n_pe", "ev_time", "energy", "azimuth", "altitude",
            "h_first_int", "xmax", "xcore", "ycore"
        ),
        nsb_skip_original_events=False,
        nsb_roll_copies=0,
        nsb_roll_axis=1,
        # Random per-batch NSB pixel-roll augmentation; a seed distinct from
        # cfg_val's so the train/val "watermark" rotations aren't correlated.
        nsb_roll_augment=True,
        nsb_roll_seed=1337,
        max_gamma_samples_total=max_gamma_samples_train,
        max_nsb_samples_total=max_nsb_samples_train,

        repeat=False,

        ignore_errors=True
    )

    cfg_val = SimTelTFDatasetConfig(
        batch_size=batch_size,
        shuffle_samples=False,
        sample_shuffle_buffer=10000,
        seed=1337,
        load_ram=load_ram,
        interleave_files=True,
        waveform_level="r0",
        gamma_tel_id_only=tel_id_only,
        # Evaluate on everything
        gamma_n_pe_max=n_pe_max_val, # only train on n_pe with less than 350
        gamma_n_pe_min=n_pe_min_val,
        gamma_skip_if_missing_n_pe=True,
        include_event_features=True,
        event_feature_keys = (
            "n_pe", "ev_time", "energy", "azimuth", "altitude",
            "h_first_int", "xmax", "xcore", "ycore"
        ),
        nsb_skip_original_events=False,
        nsb_roll_copies=0,
        nsb_roll_axis=1,
        nsb_roll_augment=True,
        nsb_roll_seed=4242,
        max_gamma_samples_total=max_gamma_samples_val,
        max_nsb_samples_total=max_nsb_samples_val,

        repeat=False,

        ignore_errors=True
    )

    percent_validation = max(0.0, min(1.0, percent_validation))
    num_files = len(self.simtel_path)
    num_val_files = int(num_files * percent_validation)

    # generate the training and validation dataset from the simtel files
    train_dataset = SimTelTFDataset(
        # take the first percent_validation of the files for validation
        gamma_files=self.simtel_path[:-num_val_files] if percent_validation > 0.0 else self.simtel_path,
        nsb_files=self.simtel_nsb_path,
        opener_cls=AsyncFileOpenerProcess,
        config=cfg_train
    )

    val_dataset = SimTelTFDataset(
        gamma_files=self.simtel_path[-num_val_files:] if percent_validation > 0.0 else [],
        nsb_files=self.simtel_nsb_path,
        opener_cls=AsyncFileOpenerProcess,
        config=cfg_val
    )

    if callbacks is None:
        callbacks = []

    # Graceful Ctrl+C: finish current epoch then stop (skip full-model save to avoid crashes).
    stop_flags = {"requested": False}
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _request_stop(signum, frame):
        if not stop_flags["requested"]:
            print("\nCtrl+C detected: will stop after current epoch and save the model...")
        stop_flags["requested"] = True

    class _GracefulStopCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if stop_flags["requested"]:
                print("Stopping requested by user; ending training after this epoch.")
                self.model.stop_training = True
    try:
        tdscan_layer: TDSCAN = self.model.get_layer("tdscan")
        tdscan_cb = TDSCANController.TDSCANController(
            tdscan_layer=tdscan_layer,
            model=self.model
        )
        callbacks.append(tdscan_cb)
    except ValueError:
        pass

    # Always keep a safe checkpoint at the end of each epoch (weights only to reduce size/fragility).
    ckpt_path = self.generate_output_filename(folder=output_folder, base_name=base_name, suffix="weights_only.weights.h5")
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        save_freq="epoch"
    )
    callbacks.append(ckpt_cb)

    callbacks.append(_GracefulStopCallback())

    train_ds = train_dataset.dataset()
    val_ds = val_dataset.dataset()
    # Match dataset inputs to model signature (waveform only vs waveform+pedestal)
    expects_baseline = True
    try:
        expects_baseline = len(self.model.inputs) == 2
    except Exception:
        pass
    if not expects_baseline:
        print("Model expects 1 input; using waveform only (ignoring pedestal).")


    def pack(features, label):
        wf  = features["waveform"]
        ped = features["pedestal"]

        # Match model dtypes (optional but recommended)
        wf  = tf.cast(wf, tf.uint16)
        ped = tf.cast(ped, tf.int32)

        # Force exactly the ranks/shapes the model expects
        wf  = tf.reshape(wf, (-1, self.num_pixels, self.num_samples))  # -> (B,1296,50)
        ped = tf.reshape(ped, (-1, self.num_pixels))                   # -> (B,1296)

        if expects_baseline:
            return (wf, ped), label
        return wf, label

    print("Loading training and validation datasets into memory...")
    train_ds_waveform = train_ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds_waveform = val_ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)
    print("Datasets loaded.")

    # add grad_norm_logger callback to log the gradient norms of each layer during training
    grad_norm_logger = GradNormLogger(sample_ds=train_ds_waveform.take(1000))
    callbacks.append(grad_norm_logger)
    # print the size of each dataset and distribution of labels
    def print_dataset_info(ds, name):
        total = 0
        gamma_count = 0
        nsb_count = 0
        for _, label in ds:
            # labels come batched; flatten to handle vector shapes safely
            lbl = np.asarray(label.numpy()).ravel()
            total += lbl.size
            gamma_count += np.count_nonzero(lbl == 1)
            nsb_count += np.count_nonzero(lbl == 0)
        print(f"{name} dataset: {total} samples, {gamma_count} gamma, {nsb_count} nsb")
    print_dataset_info(train_ds_waveform, "Training")
    print_dataset_info(val_ds_waveform, "Validation")

    print("Starting training...")
    history = None
    # Register signal handler for graceful stop.
    signal.signal(signal.SIGINT, _request_stop)
    try:
        # with tf.device(f"/GPU:{use_gpu}" if use_gpu is not None else "/CPU:0"):
        history = self.model.fit(
            train_ds_waveform,
            validation_data=val_ds_waveform,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
    except KeyboardInterrupt:
        print("Training interrupted immediately by user; proceeding to save current model state...")
        # if the model contains a tdscan layer, print the weights of the tdscan layer
        for layer in self.model.layers:
            if isinstance(layer, TDSCAN):
                layer.print_hexagonal_kernel()
                print(layer.get_flat_weights())
            if isinstance(layer, TrainableThreshold):
                tau = layer.tau.numpy()
                print(f"TrainableThreshold layer tau: {tau}")
            sys.stdout.flush()
    finally:
        # Temporarily ignore further Ctrl+C during save to prevent corruption.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Save full model only if no Ctrl+C was requested; otherwise rely on checkpoint to avoid possible crashes.
        if not stop_flags["requested"]:
            save_path_model = self.generate_output_filename(folder=output_folder, base_name=base_name, suffix="model.keras")
            print(f"Saving trained model to {save_path_model}...")
            try:
                self.model.save(save_path_model)
            except BaseException as e:
                print(f"Full model save failed with error '{e}'. A weights-only checkpoint is available at {ckpt_path}")
        else:
            print(f"Skip full-model save because Ctrl+C was requested. Latest weights are in {ckpt_path}")

        history_dict = None
        if history is not None:
            history_dict = history.history
        elif hasattr(self.model, "history") and hasattr(self.model.history, "history"):
            history_dict = self.model.history.history

        if history_dict is not None:
            save_path_history = self.generate_output_filename(folder=output_folder, base_name=base_name, suffix="history.npy")
            print(f"Saving training history to {save_path_history}...")
            np.save(save_path_history, history_dict)
        else:
            print("No training history available to save.")

        # Restore original handler after saving.
        signal.signal(signal.SIGINT, previous_sigint)
    
    # pass over each layer and if it is TDSCAN print the weights, or TrainableThreshold print tau
    for layer in self.model.layers:
        if isinstance(layer, TDSCAN):
            # Quantized-to-grid weights: the effective values used by the
            # deployed integer inference path (matches the ring_weights qspec).
            print("Quantized (grid) weights:")
            layer.print_hexagonal_kernel(quantized=True)
            print(layer.get_flat_quantized_weights())
        if isinstance(layer, TrainableThreshold):
            tau = layer.tau.numpy()
            print(f"TrainableThreshold layer tau: {tau}")
    return history
  

  ## second version to train the chain using a AutoEncoder like approach
  # in this cases, there is no threshold, we give a datacube to the model and we try to make the output the closer possible to the true_image of the event.
  # be carefull, the model return a datacube (50, 1296) but the true_image is (1296,) so we need to summ the datacube over the time axis to have the same shape as the true_image, and then we can compute the loss between the output and the true_image.
  def train_chain_autoencoder(
        self,
        epochs=100,
        batch_size=32,
        percent_validation=0.2,
        n_pe_max_train = 350,
        n_pe_max_val = 350,
        n_pe_min_train = None,
        n_pe_min_val = None,
        tel_id_only = 2,
        max_gamma_samples_train=None,
        max_gamma_samples_val=None,
        max_nsb_samples_train=None,
        max_nsb_samples_val=None,
        load_ram=True,
        output_folder="trained_models",
        base_name="trigger_chain",
        callbacks=None
    ):
    # check the folder exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # compile (if not already)
    self.compile_chain_autoencoder()

    # normalize file lists
    gamma_files = (
        [self.simtel_path]
        if isinstance(self.simtel_path, str)
        else list(self.simtel_path)
    )
    # Autoencoder training uses ONLY gamma files; NSB is intentionally ignored.
    nsb_files = []

    percent_validation = max(0.0, min(1.0, percent_validation))
    num_files = len(gamma_files)
    num_val_files = int(num_files * percent_validation)

    gamma_files_train = gamma_files[:-num_val_files] if percent_validation > 0.0 else gamma_files
    gamma_files_val = gamma_files[-num_val_files:] if percent_validation > 0.0 else []

    # --------------------------------------------------------------
    # Dataset builder for autoencoder
    # --------------------------------------------------------------
    def _normalize_waveform(arr):
        wf = np.asarray(arr)
        if wf.ndim == 3:
            wf = wf[0]
        if wf.ndim != 2:
            return None
        # handle transposed waveforms
        if wf.shape == (self.num_samples, self.num_pixels):
            wf = wf.T
        if wf.shape != (self.num_pixels, self.num_samples):
            return None
        return wf

    def _normalize_pedestal(ped):
        if ped is None:
            return np.zeros(self.num_pixels, dtype=np.int32)
        ped_arr = np.asarray(ped, dtype=np.int32)
        if ped_arr.shape != (self.num_pixels,):
            return np.zeros(self.num_pixels, dtype=np.int32)
        return ped_arr

    def _normalize_true_image(img, is_nsb=False):
        if img is None:
            if is_nsb:
                return np.zeros(self.num_pixels, dtype=np.float32)
            return None
        img_arr = np.asarray(img, dtype=np.float32).reshape(-1)
        if img_arr.shape[0] != self.num_pixels:
            if is_nsb:
                return np.zeros(self.num_pixels, dtype=np.float32)
            return None
        return img_arr

    def _passes_n_pe_filter(stats, n_pe_min, n_pe_max):
        if n_pe_min is None and n_pe_max is None:
            return True
        if stats is None:
            return False
        n_pe = stats.get("n_pe", None)
        if n_pe is None:
            return False
        if n_pe_max is not None and not (n_pe < n_pe_max):
            return False
        if n_pe_min is not None and not (n_pe > n_pe_min):
            return False
        return True

    def _iter_samples(files, label, max_samples, n_pe_min=None, n_pe_max=None):
        count = 0
        for simtel_file in files:
            with AsyncFileOpenerProcess(simtel_file) as fo:
                for (
                    tel_ids_list, wf_r0_list, wf_r1_list, _, _, true_image_list,
                    pedestal_per_sample_list, event_stat_list, _
                ) in fo:
                    if not event_stat_list:
                        continue

                    # prefer r0 if available, else r1
                    wf_list = wf_r0_list if wf_r0_list is not None and len(wf_r0_list) > 0 else wf_r1_list
                    if wf_list is None or len(wf_list) == 0:
                        continue

                    # guard against length mismatches
                    n = min(
                        len(tel_ids_list),
                        len(wf_list),
                        len(pedestal_per_sample_list),
                        len(event_stat_list),
                        len(true_image_list)
                    )
                    if n <= 0:
                        continue

                    for i in range(n):
                        tel_id = int(tel_ids_list[i])
                        if tel_id_only is not None and tel_id != int(tel_id_only):
                            continue

                        stats = event_stat_list[i] if i < len(event_stat_list) else None
                        if label == 1 and not _passes_n_pe_filter(stats, n_pe_min, n_pe_max):
                            continue

                        wf = _normalize_waveform(wf_list[i])
                        if wf is None:
                            continue

                        ped = _normalize_pedestal(pedestal_per_sample_list[i] if i < len(pedestal_per_sample_list) else None)
                        true_img = _normalize_true_image(true_image_list[i] if i < len(true_image_list) else None, is_nsb=(label == 0))
                        if true_img is None:
                            continue

                        features = {
                            "waveform": wf.astype(np.uint16),
                            "pedestal": ped.astype(np.int32),
                        }
                        yield features, true_img.astype(np.float32)

                        count += 1
                        if max_samples is not None and count >= int(max_samples):
                            return

    def _load_samples(files, label, max_samples, n_pe_min=None, n_pe_max=None):
        wf_list = []
        ped_list = []
        y_list = []
        for feats, y in _iter_samples(files, label, max_samples, n_pe_min, n_pe_max):
            wf_list.append(feats["waveform"])
            ped_list.append(feats["pedestal"])
            y_list.append(y)
        if not wf_list:
            return None
        wf_arr = np.stack(wf_list, axis=0)
        ped_arr = np.stack(ped_list, axis=0)
        y_arr = np.stack(y_list, axis=0)
        return wf_arr, ped_arr, y_arr

    def _build_dataset(gamma_files, nsb_files, max_gamma_samples, max_nsb_samples, n_pe_min, n_pe_max, shuffle, allow_empty=False):
        feat_spec = {
            "waveform": tf.TensorSpec(shape=(self.num_pixels, self.num_samples), dtype=tf.uint16),
            "pedestal": tf.TensorSpec(shape=(self.num_pixels,), dtype=tf.int32),
        }
        label_spec = tf.TensorSpec(shape=(self.num_pixels,), dtype=tf.float32)

        if load_ram:
            parts = []
            if gamma_files:
                gamma_data = _load_samples(gamma_files, 1, max_gamma_samples, n_pe_min, n_pe_max)
                if gamma_data is not None:
                    parts.append(gamma_data)
            if nsb_files:
                nsb_data = _load_samples(nsb_files, 0, max_nsb_samples, None, None)
                if nsb_data is not None:
                    parts.append(nsb_data)

            if not parts:
                if allow_empty:
                    return None
                raise ValueError("No samples loaded for autoencoder training. Check input files or filters.")

            wf_all = np.concatenate([p[0] for p in parts], axis=0)
            ped_all = np.concatenate([p[1] for p in parts], axis=0)
            y_all = np.concatenate([p[2] for p in parts], axis=0)

            ds = tf.data.Dataset.from_tensor_slices(({"waveform": wf_all, "pedestal": ped_all}, y_all))
        else:
            if not gamma_files and not nsb_files:
                if allow_empty:
                    return None
                raise ValueError("No input files provided for autoencoder training.")
            def gen():
                for item in _iter_samples(gamma_files, 1, max_gamma_samples, n_pe_min, n_pe_max):
                    yield item
                for item in _iter_samples(nsb_files, 0, max_nsb_samples, None, None):
                    yield item
            ds = tf.data.Dataset.from_generator(gen, output_signature=(feat_spec, label_spec))

        if shuffle:
            ds = ds.shuffle(10_000, reshuffle_each_iteration=True)

        ds = ds.batch(batch_size, drop_remainder=False)
        return ds.prefetch(tf.data.AUTOTUNE)

    # Match dataset inputs to model signature (waveform only vs waveform+pedestal)
    expects_baseline = True
    try:
        expects_baseline = len(self.model.inputs) == 2
    except Exception:
        pass
    if not expects_baseline:
        print("Model expects 1 input; using waveform only (ignoring pedestal).")

    def pack(features, y_true):
        wf  = tf.cast(features["waveform"], tf.uint16)
        ped = tf.cast(features["pedestal"], tf.int32)

        wf  = tf.reshape(wf, (-1, self.num_pixels, self.num_samples))
        ped = tf.reshape(ped, (-1, self.num_pixels))
        y   = tf.reshape(tf.cast(y_true, tf.float32), (-1, self.num_pixels))

        if expects_baseline:
            return (wf, ped), y
        return wf, y

    print("Preparing training and validation datasets...")
    train_ds = _build_dataset(
        gamma_files_train,
        nsb_files,
        max_gamma_samples_train,
        None,
        n_pe_min_train,
        n_pe_max_train,
        shuffle=True,
        allow_empty=False
    ).map(pack, num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = _build_dataset(
        gamma_files_val,
        nsb_files,
        max_gamma_samples_val,
        None,
        n_pe_min_val,
        n_pe_max_val,
        shuffle=False,
        allow_empty=True
    )
    if val_ds is not None:
        val_ds = val_ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)

    if callbacks is None:
        callbacks = []

    # Graceful Ctrl+C: finish current epoch then stop (skip full-model save to avoid crashes).
    stop_flags = {"requested": False}
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _request_stop(signum, frame):
        if not stop_flags["requested"]:
            print("\nCtrl+C detected: will stop after current epoch and save the model...")
        stop_flags["requested"] = True

    class _GracefulStopCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if stop_flags["requested"]:
                print("Stopping requested by user; ending training after this epoch.")
                self.model.stop_training = True

    try:
        tdscan_layer: TDSCAN = self.model.get_layer("tdscan")
        tdscan_cb = TDSCANController.TDSCANController(
            tdscan_layer=tdscan_layer,
            model=self.model
        )
        callbacks.append(tdscan_cb)
    except ValueError:
        pass

    # Always keep a safe checkpoint at the end of each epoch (weights only to reduce size/fragility).
    ckpt_path = self.generate_output_filename(folder=output_folder, base_name=base_name, suffix="weights_only.weights.h5")
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        save_freq="epoch"
    )
    callbacks.append(ckpt_cb)
    callbacks.append(_GracefulStopCallback())

    print("Starting autoencoder training...")
    history = None
    signal.signal(signal.SIGINT, _request_stop)
    try:
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    except KeyboardInterrupt:
        print("Training interrupted immediately by user; proceeding to save current model state...")
        for layer in self.model.layers:
            if isinstance(layer, TDSCAN):
                layer.print_hexagonal_kernel()
                print(layer.get_flat_weights())
            if isinstance(layer, TrainableThreshold):
                tau = layer.tau.numpy()
                print(f"TrainableThreshold layer tau: {tau}")
            sys.stdout.flush()
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if not stop_flags["requested"]:
            save_path_model = self.generate_output_filename(folder=output_folder, base_name=base_name, suffix="model.keras")
            print(f"Saving trained model to {save_path_model}...")
            try:
                self.model.save(save_path_model)
            except BaseException as e:
                print(f"Full model save failed with error '{e}'. A weights-only checkpoint is available at {ckpt_path}")
        else:
            print(f"Skip full-model save because Ctrl+C was requested. Latest weights are in {ckpt_path}")

        history_dict = None
        if history is not None:
            history_dict = history.history
        elif hasattr(self.model, "history") and hasattr(self.model.history, "history"):
            history_dict = self.model.history.history

        if history_dict is not None:
            save_path_history = self.generate_output_filename(folder=output_folder, base_name=base_name, suffix="history.npy")
            print(f"Saving training history to {save_path_history}...")
            np.save(save_path_history, history_dict)
        else:
            print("No training history available to save.")

        signal.signal(signal.SIGINT, previous_sigint)

    # Print key layer params after training
    for layer in self.model.layers:
        if isinstance(layer, TDSCAN):
            layer.print_hexagonal_kernel()
            print(layer.get_flat_weights())
        if isinstance(layer, TrainableThreshold):
            tau = layer.tau.numpy()
            print(f"TrainableThreshold layer tau: {tau}")
    return history

  # apply inferene of the model on the given simtel files
  # give the overall efficiency and nsb rate
  def compute_statistics(self, base_name="gamma", folder="",
                         batch_size=4096,
                         tel_id_only=2,
                         nsb_roll_copies=0,
                         nsb_skip_original_events=True,
                         ignore_errors=True,
                         ):

    cfg = SimTelTFDatasetConfig(
        batch_size=batch_size,
        shuffle_samples=False,
        sample_shuffle_buffer=10000,
        seed=1337,
        load_ram=False, # too much data to load in ram
        interleave_files=True,
        waveform_level="r0", 
        gamma_tel_id_only=tel_id_only,
        # Evaluate on everything
        gamma_n_pe_max=None,
        gamma_n_pe_min=None,
        gamma_skip_if_missing_n_pe=True,
        include_event_features=True,
        event_feature_keys=STATS_EVENT_FLOAT_KEYS,
        nsb_skip_original_events=nsb_skip_original_events,
        nsb_roll_copies=nsb_roll_copies,
        nsb_roll_axis=1,

        repeat=False,

        ignore_errors=ignore_errors
    )

    # Match dataset inputs to model signature (waveform only vs waveform+pedestal)
    expects_baseline = True
    try:
        expects_baseline = len(self.model.inputs) == 2
    except Exception:
        pass
    if not expects_baseline:
        print("Model expects 1 input; using waveform only (ignoring pedestal).")

    def pack(features, label):
        wf  = tf.cast(features["waveform"], tf.uint16)
        ped = tf.cast(features["pedestal"], tf.int32)

        wf  = tf.reshape(wf, (-1, self.num_pixels, self.num_samples))
        ped = tf.reshape(ped, (-1, self.num_pixels))

        y   = tf.reshape(tf.cast(label, tf.int32), (-1,))

        extra = {
            "event_id": tf.reshape(tf.cast(features["event_id"], tf.int64), (-1,)),
            "tel_id":   tf.reshape(tf.cast(features["tel_id"], tf.int32), (-1,)),
            "n_pe":     tf.reshape(tf.cast(features["n_pe"], tf.float32), (-1,)),
        }
        for key in STATS_EVENT_FLOAT_KEYS:
            extra[key] = tf.reshape(tf.cast(features[key], tf.float32), (-1,))
        if expects_baseline:
            return (wf, ped), y, extra
        return wf, y, extra
    
    

    print("Computing statistics with current model...")
    stats_dataset = SimTelTFDataset(
        gamma_files=self.simtel_path,
        nsb_files=self.simtel_nsb_path,
        opener_cls=AsyncFileOpenerProcess,
        config=cfg
    )
    # now compute the real statistics with the (possibly) adjusted threshold
    stats_ds = stats_dataset.dataset()
    ds = stats_ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # Use the last TrainableThreshold layer when several thresholds exist -- but
    # only when it is the chain's decision point. In an OR chain several branches
    # each end in a threshold and an OrMerge combines them, so no single
    # threshold's score is the trigger; we then derive the decision from the
    # model output (the OR probability) via the p>0.5 fallback below.
    is_or_chain = isinstance(self.model.layers[-1], OrMerge)
    threshold_layer = None if is_or_chain else self._get_last_trainable_threshold_layer()
    if is_or_chain:
        print("OR chain detected: trigger = OR of branches (model output > 0.5); "
              "per-branch pre-threshold scores are not logged.")
    if threshold_layer is not None:
        try:
            tau = float(threshold_layer.tau.numpy())
            print(f"Using tau={tau:.6f} from layer '{threshold_layer.name}' (trigger = p_trig>0.5)")
        except Exception:
            tau = None
    else:
        tau = None
    # try to get the TDSCAN weights
    try:
        tdscan_layer = self.model.get_layer("tdscan")
        if tdscan_layer.share_neighbors:
            kernel_rings = tdscan_layer.kernel_rings.numpy().flatten()
            print(f"TDSCAN weights (shared by ring):")
            print(f"Kernel rings: {kernel_rings}")
        else:
            kernel = tdscan_layer.kernel.numpy().flatten()
            print(f"TDSCAN weights (independent neighbors):")
            print(f"Kernel: {kernel}")
    except Exception:
        pass

    trigger_chain_info = self.generate_chain_list()

    out_h5_path = self.generate_output_filename(folder, base_name=base_name)

    # Graceful Ctrl+C: finish current batch then discard partial stats.
    stop_flags = {"requested": False, "interrupted": False}
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _request_stop(signum, frame):
        if stop_flags["requested"]:
            print("\nCtrl+C pressed again; exiting immediately.")
            raise KeyboardInterrupt
        print("\nCtrl+C detected: will stop after current batch and delete partial statistics file...")
        stop_flags["requested"] = True

    # check if the file already exists
    if os.path.exists(out_h5_path):
        # ask user to confirm overwriting
        answer = builtins.input(f"File {out_h5_path} already exists. Overwrite? (y/n): ")
        if answer.lower() != 'y':
            print("Aborting statistics computation.")
            return

    print(f"Writing statistics to {out_h5_path}...")
    print(f"Trigger chain info: {self._format_chain_for_log(trigger_chain_info)}")
    writer = H5StatsWriter(
        out_h5_path,
        trigger_chain=trigger_chain_info,
        camera_name=getattr(self, "camera_name", "unknown"),
        compression="lzf",
        chunk_rows=200_000
    )
    writer.f.attrs[PRE_THRESHOLD_AVAILABLE_ATTR] = bool(threshold_layer is not None)
    if tau is not None:
        writer.f.attrs[PRE_THRESHOLD_REFERENCE_ATTR] = float(tau)
    if threshold_layer is not None:
        writer.f.attrs[PRE_THRESHOLD_COMPARISON_ATTR] = getattr(threshold_layer, "comparison", "gt")
    signal.signal(signal.SIGINT, _request_stop)
    trig_rate = None
    gamma_total = 0
    gamma_trig = 0
    nsb_total = 0
    nsb_trig = 0
    total_events = 0

    # Live gamma-vs-NSB ROC AUC (Mann-Whitney) of the pre-threshold score: the
    # same quantity the training loss optimizes and the stats report shows. We
    # accumulate the per-class scores and recompute periodically (the exact
    # rank-based AUC is cheap enough every few batches and once at the end).
    gamma_score_chunks: list = []
    nsb_score_chunks: list = []
    roc_auc = float("nan")

    try:
        for i_batch, (inp, y, extra) in enumerate(ds):
            p = self.model(inp, training=False) 
            p = tf.reshape(tf.cast(p, tf.float32), (-1,))
            pre_threshold_score_np = None
            trigger_ref_np = None

            if threshold_layer is not None:
                score_tensor = getattr(threshold_layer, "last_score", None)
                if score_tensor is not None:
                    score_tensor = self._collapse_scores_to_event_scores(score_tensor)
                    pre_threshold_score_np = score_tensor.numpy().astype(np.float32)
                    if tau is not None:
                        if hasattr(threshold_layer, "hard_decision"):
                            trigger_ref_np = threshold_layer.hard_decision(
                                pre_threshold_score_np
                            ).numpy().astype(np.uint8)
                        else:
                            trigger_ref_np = (pre_threshold_score_np > float(tau)).astype(np.uint8)

            if trigger_ref_np is None:
                trigger_ref_np = (p.numpy() > 0.5).astype(np.uint8)

            # pull to numpy once/batch
            y_np = y.numpy().astype(np.uint8)
            
            # Update statistics
            batch_size = len(y_np)
            total_events += batch_size
            
            n_pe_np = extra["n_pe"].numpy().astype(np.float32)

            # Gamma statistics (label=1)
            gamma_mask = y_np == 1
            gamma_total += np.sum(gamma_mask)
            gamma_trig += np.sum(trigger_ref_np[gamma_mask] > 0)

            # NSB statistics (label=0)
            nsb_mask = y_np == 0
            nsb_total += np.sum(nsb_mask)
            nsb_trig += np.sum(trigger_ref_np[nsb_mask] > 0)

            # Accumulate per-class pre-threshold scores for the live ROC AUC.
            if pre_threshold_score_np is not None:
                gamma_score_chunks.append(pre_threshold_score_np[gamma_mask])
                nsb_score_chunks.append(pre_threshold_score_np[nsb_mask])

            cols = {
                "label":     y_np,                                  # 1 gamma / 0 nsb
                "event_id":  extra["event_id"].numpy().astype(np.int64),
                "tel_id":    extra["tel_id"].numpy().astype(np.int32),
                "n_pe":      n_pe_np,
                "p_trig":    p.numpy().astype(np.float32),
            }
            for key in STATS_EVENT_FLOAT_KEYS:
                cols[key] = extra[key].numpy().astype(np.float32)
            if pre_threshold_score_np is not None:
                cols[PRE_THRESHOLD_SCORE_DATASET] = pre_threshold_score_np
            writer.append(cols)
            
            # Calculate current statistics
            gamma_eff = gamma_trig / gamma_total if gamma_total > 0 else 0.0
            nsb_rate_hz = (nsb_trig / nsb_total / self.window_size) if nsb_total > 0 else 0.0
            # Recompute the exact ROC AUC periodically (cheap, and once at the end).
            if gamma_score_chunks and nsb_score_chunks and (i_batch % 20 == 0):
                roc_auc = roc_auc_mann_whitney(
                    np.concatenate(gamma_score_chunks), np.concatenate(nsb_score_chunks)
                )
            auc_txt = f"{roc_auc*100:.2f}%" if np.isfinite(roc_auc) else "NA"

            # show stat ex :
            # Processed batch 2, total events so far: 8192
            # Current gamma trig/total: 387/480 => 8.062500e-01
            # Current NSB rate (hard): 10/7642  NSB Rate => 6542.8 Hz
            print(f"Processed batch {i_batch+1}, total events so far: {total_events}")
            # show efficiency in from to to 100% with 2 decimals
            print(f"Current gamma trig/total: {gamma_trig}/{gamma_total} => {gamma_eff*100:.2f}%  | AUC(gamma vs NSB) => {auc_txt}")
            print(f"Current NSB rate (hard): {nsb_trig}/{nsb_total}  NSB Rate => {nsb_rate_hz:.1f} Hz")

            if stop_flags["requested"]:
                stop_flags["interrupted"] = True
                print("Stopping after current batch as requested; partial stats will be discarded.")
                break

        if not stop_flags["interrupted"]:
            trig_rate = writer.close(window_sec=self.window_size)
            # Exact ROC AUC over the full sample (matches the stats report).
            if gamma_score_chunks and nsb_score_chunks:
                roc_auc = roc_auc_mann_whitney(
                    np.concatenate(gamma_score_chunks), np.concatenate(nsb_score_chunks)
                )
            print(f"Wrote {out_h5_path}")
            print(f"NSB trigger rate = {trig_rate:.1f} Hz")
            if np.isfinite(roc_auc):
                print(f"Gamma vs NSB ROC AUC = {roc_auc*100:.2f}%")
            # show the trigger chain info again at the end
            print(f"Trigger chain info: {self._format_chain_for_log(trigger_chain_info)}")
    except KeyboardInterrupt:
        stop_flags["interrupted"] = True
        print("Interrupted immediately by user; partial stats will be discarded.")
    finally:
        # Ensure file handles are closed before cleanup.
        if stop_flags["interrupted"] and hasattr(writer, "close"):
            try:
                writer.close(window_sec=self.window_size)
            except Exception:
                pass
        # Restore original SIGINT handler
        signal.signal(signal.SIGINT, previous_sigint)

        # Delete partial file if run was interrupted
        if stop_flags["interrupted"] and os.path.exists(out_h5_path):
            try:
                os.remove(out_h5_path)
                print(f"Removed partial statistics file: {out_h5_path}")
            except Exception as e:
                print(f"Could not remove partial statistics file {out_h5_path}: {e}")

    if stop_flags["interrupted"]:
        print("Statistics computation aborted.")
    
    # return final statistics as a dict
    stats = {
        "gamma_total": gamma_total,
        "gamma_trig": gamma_trig,
        "gamma_efficiency": gamma_eff,
        "roc_auc": roc_auc,
        "nsb_total": nsb_total,
        "nsb_trig": nsb_trig,
        "nsb_rate_hz": nsb_rate_hz,
        "trigger_rate_hz": trig_rate
    }
    return stats

  @staticmethod
  def _format_chain_for_log(chain, weight_decimals=4, float_decimals=6):
    """Compact, readable rendering of a trigger-chain config for logging.

    The stored HDF5 keeps the full-precision weights; this only tidies the
    console output by squeezing the (L, R, 1, Cin, Cout) weight tensors down to
    their non-trivial axes and rounding floats.
    """
    def tidy(key, value):
        if key in ("ring_weights", "kernel_weights"):
            try:
                arr = np.round(np.squeeze(np.asarray(value, dtype=float)), weight_decimals)
                return arr.tolist()
            except (TypeError, ValueError):
                return value
        if isinstance(value, float):
            return round(value, float_decimals)
        return value

    pretty = []
    for stage, params in chain:
        try:
            params = {k: tidy(k, v) for k, v in dict(params).items()}
        except (TypeError, ValueError):
            pass
        pretty.append((stage, params))
    return pretty

  @staticmethod
  def _collapse_scores_to_event_scores(score):
    score = tf.cast(score, tf.float32)
    rank = score.shape.rank

    if rank is None:
        score = tf.reshape(score, (tf.shape(score)[0], -1))
        return tf.reduce_max(score, axis=1)

    if rank == 1:
        return tf.reshape(score, (-1,))

    axes = tuple(range(1, rank))
    return tf.reshape(tf.reduce_max(score, axis=axes), (-1,))

  def _pick_tau_from_empirical_scores(self, scores, desired_fraction, comparison="gt"):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return None, None, None

    desired_fraction = float(np.clip(desired_fraction, 0.0, 1.0))
    comparison = TrainableThreshold.normalize_comparison(comparison)
    unique_scores, counts = np.unique(scores, return_counts=True)
    counts = counts.astype(np.int64)
    total = int(scores.size)

    cumulative = np.cumsum(counts, dtype=np.int64)
    counts_gt = total - cumulative
    counts_ge = counts_gt + counts

    if comparison == "ge":
        tau_strict = np.nextafter(unique_scores.astype(np.float32), np.float32(np.inf))
        tau_include_ties = unique_scores.astype(np.float32)
    else:
        tau_strict = unique_scores.astype(np.float32)
        tau_include_ties = np.nextafter(unique_scores.astype(np.float32), np.float32(-np.inf))

    candidate_taus = np.concatenate([tau_strict, tau_include_ties])
    candidate_fractions = np.concatenate([
        counts_gt.astype(np.float64) / total,
        counts_ge.astype(np.float64) / total,
    ])
    candidate_modes = np.concatenate([
        np.zeros_like(counts_gt, dtype=np.uint8),
        np.ones_like(counts_ge, dtype=np.uint8),
    ])

    errors = np.abs(candidate_fractions - desired_fraction)
    best_error = float(np.min(errors))
    best_indices = np.flatnonzero(np.isclose(errors, best_error, rtol=0.0, atol=1e-12))

    # If both sides of a plateau are equally good, prefer the lower tau so the
    # final hard trigger does not systematically undershoot on quantized scores.
    best_include_ties = best_indices[candidate_modes[best_indices] == 1]
    best_idx = int(best_include_ties[0] if best_include_ties.size > 0 else best_indices[0])

    mode = "score >= bin" if candidate_modes[best_idx] == 1 else "score > bin"
    return float(candidate_taus[best_idx]), float(candidate_fractions[best_idx]), mode

  def find_threshold_for_target_rate(self, target_rate_hz, tolerance_hz=2, N_event_esimate_threshold=25_000, batch_size=4096, nsb_skip_original_events=True, nsb_roll_copies=1):
    cfg = SimTelTFDatasetConfig(
        batch_size=batch_size,
        shuffle_samples=False,
        sample_shuffle_buffer=10000,
        seed=1337,
        load_ram=False, # too much data to load in ram
        interleave_files=True,
        waveform_level="r0", 
        gamma_tel_id_only=1,
        # Evaluate on everything
        gamma_n_pe_max=None,
        gamma_n_pe_min=None,
        gamma_skip_if_missing_n_pe=True,
        include_event_features=True,
        event_feature_keys=STATS_EVENT_FLOAT_KEYS,
        nsb_skip_original_events=nsb_skip_original_events,
        nsb_roll_copies=nsb_roll_copies,
        nsb_roll_axis=1,

        repeat=False,

        ignore_errors=True
    )
    stats_dataset_nsb = SimTelTFDataset(
        gamma_files=[], # Only need NSB events to tune threshold
        nsb_files=self.simtel_nsb_path,
        opener_cls=AsyncFileOpenerProcess,
        config=cfg
    )
    expects_baseline = True
    try:
        expects_baseline = len(self.model.inputs) == 2
    except Exception:
        pass
    if not expects_baseline:
        print("Threshold tuning model expects 1 input; using waveform only (ignoring pedestal).")

    def pack(features, label):
        wf  = tf.cast(features["waveform"], tf.uint16)
        ped = tf.cast(features["pedestal"], tf.int32)

        wf  = tf.reshape(wf, (-1, self.num_pixels, self.num_samples))
        ped = tf.reshape(ped, (-1, self.num_pixels))

        y   = tf.reshape(tf.cast(label, tf.int32), (-1,))

        extra = {
            "event_id": tf.reshape(tf.cast(features["event_id"], tf.int64), (-1,)),
            "n_pe":     tf.reshape(tf.cast(features["n_pe"], tf.float32), (-1,)),
            "energy":   tf.reshape(tf.cast(features["energy"], tf.float32), (-1,)),
        }
        if expects_baseline:
            return (wf, ped), y, extra
        return wf, y, extra
    
    dataset_nsb = stats_dataset_nsb.dataset().map(pack, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    # Estimate the pre-threshold score distribution on NSB events and
    # set tau so that P(score>tau) ~= target_rate_hz * window_size.
    print(f"Adjusting trainable threshold to achieve target NSB rate of {target_rate_hz} Hz (tolerance {tolerance_hz}%)...")

    # Use the last TrainableThreshold layer so tuning matches the final
    # decision point when several thresholds exist in the network.
    threshold_layer = self._get_last_trainable_threshold_layer()

    tau_new = None
    predicted_rate = None

    if threshold_layer is None:
        print("No TrainableThreshold layer found; cannot auto-adjust threshold.")
    else:
        # Model that outputs the score right before the threshold
        pre_threshold_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=threshold_layer.input
        )

        desired_rate = float(target_rate_hz)
        desired_fraction = float(np.clip(desired_rate * self.window_size, 0.0, 1.0))
        lower_rate = desired_rate * (1.0 - tolerance_hz / 100.0)
        upper_rate = desired_rate * (1.0 + tolerance_hz / 100.0)

        nsb_scores = []
        nsb_seen = 0

        for i_batch, (inp, y, extra) in enumerate(dataset_nsb):
            # Forward pass until just before the threshold layer
            x = pre_threshold_model(inp, training=False)
            x = self._collapse_scores_to_event_scores(x)

            # Keep only NSB events (label=0)
            nsb_mask = y.numpy().astype(np.uint8) == 0
            if np.sum(nsb_mask) == 0:
                continue

            x_nsb = x.numpy()[nsb_mask]

            nsb_scores.append(x_nsb.astype(np.float32))
            nsb_seen += x_nsb.shape[0]

            if nsb_seen >= N_event_esimate_threshold:
                break
            print(f"Collected {nsb_seen} NSB samples for threshold tuning...", end="\r")

        if nsb_seen == 0:
            print("No NSB samples found to tune threshold; keeping existing tau.")
        else:
            all_scores = np.concatenate(nsb_scores)
            score_min = float(np.min(all_scores))
            score_max = float(np.max(all_scores))
            q50, q90, q99 = np.quantile(all_scores, [0.5, 0.9, 0.99])
            unique_scores, counts = np.unique(all_scores.astype(np.float32), return_counts=True)
            top_bin_score = float(unique_scores[-1])
            top_bin_fraction = float(counts[-1] / all_scores.size)

            print(
                f"Pre-threshold NSB score summary: min={score_min:.6f}, median={q50:.6f}, "
                f"p90={q90:.6f}, p99={q99:.6f}, max={score_max:.6f}, "
                f"top-bin occupancy={top_bin_fraction * 100.0:.2f}% at score {top_bin_score:.6f}."
            )
            tau_new, predicted_fraction, selection_mode = self._pick_tau_from_empirical_scores(
                all_scores,
                desired_fraction=desired_fraction,
                comparison=getattr(threshold_layer, "comparison", "gt"),
            )
            predicted_rate = predicted_fraction / self.window_size if predicted_fraction is not None else None

            print(
                f"Estimated tau={tau_new:.6f} from {nsb_seen} NSB events; "
                f"predicted NSB rate {predicted_rate:.1f} Hz using {selection_mode} "
                f"(target {desired_rate} Hz, tolerance ±{tolerance_hz}%)."
            )

            if predicted_rate < lower_rate or predicted_rate > upper_rate:
                if predicted_rate == 0.0 and top_bin_fraction > desired_fraction:
                    min_nonzero_rate = top_bin_fraction / self.window_size
                    comparison_symbol = ">=" if getattr(threshold_layer, "comparison", "gt") == "ge" else ">"
                    print(
                        f"Warning: target {desired_rate:.1f} Hz requires only "
                        f"{desired_fraction * 100.0:.2f}% of NSB windows to trigger, "
                        f"but the highest score bin ({top_bin_score:.6f}) already contains "
                        f"{top_bin_fraction * 100.0:.2f}% of the sampled NSB events. "
                        f"With a hard trigger defined by score {comparison_symbol} tau, the closest achievable "
                        f"rates are therefore 0.0 Hz or {min_nonzero_rate:.1f} Hz. "
                        "This points to a saturated or too-coarsely quantized score "
                        "distribution; increasing N_event_esimate_threshold will not fix it."
                    )
                else:
                    print(
                        f"Warning: predicted rate {predicted_rate:.1f} Hz is outside "
                        f"[{lower_rate:.1f}, {upper_rate:.1f}] Hz; consider increasing "
                        "N_event_esimate_threshold or tolerance."
                    )
    # make sure to free memory dataset_nsb
    del dataset_nsb
    del stats_dataset_nsb
    return tau_new, predicted_rate


  def _collect_branch_event_scores(self, threshold_layers, gamma_files, max_events, batch_size):
    """Per-branch event-level pre-threshold scores on NSB and gamma events.

    Builds one model that outputs every branch's pooled pre-threshold score (the
    input to its TrainableThreshold), runs it over a mixed gamma+NSB dataset, and
    returns ``(nsb, gamma)`` where each is a ``(n_events, n_branches)`` float
    array. These are what the paired-threshold tuner thresholds and OR's together
    to land the combined rate on target. Gamma scores let the tuner break ties
    among equal-rate ``(tau_i)`` pairs by gamma efficiency.
    """
    cfg = SimTelTFDatasetConfig(
        batch_size=batch_size,
        shuffle_samples=False,
        sample_shuffle_buffer=10000,
        seed=1337,
        load_ram=False,
        interleave_files=True,
        waveform_level="r0",
        gamma_tel_id_only=1,
        gamma_n_pe_max=None,
        gamma_n_pe_min=None,
        gamma_skip_if_missing_n_pe=True,
        include_event_features=True,
        event_feature_keys=STATS_EVENT_FLOAT_KEYS,
        nsb_skip_original_events=True,
        nsb_roll_copies=1,
        nsb_roll_axis=1,
        repeat=False,
        ignore_errors=True,
    )
    ds_src = SimTelTFDataset(
        gamma_files=gamma_files,
        nsb_files=self.simtel_nsb_path,
        opener_cls=AsyncFileOpenerProcess,
        config=cfg,
    )

    expects_baseline = True
    try:
        expects_baseline = len(self.model.inputs) == 2
    except Exception:
        pass

    def pack(features, label):
        wf = tf.reshape(tf.cast(features["waveform"], tf.uint16), (-1, self.num_pixels, self.num_samples))
        ped = tf.reshape(tf.cast(features["pedestal"], tf.int32), (-1, self.num_pixels))
        y = tf.reshape(tf.cast(label, tf.int32), (-1,))
        if expects_baseline:
            return (wf, ped), y
        return wf, y

    ds = ds_src.dataset().map(pack, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # One model -> the list of per-branch pre-threshold scores.
    branch_score_model = tf.keras.Model(
        inputs=self.model.inputs,
        outputs=[tl.input for tl in threshold_layers],
    )

    nsb_rows, gamma_rows = [], []
    nsb_seen = 0
    for inp, y in ds:
        outs = branch_score_model(inp, training=False)
        # Collapse each branch's output to one score per event, stack to (B, n_branches).
        cols = [self._collapse_scores_to_event_scores(o).numpy().astype(np.float32) for o in outs]
        scores = np.stack(cols, axis=1)
        y_np = y.numpy().astype(np.uint8).reshape(-1)
        nsb_rows.append(scores[y_np == 0])
        gamma_rows.append(scores[y_np == 1])
        nsb_seen += int(np.sum(y_np == 0))
        print(f"Collected {nsb_seen} NSB events for paired-threshold tuning...", end="\r")
        if max_events is not None and nsb_seen >= max_events:
            break

    nsb = np.concatenate(nsb_rows, axis=0) if nsb_rows else np.empty((0, len(threshold_layers)), np.float32)
    gamma = np.concatenate(gamma_rows, axis=0) if gamma_rows else np.empty((0, len(threshold_layers)), np.float32)
    del ds, ds_src, branch_score_model
    return nsb, gamma


  def find_paired_thresholds_for_target_rate(
        self, threshold_layers, gamma_files,
        target_rate_hz, tolerance_hz=2,
        N_event_esimate_threshold=25_000, batch_size=256):
    """Tune per-branch taus so the OR of the branches hits the target NSB rate.

    A single threshold has one tau per target rate; an OR of B branches has a
    whole surface of ``(tau_1..tau_B)`` that yield the same *combined* rate, so we
    need a policy to choose among them. This tuner:

      1. collects per-branch event-level scores on NSB (rate) and gamma (figure
         of merit) events,
      2. enumerates each branch's candidate taus (the NSB score quantiles),
      3. for the 2-branch case scores the *full* ``(tau_A, tau_B)`` grid: combined
         NSB fire fraction = mean over NSB events of ``OR_i(score_i > tau_i)``,
         combined gamma efficiency likewise on gamma events, and
      4. picks, among pairs whose combined NSB rate is within tolerance of the
         target, the one with the highest gamma efficiency (ties -> highest taus,
         i.e. the most NSB-robust corner).

    Assigns the chosen taus onto the branches' threshold layers and returns
    ``(taus, predicted_rate_hz, gamma_efficiency)``. For >2 branches it falls
    back to coordinate ascent over the same grid (cheap, near-optimal here).
    """
    if len(threshold_layers) < 2:
        raise ValueError("find_paired_thresholds_for_target_rate needs >= 2 branches.")

    nsb, gamma = self._collect_branch_event_scores(
        threshold_layers, gamma_files,
        max_events=N_event_esimate_threshold, batch_size=batch_size,
    )
    if nsb.shape[0] == 0:
        raise RuntimeError("No NSB events collected for paired-threshold tuning.")

    n_branches = nsb.shape[1]
    desired_fraction = float(np.clip(float(target_rate_hz) * self.window_size, 0.0, 1.0))

    # Candidate taus per branch: a bounded set of NSB-score quantiles, taken
    # *as-is* so the deploy-time ``score > tau`` rule spans the full rate range.
    # The q=1.0 candidate equals the max NSB score, and ``score > max`` fires on
    # nobody -> fraction 0; the q=0.0 candidate is the min score -> fraction ~1.
    # (Nudging the candidates downward, as an earlier version did, kept the top
    # event always firing and pinned the minimum achievable rate at 100%.)
    n_cand = 64
    qs = np.linspace(0.0, 1.0, n_cand)
    cand = [np.unique(np.quantile(nsb[:, i], qs).astype(np.float32)) for i in range(n_branches)]

    def or_fraction(scores, taus):
        # OR over branches of (score_i > tau_i), averaged over events.
        fired = np.zeros(scores.shape[0], dtype=bool)
        for i, t in enumerate(taus):
            fired |= scores[:, i] > t
        return float(np.mean(fired)) if scores.shape[0] else 0.0

    lower = float(target_rate_hz) * (1.0 - tolerance_hz / 100.0)
    upper = float(target_rate_hz) * (1.0 + tolerance_hz / 100.0)

    best = None  # (gamma_eff, sum_taus, taus, nsb_fraction)
    if n_branches == 2:
        # Full grid: per-event boolean fire masks, vectorized over candidate taus.
        for ta in cand[0]:
            fire_a_nsb = nsb[:, 0] > ta
            fire_a_gam = gamma[:, 0] > ta if gamma.shape[0] else None
            for tb in cand[1]:
                nsb_frac = float(np.mean(fire_a_nsb | (nsb[:, 1] > tb)))
                if not (nsb_frac > 0):
                    continue
                gam_eff = (
                    float(np.mean(fire_a_gam | (gamma[:, 1] > tb)))
                    if gamma.shape[0] else 0.0
                )
                rate = nsb_frac / self.window_size
                in_tol = lower <= rate <= upper
                # Tiered key (tuples compare element-wise, so the in_tol flag
                # dominates): among in-tolerance pairs maximize gamma efficiency
                # then prefer the higher (more NSB-robust) taus; when *no* pair is
                # in tolerance, fall back to the one closest to the target rate
                # rather than the highest gamma efficiency -- otherwise the tuner
                # would just fire on everything (rate -> max, eff -> 1).
                if in_tol:
                    cand_key = (1, gam_eff, float(ta + tb))
                else:
                    cand_key = (0, -abs(rate - float(target_rate_hz)), 0.0)
                if best is None or cand_key > best[0]:
                    best = (cand_key, [float(ta), float(tb)], nsb_frac, gam_eff)
    else:
        # Coordinate ascent: start at the per-branch tau that alone hits an equal
        # share of the rate budget, then refine one branch at a time.
        taus = []
        for i in range(n_branches):
            f_i = desired_fraction / n_branches
            taus.append(float(np.quantile(nsb[:, i], 1.0 - f_i)))
        for _ in range(3):
            for i in range(n_branches):
                best_i = None
                for t in cand[i]:
                    trial = list(taus); trial[i] = float(t)
                    nsb_frac = or_fraction(nsb, trial)
                    gam_eff = or_fraction(gamma, trial) if gamma.shape[0] else 0.0
                    err = abs(nsb_frac - desired_fraction)
                    key = (-err, gam_eff, float(t))
                    if best_i is None or key > best_i[0]:
                        best_i = (key, float(t), nsb_frac, gam_eff)
                taus[i] = best_i[1]
        nsb_frac = or_fraction(nsb, taus)
        gam_eff = or_fraction(gamma, taus) if gamma.shape[0] else 0.0
        best = (None, taus, nsb_frac, gam_eff)

    _, taus, nsb_frac, gam_eff = best
    predicted_rate = nsb_frac / self.window_size

    for tl, tau in zip(threshold_layers, taus):
        tl.tau.assign(tau)

    print(
        f"Paired thresholds: taus={[round(t, 4) for t in taus]} -> "
        f"combined NSB rate {predicted_rate:.1f} Hz (target {target_rate_hz} Hz, "
        f"±{tolerance_hz}%), gamma efficiency {gam_eff * 100.0:.2f}%."
    )
    if predicted_rate < lower or predicted_rate > upper:
        print(
            f"Warning: combined rate {predicted_rate:.1f} Hz is outside "
            f"[{lower:.1f}, {upper:.1f}] Hz; the grid may be too coarse or the OR's "
            "minimum achievable rate (both taus at max) already exceeds the target."
        )
    return taus, predicted_rate, gam_eff


  def get_layer(self, layer_name):
    return self.model.get_layer(layer_name)
  
  def search_event(
    self,
    event_id=None,
    event_index=None,
    energy_range=None,
    minnpe=None,
    maxnpe=None,
    rangenpe=None,
    rangenpe2=None, # in case we search for an event captured by two 2 different telescopes
    skip_first_n_events=-1):
    
    cpt_skipped = 0
    simtel_files = (
        [self.simtel_path]
        if isinstance(self.simtel_path, str)
        else self.simtel_path
    )
    wf_r0_list_telescope = None
    wf_r1_list_telescope = None
    dl0_list_telescope = None
    dl1_list_telescope = None
    true_image_list_telescope = None
    pedestal_per_sample_list_telescope = None
    stat_event = None

    # --------------------------------------------------------------
    # 1. Find event that matches the selection criteria
    # --------------------------------------------------------------
    for simtel_file in simtel_files:
        # with FileOpenerCTAO(simtel_file) as fo:
        with AsyncFileOpenerProcess(simtel_file) as fo:
            for tel_ids_list, wf_r0_list, wf_r1_list, dl0_list, dl1_list, true_image_list, pedestal_per_sample_list, event_stat_list, i_event in fo:
                if not event_stat_list:
                    continue

                # Treat event_id and event_index as event-level
                ev_id0 = event_stat_list[0]["event_id"]

                if event_id is not None and ev_id0 != event_id:
                    continue
                if event_index is not None and i_event != event_index:
                    continue

                # Helper: per-telescope predicate
                def _tel_passes(stats, extra_range=None):
                    n_pe = stats["n_pe"]
                    if energy_range is not None:
                        energy = stats.get("energy", None)
                        if energy is None:
                            return False
                        if energy < energy_range[0] or energy > energy_range[1]:
                            return False
                    if minnpe is not None and n_pe < minnpe:
                        return False
                    if maxnpe is not None and n_pe > maxnpe:
                        return False
                    if extra_range is not None:
                        if n_pe < extra_range[0] or n_pe > extra_range[1]:
                            return False
                    return True

                # Event-level selection
                if rangenpe is not None and rangenpe2 is not None:
                    # We want at least one telescope in rangenpe
                    # and a different telescope in rangenpe2.
                    idx1 = [
                        i
                        for i, st in enumerate(event_stat_list)
                        if _tel_passes(st, extra_range=rangenpe)
                    ]
                    idx2 = [
                        i
                        for i, st in enumerate(event_stat_list)
                        if _tel_passes(st, extra_range=rangenpe2)
                    ]
                    event_ok = any(i != j for i in idx1 for j in idx2)
                else:
                    # Old behaviour: require at least one telescope
                    # that passes min/max and (optionally) one range.
                    extra_range = rangenpe if rangenpe is not None else rangenpe2
                    event_ok = any(
                        _tel_passes(st, extra_range=extra_range)
                        for st in event_stat_list
                    )

                if not event_ok:
                    continue

                # Skip first N matching events if requested
                if cpt_skipped < skip_first_n_events:
                    cpt_skipped += 1
                    continue

                # We keep the whole event once it passes
                wf_r0_list_telescope = wf_r0_list
                wf_r1_list_telescope = wf_r1_list
                dl0_list_telescope = dl0_list
                dl1_list_telescope = dl1_list
                true_image_list_telescope = true_image_list
                pedestal_per_sample_list_telescope = pedestal_per_sample_list
                stat_event = event_stat_list
                break  # break event loop

        if wf_r0_list_telescope is not None:
            break  # break file loop
    return wf_r0_list_telescope, wf_r1_list_telescope, dl0_list_telescope, dl1_list_telescope, true_image_list_telescope, pedestal_per_sample_list_telescope, stat_event

  def show_trigger_chain(
    self,
    event_id=None,
    event_index=None,
    minnpe=None,
    maxnpe=None,
    rangenpe=None,
    rangenpe2=None, # in case we search for an event captured by two 2 different telescopes
    skip_first_n_events=-1,
    generate_image_gif=False,
    # limit_telescope=None, # to limit the number of telescope for each event, if 1, 
    cmap='inferno',
    dpi=125,
    transpose_layout=False,
    stage_name_overrides=None,
    hide_stages=None,
    hide_range_axis=False,
    show_trigger_status=True,
    show_distrib=False,
    hex_size_scale=1.0,
    telescope_id_overrides=None
):
    
    wf_list_telescope, wf_r1_list_telescope, dl0_list_telescope, dl1_list_telescope, true_image_list_telescope, pedestal_per_sample_list_telescope, stat_event = self.search_event(
        event_id=event_id,
        event_index=event_index,
        minnpe=minnpe,
        maxnpe=maxnpe,
        rangenpe=rangenpe,
        rangenpe2=rangenpe2,
        skip_first_n_events=skip_first_n_events
    )

    # --------------------------------------------------------------
    # 2. Check that we found something
    # --------------------------------------------------------------
    if wf_list_telescope is None:
        print("No event found with the given criteria.")
        return

    num_telescopes = len(dl1_list_telescope)

    # Print info per telescope
    for stats in stat_event:
        print(
            f"Telescope {stats['telescope']}: "
            f"event_id={stats['event_id']}, "
            f"n_pe={stats['n_pe']}, "
            f"energy={stats['energy']:.2f} TeV"
        )

    # --------------------------------------------------------------
    # 3. Build waveform list per stage
    #    wf_list_result[stage][tel] has shape (n_pixels, n_samples)
    # --------------------------------------------------------------
    wf_list_result = [wf_list_telescope]

    # for stage in self.stages:
    #     wf_list_next = []
    #     for i, wf in enumerate(wf_list_result[-1]):
    #         wf_out = stage.execute(wf, baseline=pedestal_per_sample_list_telescope[i])
    #         wf_list_next.append(wf_out)
    #     wf_list_result.append(wf_list_next)
    # new version using tensorflow model prediction
    # prepare input tensors

    def _pedestal_to_baseline(pedestal, n_pixels, n_samples):
        if pedestal is None:
            return np.zeros(n_pixels, dtype=np.int32)
        ped = np.asarray(pedestal)
        if ped.ndim == 1:
            if ped.shape[0] != n_pixels:
                return np.zeros(n_pixels, dtype=np.int32)
            return ped.astype(np.int32)
        if ped.ndim == 2:
            # Reduce per-sample pedestal to per-pixel baseline.
            if ped.shape == (n_pixels, n_samples) or ped.shape[0] == n_pixels:
                return np.rint(ped.mean(axis=1)).astype(np.int32)
            if ped.shape == (n_samples, n_pixels) or ped.shape[1] == n_pixels:
                return np.rint(ped.mean(axis=0)).astype(np.int32)
        return np.zeros(n_pixels, dtype=np.int32)

    def _normalize_stage_key(value):
        return str(value).lower().replace("_", "").replace("-", "").replace(" ", "")

    def _stage_keys(layer, display_name=None):
        keys = set()
        if display_name is not None:
            keys.add(_normalize_stage_key(display_name))
        if layer is None:
            keys.add("input")
            return keys
        if hasattr(layer, "stage_type"):
            stage_type = layer.stage_type()
            keys.add(_normalize_stage_key(stage_type))
            keys.add(_normalize_stage_key(stage_type.replace("_", "")))
        if hasattr(layer, "stage_name"):
            keys.add(_normalize_stage_key(layer.stage_name()))
        keys.add(_normalize_stage_key(layer.__class__.__name__))
        return keys

    def _stage_identity_keys(layer):
        return _stage_keys(layer)

    def _display_stage_name(layer, default_name):
        for key in _stage_keys(layer, default_name):
            if key in normalized_stage_name_overrides:
                return normalized_stage_name_overrides[key]
        return default_name

    normalized_stage_name_overrides = {}
    if stage_name_overrides:
        normalized_stage_name_overrides = {
            _normalize_stage_key(stage): name
            for stage, name in stage_name_overrides.items()
        }

    normalized_hide_stages = set()
    if hide_stages:
        if isinstance(hide_stages, str):
            normalized_hide_stages = {_normalize_stage_key(hide_stages)}
        else:
            normalized_hide_stages = {
                _normalize_stage_key(stage)
                for stage in hide_stages
            }

    if hex_size_scale <= 0:
        raise ValueError(f"hex_size_scale must be > 0, got {hex_size_scale}.")

    def _display_telescope_id(tel_index, stats):
        if telescope_id_overrides is None:
            return tel_index + 1
        if isinstance(telescope_id_overrides, dict):
            original_tel_id = stats.get("telescope", tel_index + 1)
            return telescope_id_overrides.get(
                original_tel_id,
                telescope_id_overrides.get(tel_index, tel_index + 1),
            )
        if isinstance(telescope_id_overrides, (list, tuple)):
            if tel_index < len(telescope_id_overrides):
                return telescope_id_overrides[tel_index]
            return tel_index + 1
        return telescope_id_overrides

    scaled_geometry_cache = {}
    def _display_geometry(geom):
        if hex_size_scale == 1.0:
            return geom
        cache_key = id(geom)
        if cache_key not in scaled_geometry_cache:
            scaled_geometry_cache[cache_key] = CameraGeometry(
                name=geom.name,
                pix_id=geom.pix_id,
                pix_x=geom.pix_x,
                pix_y=geom.pix_y,
                pix_area=geom.pix_area * (hex_size_scale ** 2),
                pix_type=geom.pix_type,
                pix_rotation=geom.pix_rotation,
                cam_rotation=geom.cam_rotation,
                neighbors=getattr(geom, "neighbors", None),
                frame=getattr(geom, "frame", None),
            )
        return scaled_geometry_cache[cache_key]

    # add batch dimension
    stage_layers = []
    for layer in self.model.layers:
        if hasattr(layer, "stage_type"):
            stage_layers.append(layer)
    
    # pass over each stage and telescope to get the output
    for layer in stage_layers:
        wf_list_next = []
        for i, wf in enumerate(wf_list_result[-1]):
            # prepare input tensors
            print(f"Processing stage: {layer.stage_type()} for telescope {i+1}")
            wf_input = tf.convert_to_tensor(wf[np.newaxis, :, :])  # add batch dimension, keep intermediate dtype
            if isinstance(layer, FADC):
                ped = None
                if pedestal_per_sample_list_telescope is not None and i < len(pedestal_per_sample_list_telescope):
                    ped = pedestal_per_sample_list_telescope[i]
                baseline = _pedestal_to_baseline(ped, wf.shape[0], wf.shape[1])
                pedestal_input = tf.convert_to_tensor(baseline[np.newaxis, :], dtype=tf.int32) # add batch dimension
                wf_out = layer([wf_input, pedestal_input], training=False)
            else:
                wf_out = layer(wf_input, training=False)
            wf_out_np = wf_out.numpy()[0] # remove batch dimension
            # if the output has a last dimension of 1, remove it
            if wf_out_np.ndim == 3 and wf_out_np.shape[2] == 1:
                wf_out_np = wf_out_np[:, :, 0]
            wf_list_next.append(wf_out_np)
        wf_list_result.append(wf_list_next)
    

    # --------------------------------------------------------------
    # 4. Prepare figure and data: wfs[stage][tel](frame, pixel)
    # --------------------------------------------------------------
    # n_stages_anim = len(self.stages) + 1  # input + stage outputs
    display_specs = []
    input_name = _display_stage_name(None, "Input")
    if not (_stage_identity_keys(None) & normalized_hide_stages):
        display_specs.append((0, None, input_name, _display_geometry(self.geom)))

    for source_index, stage_obj in enumerate(stage_layers, start=1):
        default_stage_name = (
            stage_obj.stage_name()
            if hasattr(stage_obj, "stage_name")
            else stage_obj.__class__.__name__
        )
        stage_name = _display_stage_name(stage_obj, default_stage_name)
        if _stage_identity_keys(stage_obj) & normalized_hide_stages:
            continue
        stage_geom = getattr(
            stage_obj,
            "output_geometry",
            getattr(stage_obj, "input_geometry", self.geom),
        )
        display_specs.append((source_index, stage_obj, stage_name, _display_geometry(stage_geom)))

    n_stages_anim = len(display_specs)
    if n_stages_anim == 0:
        print("No stages left to display after applying hide_stages.")
        return

    print(f"Number of stages to animate: {n_stages_anim}")
    # fig, axes = plt.subplots(
    #     num_telescopes, n_stages_total, figsize=(5 * n_stages_total, 5)
    # )

    if transpose_layout:
        nrows, ncols = n_stages_anim, num_telescopes
    else:
        nrows, ncols = num_telescopes, n_stages_anim

    if show_distrib:
        # Double each camera row: a camera panel row followed by a shorter
        # distribution (histogram) row right below it. In transpose layout the
        # stages are the rows, otherwise the telescopes are the rows.
        n_groups = n_stages_anim if transpose_layout else num_telescopes
        height_ratios = [3, 1] * n_groups  # camera taller than its histogram
        nrows = n_groups * 2
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 1.5 * sum(height_ratios)),
            gridspec_kw={"height_ratios": height_ratios},
        )
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    axes = np.asarray(axes)
    if axes.ndim == 0:
        # Single Axes -> make it 2D (1,1)
        axes = axes.reshape(1, 1)
    else:
        # 1D or 2D -> force it to (nrows, ncols)
        axes = axes.reshape(nrows, ncols)

    # Resolve the camera (and, when show_distrib, the histogram) axes for a
    # given (stage, telescope). With show_distrib each group of camera panels
    # is followed by a distribution row directly below it.
    def _camera_ax(stage_idx, tel_idx):
        if show_distrib:
            if transpose_layout:
                return axes[2 * stage_idx, tel_idx]
            return axes[2 * tel_idx, stage_idx]
        if transpose_layout:
            return axes[stage_idx, tel_idx]
        return axes[tel_idx, stage_idx]

    def _distrib_ax(stage_idx, tel_idx):
        if transpose_layout:
            return axes[2 * stage_idx + 1, tel_idx]
        return axes[2 * tel_idx + 1, stage_idx]

    # Use input waveform shape for n_frames
    try:
        n_frames = wf_list_result[0][0].shape[1] # try with r0 waveform
        n_pixels = wf_list_result[0][0].shape[0]
    except IndexError:
        n_frames = wf_r1_list_telescope[0].shape[1] # try with r1 waveform
        n_pixels = wf_r1_list_telescope[0].shape[0]

    # wfs[stage][telescope] -> shape (n_frames, n_pixels)
    wfs = []
    for source_index, _, _, _ in display_specs:
        stage_wf_list = wf_list_result[source_index]
        stage_wfs = []
        if stage_wf_list is None or len(stage_wf_list) == 0:
            for _ in range(num_telescopes):
                arr_t = np.zeros((n_frames, n_pixels), dtype=np.float32)
                stage_wfs.append(arr_t)
            wfs.append(stage_wfs)
            continue
        for wf in stage_wf_list:
            arr = np.asarray(wf, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    "Waveform must be 2D (n_pixels, n_samples), "
                    f"got shape {arr.shape}"
                )
            arr_t = arr.T  # (n_samples, n_pixels)
            stage_wfs.append(np.ascontiguousarray(arr_t[:n_frames]))
        wfs.append(stage_wfs)

    # --------------------------------------------------------------
    # 5. Create CameraDisplay for each (stage, telescope) animated panel
    # --------------------------------------------------------------
    disps = [[None for _ in range(num_telescopes)] for _ in range(n_stages_anim)]
    animated_artists = []

    for s in range(n_stages_anim):          # stage index, 0 = input
        for t in range(num_telescopes):     # telescope index
            ax = _camera_ax(s, t)
            wf = wfs[s][t]                  # (n_frames, n_pixels)

            source_index, stage_obj, stage_name, stage_geom = display_specs[s]
            if stage_obj is None:
                stage_label = stage_name
            else:
                stage_label = stage_name

            vmin_i = float(np.nanmin(wf))
            vmax_i = float(np.nanmax(wf))

            if np.isnan(wf).any():
                print(f"Warning: NaN values found in stage {s}, telescope {t}")

            # Simple trigger flag based on waveform content
            if show_trigger_status and stage_obj is not None:
                triggered = np.any(wf > 0)
                stage_label += "\n(Triggered)" if triggered else "\n(Not Triggered)"

            disp = CameraDisplay(stage_geom, ax=ax, title=stage_label, cmap=cmap)
            disp.set_limits_minmax(vmin_i, vmax_i)
            if not hide_range_axis:
                disp.add_colorbar(label="ADC counts")

            # Make it an animated artist for blitting
            disp.pixels.set_animated(True)
            ax.set_axis_off()

            disps[s][t] = disp
            animated_artists.append(disp.pixels)

            # Distribution of the whole datacube values for this stage (all
            # frames x pixels), drawn once on the row right below the camera.
            if show_distrib:
                hist_ax = _distrib_ax(s, t)
                values = wf[np.isfinite(wf)].ravel()
                if values.size > 0 and float(np.nanmax(wf)) > float(np.nanmin(wf)):
                    hist_ax.hist(values, bins=50, color="#3b6ea5")
                else:
                    hist_ax.hist(values, bins=10, color="#3b6ea5")
                hist_ax.set_yscale("log")
                hist_ax.set_ylabel("count", fontsize=8)
                hist_ax.tick_params(axis="both", labelsize=7)
                hist_ax.margins(x=0.01)

    # --------------------------------------------------------------
    # 7. Animation callbacks
    # --------------------------------------------------------------
    def init():
        for s in range(n_stages_anim):
            for t in range(num_telescopes):
                disp = disps[s][t]
                wf = wfs[s][t]
                disp.pixels.set_array(wf[0])
        return animated_artists

    def update(frame):
        for s in range(n_stages_anim):
            for t in range(num_telescopes):
                disp = disps[s][t]
                wf = wfs[s][t]
                disp.pixels.set_array(wf[frame])
        return animated_artists

    # First draw so that blitting works
    fig.canvas.draw()

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=50,
        blit=True,
        repeat=True,
        repeat_delay=100,
        cache_frame_data=False,
    )

    # --------------------------------------------------------------
    # 8. Text box with per-telescope info
    # --------------------------------------------------------------
    fig.subplots_adjust(bottom=0.18)

    lines = []
    for tel_id, stats in enumerate(stat_event):
        telescope_label = _display_telescope_id(tel_id, stats)
        lines.append(
            f"Telescope {telescope_label}: "
            f"Event ID: {stats['event_id']} | "
            f"n_pe: {stats['n_pe']} | "
            f"Energy: {stats['energy']:.2f} TeV"
        )

    info_text = "\n".join(lines)

    fig.text(
        0.5,
        0.02,  # vertical position in figure coordinates
        info_text,
        ha="center",
        va="bottom",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
    )

    # --------------------------------------------------------------
    # 9. Show or save to GIF
    # --------------------------------------------------------------
    signature_name = self.generate_chain_signature()
    if generate_image_gif:
        ev_id0 = stat_event[0]["event_id"]
        gif_filename = f"trigger_chain_event_{ev_id0}_{signature_name}.gif"

        ani.save(gif_filename, writer="pillow", fps=13, dpi=dpi)
        print(f"Saved animation to {gif_filename}")

        folder_name = f"trigger_chain_event_{ev_id0}_{signature_name}_frames"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for frame in range(n_frames):
            update(frame)
            fig.canvas.draw()
            plt.savefig(os.path.join(folder_name, f"frame_{frame:04d}.png"))

        print(f"Saved individual frames to folder {folder_name}")
    else:
        plt.show()

# show r0, r1, dl0, dl1 calibrated data like the above show_trigger_chain function but each stage is reaplced by r0, r1, dl0, dl1
  def show_data_calibrated(
        self,
        event_id=None,
        event_index=None,
        minnpe=None,
        maxnpe=None,
        rangenpe=None,
        rangenpe2=None, # in case we search for an event captured by two 2 different telescopes
        energy_range=None,
        skip_first_n_events=-1,
        generate_image_gif=False,
        # limit_telescope=None, # to limit the number of telescope for each event, if 1, 
        cmap='inferno',
        dpi=125,
        folder=".",
        base_name="calibrated_data",
        transpose_layout=False,
        stages_to_show=None,
        n_events=1
        ):

    def _render_single_event(
        wf_r0_list_telescope,
        wf_r1_list_telescope,
        dl0_list_telescope,
        dl1_list_telescope,
        true_image_list_telescope,
        pedestal_per_sample_list_telescope,
        stat_event,
    ):
        num_telescopes = len(dl1_list_telescope)
        for stats in stat_event:
            print(
                f"Telescope {stats['telescope']}: "
                f"event_id={stats['event_id']}, "
                f"n_pe={stats['n_pe']}, "
                f"energy={stats['energy']:.2f} TeV"
            )

        # if the number of pixels is 432 R0 alpha, else R0
        if True:
            stage_specs_all = [
                ("r0", wf_r0_list_telescope, "R0", "ADC counts"),
                ("r1", wf_r1_list_telescope, "R1", "photoelectrons"),
                ("dl0", dl0_list_telescope, "DL0", "photoelectrons"),
                ("dl1", dl1_list_telescope, "DL1", "photoelectrons"),
                ("true_image", true_image_list_telescope, "True Image", "photoelectrons"),
            ]
        else:
            stage_specs_all = [
                ("r0", wf_r0_list_telescope, "R0 Alpha", "ADC counts"),
                ("r1", wf_r1_list_telescope, "R1 Alpha", "photoelectrons"),
                ("dl0", dl0_list_telescope, "DL0 Alpha", "photoelectrons"),
                ("dl1", dl1_list_telescope, "DL1 Alpha", "photoelectrons"),
                ("true_image", true_image_list_telescope, "True Image Alpha", "photoelectrons"),
            ]

        if stages_to_show is None:
            stage_specs = stage_specs_all
        else:
            if isinstance(stages_to_show, str):
                requested = {stages_to_show.lower()}
            else:
                requested = {s.lower() for s in stages_to_show}

            alias_map = {
                "trueimage": "true_image",
                "true": "true_image",
                "truth": "true_image",
            }
            normalized = set()
            for item in requested:
                key = item.replace("_", "")
                key = alias_map.get(key, key)
                normalized.add(key)

            stage_specs = [spec for spec in stage_specs_all if spec[0].replace("_", "") in normalized]
            if not stage_specs:
                print("No valid stages_to_show provided. Valid options: r0, r1, dl0, dl1, true_image.")
                return

        wf_list_result = [spec[1] for spec in stage_specs]
        stage_names = [spec[2] for spec in stage_specs]
        stage_colorbars = [spec[3] for spec in stage_specs]
        n_stages_anim = len(stage_specs)

        if transpose_layout:
            nrows, ncols = n_stages_anim, num_telescopes
        else:
            nrows, ncols = num_telescopes, n_stages_anim

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = np.asarray(axes)
        if axes.ndim == 0:
            axes = axes.reshape(1, 1)
        else:
            axes = axes.reshape(nrows, ncols)

        n_frames = None
        n_pixels = None
        for stage_wf_list in wf_list_result:
            if stage_wf_list is None:
                continue
            for wf in stage_wf_list:
                if wf is None:
                    continue
                arr = np.asarray(wf)
                if arr.ndim == 2:
                    n_pixels = arr.shape[0]
                    n_frames = arr.shape[1]
                    break
                if arr.ndim == 1:
                    n_pixels = arr.shape[0]
                    n_frames = 1
                    break
            if n_frames is not None:
                break

        if n_frames is None or n_pixels is None:
            print("No waveform data available to display for the requested stages.")
            plt.close(fig)
            return

        wfs = []
        for stage_wf_list in wf_list_result:
            stage_wfs = []
            if stage_wf_list is None or len(stage_wf_list) == 0:
                for _ in range(num_telescopes):
                    arr_t = np.zeros((n_frames, n_pixels), dtype=np.float32)
                    stage_wfs.append(arr_t)
                wfs.append(stage_wfs)
                continue
            for wf in stage_wf_list:
                if wf is None:
                    arr_t = np.zeros((n_frames, n_pixels), dtype=np.float32)
                    stage_wfs.append(arr_t)
                    continue
                arr = np.asarray(wf, dtype=np.float32)
                if arr.ndim != 2:
                    if arr.ndim == 1:
                        arr = np.tile(arr[:, np.newaxis], (1, n_frames))
                    else:
                        raise ValueError(
                            "Waveform must be 2D (n_pixels, n_samples), "
                            f"got shape {arr.shape}"
                        )
                arr_t = arr.T
                stage_wfs.append(np.ascontiguousarray(arr_t[:n_frames]))
            wfs.append(stage_wfs)

        disps = [[None for _ in range(num_telescopes)] for _ in range(n_stages_anim)]
        animated_artists = []
        for s in range(n_stages_anim):
            for t in range(num_telescopes):
                ax = axes[s, t] if transpose_layout else axes[t, s]
                wf = wfs[s][t]
                stage_label = f"{stage_names[s]} — Tel {t + 1}"
                vmin_i = float(np.nanmin(wf))
                vmax_i = float(np.nanmax(wf))
                if np.isnan(wf).any():
                    print(f"Warning: NaN values found in stage {s}, telescope {t}")
                disp = CameraDisplay(self.geom, ax=ax, title=stage_label, cmap=cmap)
                disp.set_limits_minmax(vmin_i, vmax_i)
                colorbar_label = stage_colorbars[s]
                if colorbar_label:
                    disp.add_colorbar(label=colorbar_label)
                disp.pixels.set_animated(True)
                ax.set_axis_off()
                disps[s][t] = disp
                animated_artists.append(disp.pixels)

        def init():
            for s in range(n_stages_anim):
                for t in range(num_telescopes):
                    disp = disps[s][t]
                    wf = wfs[s][t]
                    disp.pixels.set_array(wf[0])
            return animated_artists

        def update(frame):
            for s in range(n_stages_anim):
                for t in range(num_telescopes):
                    disp = disps[s][t]
                    wf = wfs[s][t]
                    disp.pixels.set_array(wf[frame])
            return animated_artists

        fig.canvas.draw()
        ani = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=n_frames,
            interval=50,
            blit=True,
            repeat=True,
            repeat_delay=100,
            cache_frame_data=False,
        )

        fig.subplots_adjust(bottom=0.18)
        lines = []
        for tel_id, stats in enumerate(stat_event):
            lines.append(
                f"Telescope {tel_id + 1}: "
                f"Event ID: {stats['event_id']} | "
                f"n_pe: {stats['n_pe']} | "
                f"Energy: {stats['energy']:.2f} TeV"
            )
        info_text = "\n".join(lines)
        fig.text(
            0.5,
            0.02,
            info_text,
            ha="center",
            va="bottom",
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
        )

        if generate_image_gif:
            ev_id0 = stat_event[0]["event_id"]
            gif_filename = f"{base_name}_event_{ev_id0}.gif"
            gif_path = os.path.join(folder, gif_filename)
            ani.save(gif_path, writer="pillow", fps=13, dpi=dpi)
            print(f"Saved animation to {gif_path}")
            folder_name = os.path.join(folder, f"{base_name}_event_{ev_id0}_frames")
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            for frame in range(n_frames):
                update(frame)
                fig.canvas.draw()
                plt.savefig(os.path.join(folder_name, f"frame_{frame:04d}.png"))
            print(f"Saved individual frames to folder {folder_name}")
        else:
            plt.show()

        plt.close(fig)

    if n_events is None or n_events < 1:
        n_events = 1

    if (event_id is not None or event_index is not None) and n_events != 1:
        print("event_id or event_index specified; showing only that event.")
        n_events = 1

    base_skip = skip_first_n_events if skip_first_n_events is not None else -1
    events_rendered = 0

    while events_rendered < n_events:
        current_skip = base_skip
        if current_skip is not None and current_skip >= 0:
            current_skip = base_skip + events_rendered

        wf_r0_list_telescope, wf_r1_list_telescope, dl0_list_telescope, dl1_list_telescope, true_image_list_telescope, pedestal_per_sample_list_telescope, stat_event = self.search_event(
            event_id=event_id,
            event_index=event_index,
            energy_range=energy_range,
            minnpe=minnpe,
            maxnpe=maxnpe,
            rangenpe=rangenpe,
            rangenpe2=rangenpe2,
            skip_first_n_events=current_skip
        )

        if wf_r0_list_telescope is None:
            if events_rendered == 0:
                print("No event found with the given criteria.")
            else:
                print(f"No more events found after {events_rendered} event(s).")
            break

        _render_single_event(
            wf_r0_list_telescope,
            wf_r1_list_telescope,
            dl0_list_telescope,
            dl1_list_telescope,
            true_image_list_telescope,
            pedestal_per_sample_list_telescope,
            stat_event,
        )

        events_rendered += 1
  def generate_chain_signature(self):
    stages_name = "_".join([layer.stage_name() for layer in self.model.layers if hasattr(layer, "stage_name")])
    return stages_name

  def generate_chain_list(self):
    trigger_chain_info = []
    for layer in self.model.layers:
        if hasattr(layer, "stage_type") and hasattr(layer, "get_params"):
            trigger_chain_info.append((layer.stage_type(), layer.get_params()))
    return trigger_chain_info
    
  def generate_output_filename(self, folder, base_name="gamma", suffix="stats.h5"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    stages_name = self.generate_chain_signature()
    # file_name = f"{base_name}_{stages_name}_stats.h5"
    file_name = f"{base_name}_stats_{stages_name}_{suffix}"
    # Most filesystems cap a single name component at 255 bytes. Chains with many
    # score-quantizer edges (each listed inline) overflow that; replace the long
    # stages_name with a stable hash so the name fits. The hash is deterministic
    # and identical across suffixes, so the model/history/weights still share a
    # prefix and inspect_training resolves them together.
    MAX_NAME_BYTES = 255
    if len(file_name.encode("utf-8")) > MAX_NAME_BYTES:
        import hashlib
        digest = hashlib.md5(stages_name.encode("utf-8")).hexdigest()
        file_name = f"{base_name}_stats_chain{digest}_{suffix}"
    return os.path.join(folder, file_name)
    
  def plotDistribution(self, type_data='energy', dataset='gamma', bins=50, show=False):
    event_stat_list = []
    if dataset == 'gamma':
        simtel_files = ([self.simtel_path] if isinstance(self.simtel_path, str) else self.simtel_path)
    elif dataset == 'nsb':
        simtel_files = ([self.simtel_nsb_path] if isinstance(self.simtel_nsb_path, str) else self.simtel_nsb_path)
    else:
        raise ValueError("dataset must be 'gamma' or 'nsb'")
    for simtel_file in simtel_files:
        # with FileOpenerCTAO(simtel_file) as fo:
        with AsyncFileOpenerProcess(simtel_file) as fo:
            for tel_ids_list, wf_list, _, _, _, true_image_list, pedestal_per_sample_list, event_stat_list_per_event, i_event in fo:
                for tel_id, stats in zip(tel_ids_list, event_stat_list_per_event):
                    n_pe = stats['n_pe']
                    energy = stats['energy']
                    event_stat_list.append( (n_pe, energy) )
                if i_event % 100 == 0:
                    print(f"Processed {i_event} events for distribution plot from {simtel_file}")
    # plot the distribution using matplotlib

    energy_bins = np.logspace(np.log10(0.005), np.log10(50), bins)
    n_pe_bins = np.linspace(0, 2000, num=bins)
    plt.figure(figsize=(10,6))
    if type_data == 'energy':
        plt.title(f"Energy Distribution - {dataset} dataset")
        energies = [stat[1] for stat in event_stat_list]
        plt.hist(energies, bins=energy_bins, alpha=0.7, label=f"{dataset} events", color='blue', edgecolor='black')
    elif type_data == 'n_pe':
        plt.title(f"Number of Photoelectrons (n_pe) Distribution - {dataset} dataset")
        n_pes = [stat[0] for stat in event_stat_list]
        plt.hist(n_pes, bins=n_pe_bins, alpha=0.7, label=f"{dataset} events", color='blue', edgecolor='black')
    else:
        raise ValueError("type_data must be 'energy' or 'n_pe'")
    if type_data == 'energy':
        plt.xlabel("Energy (TeV)")
        plt.xscale('log')
    elif type_data == 'n_pe':
        plt.xlabel("Number of Photoelectrons (n_pe)")
        plt.xscale('log')
    plt.ylabel("Number of Events")
    plt.grid()
    # plt.yscale('log')
    plt.legend()
    plt.savefig(f"{dataset}_distribution_{type_data}.png")
    if show:
        plt.show()
    else:
        plt.close()
