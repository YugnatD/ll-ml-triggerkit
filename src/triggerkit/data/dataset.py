"""``TriggerDataset``: a single source of truth for the gamma/NSB data feeding a
trigger chain.

It is a thin facade over ``SimTelTFDataset`` / ``SimTelTFDatasetConfig`` (the
streaming reader already used by ``TriggerChain.train_chain``). It owns the file
lists and every dataset knob, does the train/val split (fold-aware for
cross-validation), and hands back the two raw ``tf.data`` pipelines of
``(features, label)`` pairs. Turning those into the exact ``(inputs, y)`` tuples a
model expects (the "pack" step) stays in ``TriggerChain`` because it depends on
the model signature, not on the data.

Design points agreed for the package refactor:

* NSB augmentation rule: a single NSB file is densified by a random per-batch
  circular pixel roll (``nsb_roll_augment``), exactly as ``train_chain`` already
  did. ``nsb_roll_copies`` (fixed +1,+2,... duplicates) is also exposed.
* ``targets``: only ``("class",)`` (the binary gamma/NSB label) is wired today.
  Auxiliary targets such as the true Cherenkov image are NOT emitted by the
  streaming reader yet (``SimTelTFDataset`` opens files with
  ``keep_true_image=False``), so requesting them raises ``NotImplementedError``
  with a pointer rather than silently dropping them.
* ``gated_by``: reserved for the TDSCAN->CNN cascade (train the downstream stage
  only on events that pass a frozen upstream chain). Not wired yet.
* ``gamma_rotations``: reserved for hexagonal rotation augmentation. Rotation
  differs by representation (TDSCAN = pixel-index permutation; CNN = grid H x W
  after remapping), so it needs a real algorithm; for now anything other than
  ``(0,)`` warns and is ignored.
"""

import warnings

from triggerkit.FileIO.FileOpenerCTAO import (
    AsyncFileOpenerProcess,
    SimTelTFDataset,
    SimTelTFDatasetConfig,
)

# Same scalar event features train_chain requested, kept on the stream so a
# downstream pack step (or a future regression target) can read them.
DEFAULT_EVENT_FEATURE_KEYS = (
    "n_pe", "ev_time", "energy", "azimuth", "altitude",
    "h_first_int", "xmax", "xcore", "ycore",
)


class TriggerDataset:
    """Gamma/NSB data source for a trigger chain.

    Pass a list of gamma files and a list of NSB files. Everything else mirrors
    the knobs ``TriggerChain.train_chain`` used to take directly, so the default
    construction inside ``train_chain`` (when no dataset is given) reproduces the
    old behaviour exactly.

    Parameters
    ----------
    gamma_files, nsb_files : sequence of str
        Input files. Gammas are label 1, NSB label 0. A single NSB file is fine
        (it gets roll-augmented).
    batch_size : int
    tel_id_only : int or None
        Keep only this telescope id for gammas.
    n_pe_{min,max}_{train,val} : float or None
        Per-split gamma ``n_pe`` window.
    max_{gamma,nsb}_samples_{train,val} : int or None
        Per-split global caps.
    load_ram : bool
        Cache the whole split in RAM (faster, memory-hungry). Training only.
    waveform_level : {"r0", "r1"}
    percent_validation : float
        Fraction of gamma files held out for validation when ``n_folds == 1``.
    n_folds, fold : int
        K-fold cross-validation over the gamma file list. ``n_folds == 1`` falls
        back to the tail ``percent_validation`` split. NSB files are shared
        across folds (the set is small and label-0 only).
    targets : tuple of str
        Which targets the label pipeline should carry. Only ``("class",)`` is
        implemented; other values raise ``NotImplementedError``.
    gated_by : TriggerChain or None
        Reserved for cascade training. Not implemented yet.
    gamma_rotations : tuple of int
        Reserved for hex rotation augmentation. Non-``(0,)`` warns and is ignored.
    seed : int
    """

    def __init__(
        self,
        gamma_files,
        nsb_files,
        *,
        batch_size=4096,
        tel_id_only=2,
        n_pe_min_train=None,
        n_pe_max_train=None,
        n_pe_min_val=None,
        n_pe_max_val=None,
        max_gamma_samples_train=None,
        max_gamma_samples_val=None,
        max_nsb_samples_train=None,
        max_nsb_samples_val=None,
        load_ram=True,
        waveform_level="r0",
        percent_validation=0.2,
        n_folds=1,
        fold=0,
        targets=("class",),
        gated_by=None,
        gamma_rotations=(0,),
        event_feature_keys=DEFAULT_EVENT_FEATURE_KEYS,
        opener_cls=AsyncFileOpenerProcess,
        seed=1337,
    ):
        self.gamma_files = list(gamma_files)
        self.nsb_files = [nsb_files] if isinstance(nsb_files, str) else list(nsb_files)

        self.batch_size = batch_size
        self.tel_id_only = tel_id_only
        self.n_pe_min_train = n_pe_min_train
        self.n_pe_max_train = n_pe_max_train
        self.n_pe_min_val = n_pe_min_val
        self.n_pe_max_val = n_pe_max_val
        self.max_gamma_samples_train = max_gamma_samples_train
        self.max_gamma_samples_val = max_gamma_samples_val
        self.max_nsb_samples_train = max_nsb_samples_train
        self.max_nsb_samples_val = max_nsb_samples_val
        self.load_ram = load_ram
        self.waveform_level = waveform_level
        self.percent_validation = max(0.0, min(1.0, percent_validation))
        self.event_feature_keys = tuple(event_feature_keys)
        self.opener_cls = opener_cls
        self.seed = seed

        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")
        if not (0 <= fold < n_folds):
            raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
        self.n_folds = n_folds
        self.fold = fold

        self.targets = tuple(targets)
        self._validate_targets()

        if gated_by is not None:
            raise NotImplementedError(
                "gated_by (cascade gating: train only on events passing a frozen "
                "upstream chain) is not wired yet; planned for the cascade step."
            )
        self.gated_by = None

        self.gamma_rotations = tuple(gamma_rotations)
        if self.gamma_rotations != (0,):
            warnings.warn(
                "gamma_rotations augmentation is not implemented yet and will be "
                "ignored. Hex rotation needs a representation-specific algorithm "
                "(TDSCAN = pixel-index permutation; CNN = grid H x W after "
                "remapping); coming later.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    def _validate_targets(self):
        if self.targets != ("class",):
            unsupported = [t for t in self.targets if t != "class"]
            raise NotImplementedError(
                f"targets={self.targets!r}: only ('class',) is implemented. "
                f"Auxiliary targets {unsupported} (e.g. 'true_image') require the "
                "streaming reader to emit them -- SimTelTFDataset currently opens "
                "files with keep_true_image=False. Wiring this is a separate step."
            )

    # ------------------------------------------------------------------ #
    def split_files(self):
        """Return ``(train_gamma_files, val_gamma_files)`` for the current fold.

        ``n_folds == 1`` -> tail ``percent_validation`` split (old behaviour).
        ``n_folds > 1``  -> gamma files chunked into ``n_folds`` contiguous folds;
        the current ``fold`` is validation, the rest is training. NSB files are
        shared (not split) and returned by :meth:`train_val_datasets`.
        """
        files = self.gamma_files
        if self.n_folds == 1:
            if self.percent_validation <= 0.0:
                return list(files), []
            n_val = int(len(files) * self.percent_validation)
            if n_val == 0:
                return list(files), []
            return files[:-n_val], files[-n_val:]

        # K-fold: contiguous chunks (files are typically sorted; the caller can
        # pre-shuffle the list with a fixed seed if a random assignment is wanted).
        n = len(files)
        bounds = [round(i * n / self.n_folds) for i in range(self.n_folds + 1)]
        lo, hi = bounds[self.fold], bounds[self.fold + 1]
        val = files[lo:hi]
        train = files[:lo] + files[hi:]
        return train, val

    # ------------------------------------------------------------------ #
    def _config(self, *, training):
        """Build the ``SimTelTFDatasetConfig`` for the train or val split.

        Mirrors the two configs ``train_chain`` used to build inline: train
        shuffles and uses the train-side caps/n_pe window and roll seed 1337; val
        does not shuffle and uses the val-side caps/window and roll seed 4242.
        """
        if training:
            n_pe_min, n_pe_max = self.n_pe_min_train, self.n_pe_max_train
            max_gamma, max_nsb = self.max_gamma_samples_train, self.max_nsb_samples_train
            shuffle, roll_seed = True, self.seed
        else:
            n_pe_min, n_pe_max = self.n_pe_min_val, self.n_pe_max_val
            max_gamma, max_nsb = self.max_gamma_samples_val, self.max_nsb_samples_val
            shuffle, roll_seed = False, self.seed + 2905  # 1337 -> 4242, as before

        return SimTelTFDatasetConfig(
            batch_size=self.batch_size,
            shuffle_samples=shuffle,
            sample_shuffle_buffer=10000,
            seed=self.seed,
            load_ram=self.load_ram,
            interleave_files=True,
            waveform_level=self.waveform_level,
            gamma_tel_id_only=self.tel_id_only,
            gamma_n_pe_max=n_pe_max,
            gamma_n_pe_min=n_pe_min,
            gamma_skip_if_missing_n_pe=True,
            include_event_features=True,
            event_feature_keys=self.event_feature_keys,
            nsb_skip_original_events=False,
            nsb_roll_copies=0,
            nsb_roll_axis=1,
            nsb_roll_augment=True,
            nsb_roll_seed=roll_seed,
            max_gamma_samples_total=max_gamma,
            max_nsb_samples_total=max_nsb,
            repeat=False,
            ignore_errors=True,
        )

    def _dataset(self, gamma_files, *, training):
        """Raw ``tf.data`` of ``(features, label)`` for a gamma-file subset.

        ``features`` is the dict emitted by ``SimTelTFDataset`` (waveform,
        pedestal, event_id, tel_id, and the scalar event features). Packing to
        model inputs happens downstream.
        """
        return SimTelTFDataset(
            gamma_files=gamma_files,
            nsb_files=self.nsb_files,
            opener_cls=self.opener_cls,
            config=self._config(training=training),
        ).dataset()

    # ------------------------------------------------------------------ #
    def train_val_datasets(self):
        """Return ``(train_ds, val_ds)`` raw ``tf.data`` pipelines.

        ``val_ds`` is ``None`` when the split yields no validation files.
        """
        train_files, val_files = self.split_files()
        train_ds = self._dataset(train_files, training=True)
        val_ds = self._dataset(val_files, training=False) if val_files else None
        return train_ds, val_ds
