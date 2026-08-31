import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Trigger")
class OrMerge(tf.keras.layers.Layer):
    """Logical OR of several parallel trigger branches.

    Each branch ends in a ``TrainableThreshold`` whose output is a per-branch
    fire signal:
      * training (binary_output=False): a soft fire *probability*
        ``p = sigmoid((score - tau) * temp)`` in [0, 1];
      * deployment (binary_output=True): a hard fire *bit* in {0, 1}.

    This layer combines the branches with the probabilistic OR

        OR = 1 - prod_i (1 - p_i)

    which is exactly logical OR when the inputs are 0/1 bits, and a smooth,
    differentiable surrogate when they are probabilities. The trigger fires if
    *any* branch fires. Because the OR's output for a given filter column mixes
    that same column from every branch, a per-column AUC loss on this output
    trains the branches *jointly*: branch B is only rewarded for catching what
    branch A misses, so the two learn to be complementary rather than identical.

    Inputs keep their shape: each branch output is ``(B, F)`` (F = number of
    parallel TDSCAN filters, 1 for the deployed chain), and so is the OR.
    """

    def __init__(self, input_geometry=None, **kwargs):
        super().__init__(**kwargs)
        # Geometry is meaningless for a pure OR of event-level fire signals, but
        # we keep the attribute so the layer matches the other stages' interface
        # (TriggerChain reads .output_geometry off every stage it adds).
        self.input_geometry = input_geometry
        self.output_geometry = input_geometry

    def stage_name(self):
        return "ormerge"

    def stage_type(self):
        return "or_merge"

    def get_params(self):
        # No tunable structure: the OR is fully defined by how many branches feed
        # it, which the fingerprint already captures via the surrounding stages.
        return {}

    def get_stages(self):
        return (self.stage_type(), self.get_params())

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("OrMerge expects a list of branch outputs.")
        # 1 - prod(1 - p_i). Clip into [0, 1] so straight-through hard bits that
        # land slightly outside the range (numerical noise) can't drive the
        # product negative.
        not_fired = None
        for branch in inputs:
            p = tf.clip_by_value(tf.cast(branch, tf.float32), 0.0, 1.0)
            term = 1.0 - p
            not_fired = term if not_fired is None else not_fired * term
        return 1.0 - not_fired

    def get_config(self):
        cfg = super().get_config()
        # input_geometry is intentionally not serialized: it is unused by call()
        # and not JSON-serializable. A reloaded layer simply has geometry=None.
        return cfg
