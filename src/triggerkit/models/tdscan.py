"""``TDSCANBody``: the fixed, deployed TDSCAN trigger chain as a pluggable body.

This reproduces ``tdscan_chain._add_branch`` / ``build_chain`` from the research
sandbox exactly: optional shift + score quantizer front-end, the learnable
TDSCAN filter, camera-wide max pool, and a trainable threshold. A multi-branch
OR trigger is built when ``branches`` is given.

The chain itself is *fixed* -- unlike the CNN, its shape does not change between
experiments -- so it is a concrete body rather than something declared as a
layer list. Knobs mirror the sandbox's ``build_chain`` arguments.
"""

from triggerkit.models.base import TriggerBody, register_body


@register_body("tdscan")
class TDSCANBody(TriggerBody):
    """Fixed TDSCAN chain: [shift?] -> [score_quantizer?] -> tdscan -> max-pool -> threshold.

    Parameters mirror ``tdscan_chain.build_chain``. Per-branch knobs
    (``eps_xy``, ``eps_t``, ``edges_*``, ``shift_value``, ``quantize_step``,
    ``ring_weights`` / ``kernel_weights``, ``tau``, the penalty terms) may be
    overridden per branch via ``branches``; ``filters`` and ``share_neighbors``
    are shared across branches (the trainer owns the multi-start count).

    Parameters
    ----------
    branches : list of dict or None
        ``None`` -> a single chain (``handles`` values are single layers).
        A list of >=2 per-branch override dicts -> that many parallel branches
        combined with an ``or_merge`` (``handles`` values are lists).
    """

    def __init__(
        self,
        *,
        eps_xy=1,
        eps_t=1,
        filters=1,
        share_neighbors=True,
        edges_range=(0.0, 0.0),
        edges_num_bit=0,
        edges_func=None,
        shift_value=None,
        quantize_step=None,
        ring_weights=None,
        kernel_weights=None,
        tau=10.0,
        temp=10.0,
        binary_output=True,
        initializer="glorot_uniform",
        mean_penalty_lambda=0.0,
        mean_penalty_target=0.0,
        std_penalty_lambda=0.0,
        std_penalty_target=0.0,
        branches=None,
    ):
        self.base_kwargs = dict(
            eps_xy=eps_xy, eps_t=eps_t, filters=filters,
            share_neighbors=share_neighbors,
            edges_range=edges_range, edges_num_bit=edges_num_bit, edges_func=edges_func,
            shift_value=shift_value, quantize_step=quantize_step,
            ring_weights=ring_weights, kernel_weights=kernel_weights,
            tau=tau, temp=temp, binary_output=binary_output, initializer=initializer,
            mean_penalty_lambda=mean_penalty_lambda, mean_penalty_target=mean_penalty_target,
            std_penalty_lambda=std_penalty_lambda, std_penalty_target=std_penalty_target,
        )
        self.branches = branches

    # ------------------------------------------------------------------ #
    def _add_branch(
        self, chain, *, eps_xy, eps_t, filters, share_neighbors,
        edges_range, edges_num_bit, edges_func, shift_value, quantize_step,
        ring_weights, kernel_weights, tau, temp, binary_output, initializer,
        mean_penalty_lambda, mean_penalty_target,
        std_penalty_lambda, std_penalty_target,
    ):
        """Add one branch from the current cursor; return (tdscan, threshold)."""
        if shift_value is not None:
            chain.add_stage("shift", value=shift_value)
            pad_value = -shift_value
        else:
            pad_value = 0.0

        if edges_func is not None and edges_num_bit > 0 and edges_range[1] > edges_range[0]:
            edges = edges_func(start=edges_range[0], stop=edges_range[1], num_bit=edges_num_bit)
            chain.add_stage("score_quantizer", edges=edges)

        tdscan_layer = chain.add_stage(
            "tdscan",
            eps_xy=eps_xy, eps_t=eps_t, filters=filters,
            initializer=initializer, share_neighbors=share_neighbors,
            quantize_step=quantize_step, pad_value=pad_value,
            mean_penalty_lambda=mean_penalty_lambda, mean_penalty_target=mean_penalty_target,
            std_penalty_lambda=std_penalty_lambda, std_penalty_target=std_penalty_target,
        )

        if share_neighbors and ring_weights is not None:
            tdscan_layer.set_weights_from_params(share_weights=True, ring_weights=ring_weights)
        elif not share_neighbors and kernel_weights is not None:
            tdscan_layer.set_weights_from_params(share_weights=False, kernel_weights=kernel_weights)
        elif share_neighbors and kernel_weights is not None:
            raise ValueError(
                "kernel_weights provided but share_neighbors=True: use ring_weights instead.")
        elif not share_neighbors and ring_weights is not None:
            raise ValueError(
                "ring_weights provided but share_neighbors=False: use kernel_weights instead. "
                f"Expected a flat array of {tdscan_layer.K * tdscan_layer.L} values "
                f"(L={tdscan_layer.L} time steps x K={tdscan_layer.K} spatial neighbors).")

        chain.add_stage("global_max_pooling_2d")
        threshold_layer = chain.add_stage(
            "threshold", init_tau=tau, temp=temp, binary_output=binary_output)
        return tdscan_layer, threshold_layer

    # ------------------------------------------------------------------ #
    def build(self, chain):
        if self.branches is None:
            tdscan_layer, threshold_layer = self._add_branch(chain, **self.base_kwargs)
            return {"tdscan": tdscan_layer, "threshold": threshold_layer}

        if len(self.branches) < 2:
            raise ValueError(
                "branches must list at least two branch kwargs dicts "
                "(use None for a single chain).")

        start = chain.branch_point()
        tdscan_layers, threshold_layers, branch_outputs = [], [], []
        for i, branch_kwargs in enumerate(self.branches):
            overrides = dict(branch_kwargs or {})
            for shared in ("filters", "share_neighbors"):
                if shared in overrides and overrides[shared] != self.base_kwargs[shared]:
                    raise ValueError(
                        f"branch {i}: {shared!r} is shared across branches and "
                        "cannot be overridden.")
            resolved = dict(self.base_kwargs)
            resolved.update(overrides)
            resolved["filters"] = self.base_kwargs["filters"]
            resolved["share_neighbors"] = self.base_kwargs["share_neighbors"]

            chain.restore_branch_point(start)
            tdscan_layer, threshold_layer = self._add_branch(chain, **resolved)
            tdscan_layers.append(tdscan_layer)
            threshold_layers.append(threshold_layer)
            branch_outputs.append(threshold_layer.output)

        chain.add_stage("or_merge", branches=branch_outputs)
        return {"tdscan": tdscan_layers, "threshold": threshold_layers}
