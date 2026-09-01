"""Pluggable trigger-chain *bodies* and the generic assembler.

A *body* is the sequence of stages that turns a ``TriggerChain``'s input tensors
into its feature / score tensor. It is the seam that lets a hand-declared CNN and
the fixed TDSCAN chain share one training / stats path.

There is deliberately NO registry or ``get_body("name")`` lookup: a body is just
an object with a ``build(chain)`` method, so you construct it directly --
``SequentialBody([...])`` for a CNN you write out layer by layer, or
``TDSCANBody(...)`` for the fixed TDSCAN chain -- and hand it to
:func:`build_chain`.

Contract
--------
``body.build(chain)`` must:

* start from ``chain.last_layer`` (the input cursor) and append stages, either
  through ``chain.add_stage(name, **kw)`` or by applying Keras layers directly
  and updating ``chain.last_layer`` / ``chain.last_geom`` itself;
* leave ``chain.last_layer`` holding the body's output tensor;
* return a ``dict`` of named layer handles (e.g. ``{"tdscan": ..., "threshold":
  ...}``). Values may be lists (one per branch / filter).

A body may either end on its own trigger *head* (a ``TrainableThreshold``, like
:class:`~triggerkit.models.tdscan.TDSCANBody`) or stop at a plain *feature*
tensor (like a CNN ``SequentialBody`` ending at ``GlobalHexMean``). In the second
case :func:`build_chain` appends a ``Dense(filters)`` classifier + threshold for
you. It tells the two apart by looking for a threshold layer (one with ``.tau``)
among the returned handles.
"""

import abc


class TriggerBody(abc.ABC):
    """Base class for a pluggable trigger-chain body. See module docstring."""

    @abc.abstractmethod
    def build(self, chain):
        """Append this body's stages onto ``chain``; return a handles dict."""
        raise NotImplementedError


def _as_chain(source):
    """Return a ``TriggerChain`` from either a chain or a ``TriggerDataset``.

    A dataset is recognised by carrying ``gamma_files`` / ``nsb_files``; a fresh
    chain is built from them. Anything else is assumed to already be a chain.
    """
    if hasattr(source, "gamma_files") and hasattr(source, "nsb_files"):
        from triggerkit.TriggerChain import TriggerChain
        return TriggerChain(source.gamma_files, simtel_nsb_path=source.nsb_files)
    return source


def _find_threshold(handles):
    """The threshold layer (has ``.tau``) among ``handles``, or ``None``."""
    if not isinstance(handles, dict):
        return None
    for h in handles.values():
        items = h if isinstance(h, (list, tuple)) else [h]
        for layer in items:
            if hasattr(layer, "tau"):
                return layer
    return None


def build_chain(source, body, *, filters=1, head_name="classifier",
                init_tau=0.0, temp=10.0, binary_output=True,
                loss=None, optimizer=None, model_path=None):
    """Assemble a full trigger model from a data source + a body.

    Parameters
    ----------
    source : TriggerDataset or TriggerChain
        A dataset (a fresh chain is built from its files) or an existing chain.
    body : TriggerBody
        The feature / score body, e.g. ``SequentialBody([...])`` or ``TDSCANBody(...)``.
    filters : int
        Width of the auto-appended ``Dense`` classifier head -- i.e. the number of
        parallel restart columns -- for a *feature* body. Ignored for a body that
        already ends on its own threshold (e.g. TDSCAN).
    head_name : str
        Name of the auto-appended ``Dense`` head (so a trainer can grab it).
    init_tau, temp, binary_output :
        Threshold knobs for the auto-appended head.
    loss, optimizer, model_path :
        Passed straight to ``chain.compile_chain``.

    Returns
    -------
    (chain, head_layer, threshold_layer)
        ``chain.model`` is compiled (threshold-terminated) afterwards.
        ``head_layer`` is the auto ``Dense`` head for a feature body, or the
        body's own main trainable layer (e.g. the TDSCAN layer) for a self-headed
        body. Values may be lists for a multi-branch body.
    """
    from tensorflow import keras

    chain = _as_chain(source)
    handles = body.build(chain)
    handles = handles if isinstance(handles, dict) else {}

    thr = _find_threshold(handles)
    if thr is not None:
        # Self-headed body (e.g. TDSCAN): keep its own threshold; do not add a head.
        head = handles.get("tdscan", handles.get(head_name, thr))
        thr_handle = handles.get("threshold", thr)
    else:
        # Feature body (e.g. CNN): append a Dense classifier + threshold.
        head = keras.layers.Dense(filters, name=head_name)
        chain.last_layer = head(chain.last_layer)
        thr_handle = chain.add_stage(
            "threshold", init_tau=init_tau, temp=temp, binary_output=binary_output)

    chain.compile_chain(loss=loss, optimizer=optimizer, model_path=model_path)
    return chain, head, thr_handle
