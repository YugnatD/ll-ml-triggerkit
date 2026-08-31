"""Pluggable trigger-chain *bodies*.

A :class:`TriggerBody` encapsulates the full sequence of stages that turns a
``TriggerChain``'s input tensors into its pre-loss output. It is the seam that
lets the fixed TDSCAN chain and an ever-changing CNN share one training/stats
path: swap the body, keep everything else.

Contract
--------
``body.build(chain)`` must:

* start from ``chain.last_layer`` (the input cursor) and append stages, either
  through ``chain.add_stage(name, **kw)`` or by applying Keras layers directly
  and updating ``chain.last_layer`` / ``chain.last_geom`` itself;
* leave ``chain.last_layer`` holding the model's output tensor (raw score /
  logits -- the loss is attached later by ``chain.compile_chain``);
* return a ``dict`` of named layer handles a runner may need to tweak
  (e.g. ``{"threshold": ..., "tdscan": ...}``). Values may be lists (one per
  branch / filter).

Bodies never compile. :func:`build_chain` wires build + compile together.
"""

import abc

_BODY_REGISTRY = {}


class TriggerBody(abc.ABC):
    """Base class for a pluggable trigger-chain body. See module docstring."""

    @abc.abstractmethod
    def build(self, chain):
        """Append this body's stages onto ``chain``; return a handles dict."""
        raise NotImplementedError


def register_body(name):
    """Class decorator: register a :class:`TriggerBody` under ``name``."""

    def _decorator(cls):
        if name in _BODY_REGISTRY and _BODY_REGISTRY[name] is not cls:
            raise KeyError(f"body {name!r} already registered to {_BODY_REGISTRY[name]!r}")
        _BODY_REGISTRY[name] = cls
        return cls

    return _decorator


def get_body(name, **kwargs):
    """Instantiate a registered body by name with the given constructor kwargs."""
    if name not in _BODY_REGISTRY:
        raise KeyError(
            f"unknown body {name!r}; registered: {sorted(_BODY_REGISTRY)}"
        )
    return _BODY_REGISTRY[name](**kwargs)


def registered_bodies():
    """Return the list of registered body names."""
    return sorted(_BODY_REGISTRY)


def build_chain(chain, body, *, loss=None, optimizer=None, model_path=None):
    """Build ``body`` onto ``chain`` and compile it.

    Returns ``(handles, chain)`` where ``handles`` is whatever ``body.build``
    returned. ``chain.model`` is the compiled Keras model afterwards.
    """
    handles = body.build(chain)
    chain.compile_chain(loss=loss, optimizer=optimizer, model_path=model_path)
    return handles, chain
