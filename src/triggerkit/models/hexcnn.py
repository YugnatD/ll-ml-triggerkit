"""Convenience preset: Jakub's Hex3DHybridCNN feature backbone as a layer list.

A layer-for-layer port of ``train_hex_cnn.build_adapter`` +
``build_hex3d_backbone``, expressed as a
:class:`~triggerkit.models.sequential.SequentialBody` layer list:

    scatter_to_grid
    -> [pad+Conv3D(5,1,1) s2 + ReLU] x1
    -> [pad+Conv3D(3,1,1) s2 + ReLU] x1
    -> time_mean
    -> [hgly.Conv2d(k2,s2) + ReLU] per spatial channel count
    -> global_hex_mean          (feature vector; head added by build_chain)

``keras_hexagdly`` is imported lazily (only when the list is built), so the
package stays importable without the optional ``[hexcnn]`` extra. This is just a
preset; for a different CNN write your own ``SequentialBody`` layer list.
"""

from tensorflow import keras

from triggerkit.models.sequential import SequentialBody


def _pad_temporal(pad):
    """Escape-hatch step: symmetric ``keras.ops.pad`` on the time axis."""

    def _step(chain):
        chain.last_layer = keras.ops.pad(
            chain.last_layer, [[0, 0], [pad, pad], [0, 0], [0, 0], [0, 0]])
        return None

    return _step


def hex3d_hybrid_layers(
    *,
    time_skip=0,
    time_window=32,
    temporal_channels=8,
    spatial_channels=(16, 32),
):
    """Return the SequentialBody *feature* layer list for the Hex3DHybrid backbone.

    The list ends at ``global_hex_mean`` (a ``(B, C)`` feature vector); the
    ``Dense`` classifier head + threshold are added by :func:`build_chain`. So
    this is a convenience preset -- for a different CNN just write your own list::

        chain, head, thr = build_chain(ds, SequentialBody(hex3d_hybrid_layers()), filters=4)

    ``keras_hexagdly`` must be installed (``pip install '.[hexcnn]'``).
    """
    import keras_hexagdly as hgly  # optional dependency, imported on demand

    layers = [
        ("scatter_to_grid", {"time_skip": time_skip, "time_window": time_window}),
        # nn.Conv3d(1, C, (5,1,1), stride=(2,1,1), padding=(2,0,0))
        _pad_temporal(2),
        keras.layers.Conv3D(
            temporal_channels, (5, 1, 1), strides=(2, 1, 1),
            padding="valid", name="temporal_0"),
        keras.layers.ReLU(),
        # nn.Conv3d(C, C, (3,1,1), stride=(2,1,1), padding=(1,0,0))
        _pad_temporal(1),
        keras.layers.Conv3D(
            temporal_channels, (3, 1, 1), strides=(2, 1, 1),
            padding="valid", name="temporal_2"),
        keras.layers.ReLU(),
        "time_mean",
    ]

    prev_c = temporal_channels
    for i, out_c in enumerate(spatial_channels):
        layers.append(
            hgly.Conv2d(
                prev_c, out_c, kernel_size=2, stride=2, bias=True,
                share_neighbors=False, name=f"spatial_{2 * i}"))
        layers.append(keras.layers.ReLU())
        prev_c = out_c

    layers.append("global_hex_mean")
    return layers


def hex3d_hybrid_body(**kwargs):
    """Convenience: ``SequentialBody(hex3d_hybrid_layers(**kwargs))`` (feature body)."""
    return SequentialBody(hex3d_hybrid_layers(**kwargs))
