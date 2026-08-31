# ll-ml-triggerkit

Trainable low-level ML trigger chains for Cherenkov cameras (SST-1M / DigiCam).

Distribution name: `ll-ml-triggerkit`. Import name: `triggerkit`.

```bash
pip install -e .            # dev install
pip install -e '.[hexcnn]'  # + hex CNN bodies (keras-hexagdly)
```

```python
import triggerkit
from triggerkit.TriggerChain import TriggerChain
```

## Layout

- `src/triggerkit/` — the library.
  - `TriggerChain.py` — core chain (`train_chain`, `compute_statistics`, `show_trigger_chain`, ...).
  - `Stages/ Loss/ Metric/ Callback/ Statistics/ FileIO/` — building blocks.
  - `data/ models/ training/` — dataset, pluggable bodies, training utilities (added incrementally).
- `examples/` — train / evaluate / show scripts.
- `tests/`

## Status

Bootstrapping: package extracted from the `python-reference` research sandbox and
renamed to `triggerkit`. Incremental refactor in progress (dataset, pluggable
model bodies, cascade statistics).
