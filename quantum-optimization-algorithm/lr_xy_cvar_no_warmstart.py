"""Ablation row: LR-QAOA + XY-mixer + CVaR with NO learned warm-start.

Identical to `hybrid_gnn_lr_xy_cvar` except the schedule deltas are pinned
to the literature defaults (delta_gamma = delta_beta = 0.5) instead of being
predicted by the GNN/ridge model. Used to measure the marginal contribution
of the learned warm-start.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_hyb_spec = importlib.util.spec_from_file_location(
    "_hybrid_for_ablation",
    ROOT / "quantum-optimization-algorithm" / "hybrid_gnn_lr_xy_cvar.py",
)
_hyb = importlib.util.module_from_spec(_hyb_spec)
sys.modules["_hybrid_for_ablation"] = _hyb
_hyb_spec.loader.exec_module(_hyb)

ALGORITHM = "lr_xy_cvar_no_warmstart"
ALGORITHM_VERSION = "1.0.0"
MAX_N = _hyb.MAX_N


def solve(inst, seed: int = 42, p: int = 20, alpha: float = 0.25,
          shots: int = 4096):
    # monkey-patch the warm-start predictor to always return the literature
    # defaults; the rest of the hybrid pipeline is unchanged.
    orig = _hyb._predict_schedule
    _hyb._predict_schedule = lambda _inst: (
        _hyb.FALLBACK_DELTA_GAMMA,
        _hyb.FALLBACK_DELTA_BETA,
        "no_warmstart_fixed_0.5",
    )
    try:
        r = _hyb.solve(inst, seed=seed, p=p, alpha=alpha, shots=shots)
    finally:
        _hyb._predict_schedule = orig
    r.algorithm = ALGORITHM
    r.algorithm_version = ALGORITHM_VERSION
    r.hyperparameters["warmstart_source"] = "literature_default_0.5"
    return r
