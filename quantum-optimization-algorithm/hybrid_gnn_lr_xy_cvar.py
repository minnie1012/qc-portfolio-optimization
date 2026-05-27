"""Novel hybrid: GNN-warmstart LR-QAOA + XY-mixer + CVaR objective.

Pipeline:
    1. GNN (3-layer GraphSAGE + MLP) predicts (Delta_gamma, Delta_beta) for the
       LR-QAOA linear-ramp schedule, given the instance as a feature graph.
       Falls back to medians from training if the model file is missing.
    2. Dicke initial state |D_K^N> + XY-ring mixer preserves Hamming weight K
       exactly under the variational evolution.
    3. Apply the LR-QAOA schedule (no inner optimizer; one forward pass per p).
    4. Sample bitstrings, compute CVaR-alpha of the energy distribution as the
       reported objective; return the best K-cardinality bitstring found.

Combines four recent advances (LR-QAOA Nature 2025; XY-mixer constraint
preservation; CVaR risk-aware sampling; GNN cross-instance parameter transfer)
in one solver."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_protocol import gate_metrics, instances  # noqa: E402
from benchmark_protocol.result_schema import BenchmarkResult  # noqa: E402

from solver_common import approx_ratio, build_qubo_from_inst, exact_reference, qubo_eval  # noqa: E402

# Reuse the XY+Dicke ansatz builder from qaoa_xy.py
_xy_spec = importlib.util.spec_from_file_location(
    "_qaoa_xy_for_hybrid",
    ROOT / "quantum-optimization-algorithm" / "qaoa_xy.py",
)
_xy_mod = importlib.util.module_from_spec(_xy_spec)
sys.modules["_qaoa_xy_for_hybrid"] = _xy_mod
_xy_spec.loader.exec_module(_xy_mod)

ALGORITHM = "hybrid_gnn_lr_xy_cvar"
ALGORITHM_VERSION = "1.0.0"
MAX_N = 10
MODEL_PATH = ROOT / "models" / "gnn_warmstart.pt"
LINEAR_MODEL_PATH = ROOT / "models" / "warmstart_linear.json"
FALLBACK_DELTA_GAMMA = 0.5
FALLBACK_DELTA_BETA = 0.5


def _instance_features(inst: instances.ProblemInstance) -> dict:
    """Compact, training-stable graph features for the predictor."""
    diag = np.diag(inst.sigma)
    return {
        "N": int(inst.N),
        "K": int(inst.K),
        "q": float(inst.q),
        "mu_mean": float(np.mean(inst.mu)),
        "mu_std": float(np.std(inst.mu)),
        "sigma_diag_mean": float(np.mean(diag)),
        "sigma_offdiag_abs_mean": float(np.mean(np.abs(inst.sigma - np.diag(diag)))),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvalsh(inst.sigma)))),
    }


def _linear_features(inst: instances.ProblemInstance) -> np.ndarray:
    diag = np.diag(inst.sigma)
    off = inst.sigma - np.diag(diag)
    return np.array([
        1.0,
        float(inst.N),
        float(inst.K) / max(inst.N, 1),
        float(inst.q),
        float(np.mean(inst.mu)),
        float(np.std(inst.mu)),
        float(np.mean(diag)),
        float(np.mean(np.abs(off))),
        float(np.max(np.abs(np.linalg.eigvalsh(inst.sigma)))),
    ], dtype=float)


def _predict_from_linear_json(inst: instances.ProblemInstance) -> tuple[float, float, str] | None:
    if not LINEAR_MODEL_PATH.exists():
        return None
    try:
        import json
        with open(LINEAR_MODEL_PATH) as f:
            m = json.load(f)
        W = np.array(m["coefficients"], dtype=float)  # (n_features, 2)
        x = _linear_features(inst)
        dg, db = (x @ W).tolist()
        lo, hi = m.get("clip", [0.05, 2.0])
        if dg <= 0 or db <= 0:
            dg, db = m.get("fallback_when_neg", [FALLBACK_DELTA_GAMMA, FALLBACK_DELTA_BETA])
        dg = float(np.clip(dg, lo, hi))
        db = float(np.clip(db, lo, hi))
        return dg, db, "linear_ridge_json"
    except Exception as e:
        return None


def _predict_schedule(inst: instances.ProblemInstance) -> tuple[float, float, str]:
    """Predictor preference: GNN checkpoint > linear-ridge JSON > literature default."""
    if MODEL_PATH.exists():
        try:
            import torch
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            from scripts.gnn_warmstart_model import GNNWarmstart, build_graph_tensors  # noqa: WPS433
            model = GNNWarmstart(**ckpt["model_kwargs"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            with torch.no_grad():
                x, edge_index, edge_attr = build_graph_tensors(inst)
                out = model(x, edge_index, edge_attr).squeeze(0).cpu().numpy()
            return float(out[0]), float(out[1]), ckpt.get("sha256", "gnn_torch")
        except Exception as e:
            pass

    lin = _predict_from_linear_json(inst)
    if lin is not None:
        return lin

    return FALLBACK_DELTA_GAMMA, FALLBACK_DELTA_BETA, "fallback_no_model"


def _model_hash() -> str:
    for path in (MODEL_PATH, LINEAR_MODEL_PATH):
        if path.exists():
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return f"{path.name}:{h.hexdigest()[:16]}"
    return "no_model_on_disk"


def _lr_params(p: int, dg: float, db: float) -> list[float]:
    out: list[float] = []
    for k in range(p):
        beta = db * (1.0 - (k + 1) / p)
        gamma = dg * (k + 1) / p
        out.extend([beta, gamma])
    return out


def solve(
    inst: instances.ProblemInstance,
    seed: int = 42,
    p: int = 20,
    alpha: float = 0.25,
    shots: int = 4096,
) -> BenchmarkResult:
    from qiskit import transpile
    from qiskit_aer.primitives import SamplerV2

    Q_full = build_qubo_from_inst(inst)
    Q_unc = _xy_mod._unconstrained_qubo(inst)
    N, K = inst.N, inst.K

    dg, db, model_hash = _predict_schedule(inst)

    qc, params, H = _xy_mod._build_circuit(N, K, p, Q_unc)
    qc_meas = qc.copy(); qc_meas.measure_all()
    transpiled = transpile(qc_meas, basis_gates=["sx", "rz", "cx"], optimization_level=1)
    metrics = gate_metrics.extract(transpiled)
    metrics["t_gate_count"] = 0

    schedule_values = _lr_params(p, dg, db)

    sampler = SamplerV2(default_shots=shots, seed=seed)

    t0 = time.perf_counter()
    bound = transpiled.assign_parameters(schedule_values)
    job = sampler.run([(bound,)], shots=shots)
    counts = job.result()[0].data.meas.get_counts()
    elapsed = time.perf_counter() - t0

    pairs: list[tuple[float, int]] = []
    total = 0
    best_x, best_cost = None, math.inf
    best_any = None; best_any_cnt = -1
    for bitstr, cnt in counts.items():
        bits = [int(b) for b in bitstr[::-1]]
        x = np.array(bits, dtype=int)
        if cnt > best_any_cnt:
            best_any_cnt, best_any = cnt, x.copy()
        c = qubo_eval(Q_full, x)
        pairs.append((c, cnt))
        total += cnt
        if int(x.sum()) == K and c < best_cost:
            best_cost, best_x = float(c), x.copy()

    pairs.sort(key=lambda r: r[0])
    cutoff = max(1, int(alpha * total))
    accumulated = 0; weighted = 0.0
    for c, cnt in pairs:
        take = min(cnt, cutoff - accumulated)
        if take <= 0:
            break
        weighted += c * take
        accumulated += take
        if accumulated >= cutoff:
            break
    cvar = weighted / max(accumulated, 1)

    feasible = best_x is not None
    if not feasible:
        best_x = best_any if best_any is not None else np.zeros(N, dtype=int)
        best_cost = float(qubo_eval(Q_full, best_x))

    if N <= 22:
        _, c_opt, c_wst = exact_reference(Q_full, K)
        ar = approx_ratio(best_cost, c_opt, c_wst)
    else:
        ar = None

    return BenchmarkResult(
        algorithm=ALGORITHM,
        algorithm_version=ALGORITHM_VERSION,
        instance_id=inst.instance_id,
        seed=seed,
        objective_value=float(best_cost),
        bitstring=best_x.tolist(),
        feasible=bool(feasible),
        approx_ratio=ar,
        wall_time_seconds=float(elapsed),
        optimizer_iters=0,
        num_circuit_evaluations=1,
        convergence_history=[float(cvar)],
        qubit_count=int(metrics["qubit_count"]),
        circuit_depth=int(metrics["circuit_depth"]),
        two_qubit_gate_count=int(metrics["two_qubit_gate_count"]),
        t_gate_count=int(metrics["t_gate_count"]),
        total_gate_count=int(metrics["total_gate_count"]),
        shots=int(shots),
        backend="qiskit_aer.AerSimulator",
        transpile_optimization_level=1,
        hyperparameters={
            "problem_variant": "qubo_unconstrained_objective",
            "ansatz": "XY_constrained_QAOA",
            "schedule": "LR-QAOA",
            "Delta_gamma": float(dg),
            "Delta_beta": float(db),
            "p": int(p),
            "objective": "CVaR",
            "alpha": float(alpha),
            "warmstart_source": model_hash,
            "gnn_model_hash": _model_hash(),
            "instance_features": _instance_features(inst),
            "cvar_value": float(cvar),
        },
        initial_params=[float(v) for v in schedule_values],
        final_params=[float(v) for v in schedule_values],
    )
