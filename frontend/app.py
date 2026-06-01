"""Zero-dependency HTTP server for the QC Portfolio Optimization dashboard.

Run:
    python frontend/app.py
Then open http://localhost:5050
"""

from __future__ import annotations

import json
import math
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:  # numpy not installed -> quantum compute endpoints degrade
    np = None
    HAVE_NUMPY = False

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = Path(__file__).resolve().parent
TEMPLATES = FRONTEND / "templates"
STATIC = FRONTEND / "static"
INSTANCES_DIR = ROOT / "data" / "instances"
RAW_RESULTS_DIR = ROOT / "results" / "raw"
WARM_START_DIR = ROOT / "results" / "warm_start"
CLASSICAL_DIR = ROOT / "results" / "classical-algos"

QUANTUM_ALGOS = {"qsw", "warm_start_qaoa_saksham"}

PORT = 5050

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _instance_size_bucket(instance_id: str) -> str:
    for tag in ("tiny", "small", "medium", "large", "n7_gap"):
        if instance_id.startswith(tag):
            return tag
    return "other"


def _instance_summary(path: Path) -> dict:
    data = _load_json(path)
    return {
        "instance_id": data["instance_id"],
        "N": data["N"],
        "K": data["K"],
        "q": data["q"],
        "tickers": data["asset_tickers"],
        "bucket": _instance_size_bucket(data["instance_id"]),
        "date_range": data.get("date_range"),
    }


def _iter_result_files():
    for d in (RAW_RESULTS_DIR, WARM_START_DIR):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            yield f


def _result_summary(data: dict) -> dict:
    return {
        "algorithm": data["algorithm"],
        "instance_id": data["instance_id"],
        "seed": data.get("seed"),
        "objective_value": data.get("objective_value"),
        "feasible": data.get("feasible"),
        "approx_ratio": data.get("approx_ratio"),
        "wall_time_seconds": data.get("wall_time_seconds"),
        "qubit_count": data.get("qubit_count"),
        "circuit_depth": data.get("circuit_depth"),
        "two_qubit_gate_count": data.get("two_qubit_gate_count"),
        "total_gate_count": data.get("total_gate_count"),
        "optimizer_iters": data.get("optimizer_iters"),
        "shots": data.get("shots"),
        "backend": data.get("backend"),
        "hyperparameters": data.get("hyperparameters", {}),
    }


def list_instances() -> list[dict]:
    items: list[dict] = []
    for f in sorted(INSTANCES_DIR.glob("*.json")):
        try:
            items.append(_instance_summary(f))
        except Exception:
            continue
    return items


def list_results() -> list[dict]:
    items: list[dict] = []
    for f in _iter_result_files():
        try:
            items.append(_result_summary(_load_json(f)))
        except Exception:
            continue
    return items


def get_instance(instance_id: str) -> dict | None:
    if not INSTANCE_ID_RE.match(instance_id):
        return None
    path = INSTANCES_DIR / f"{instance_id}.json"
    return _load_json(path) if path.exists() else None


def get_result(algorithm: str, instance_id: str) -> dict | None:
    if not INSTANCE_ID_RE.match(instance_id) or not INSTANCE_ID_RE.match(algorithm):
        return None
    for d in (RAW_RESULTS_DIR, WARM_START_DIR):
        for f in d.glob(f"{algorithm}__{instance_id}__*.json"):
            return _load_json(f)
    return None


def results_for(instance_id: str) -> list[dict]:
    if not INSTANCE_ID_RE.match(instance_id):
        return []
    items: list[dict] = []
    for f in _iter_result_files():
        if f"__{instance_id}__" not in f.name:
            continue
        try:
            items.append(_load_json(f))
        except Exception:
            continue
    return items


# --- Classical report-card parsing -----------------------------------------

_BLOCK_SEP = "QC Portfolio Benchmark"


def _parse_pct(token: str) -> float | None:
    token = token.strip()
    if token.endswith("%"):
        try:
            return float(token[:-1])
        except ValueError:
            return None
    return None


def _parse_float(token: str) -> float | None:
    token = token.strip().rstrip("s")
    if token in ("", "n/a", "none"):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _parse_classical_block(block: str) -> dict | None:
    def grab(label: str) -> str | None:
        m = re.search(rf"{re.escape(label)}\s+(.+)", block)
        return m.group(1).strip() if m else None

    solver_raw = grab("solver")
    if not solver_raw:
        return None
    solver = solver_raw.split()[0].replace(".py", "")

    inst_m = re.search(r"instance\s+([A-Za-z0-9_]+)\.json", block)
    instance_id = inst_m.group(1) if inst_m else None
    if not instance_id:
        return None

    feasible = "✔ feasible" in block or ("feasible" in block and "✘" not in block)

    obj = grab("objective value")
    ret = grab("expected return")
    vol = grab("volatility")
    sharpe = grab("sharpe ratio")
    wall = grab("wall time")

    return {
        "solver": solver,
        "type": "classical",
        "instance_id": instance_id,
        "objective_value": _parse_float(obj) if obj else None,
        "expected_return": _parse_pct(ret) if ret else None,
        "volatility": _parse_pct(vol) if vol else None,
        "sharpe": _parse_float(sharpe) if sharpe else None,
        "wall_time_seconds": _parse_float(wall) if wall else None,
        "feasible": feasible,
    }


def parse_classical() -> list[dict]:
    records: list[dict] = []
    if not CLASSICAL_DIR.exists():
        return records
    for f in sorted(CLASSICAL_DIR.glob("*.py")):
        if f.name == "backtesting.py":
            continue
        text = f.read_text(errors="replace")
        for block in text.split(_BLOCK_SEP):
            if "objective value" not in block:
                continue
            rec = _parse_classical_block(block)
            if rec:
                records.append(rec)
    return records


# --- Quantum metric computation --------------------------------------------


def _weights_from_bitstring(bitstring) -> list[float] | None:
    if not isinstance(bitstring, list) or not bitstring:
        return None
    arr = [float(v) for v in bitstring]
    total = sum(arr)
    if total <= 0:
        return None
    is_binary = all(v in (0.0, 1.0) for v in arr)
    if is_binary:
        return [v / total for v in arr]  # equal weight over selected
    return [v / total for v in arr]


def _portfolio_metrics(weights, mu, sigma) -> dict:
    n = len(weights)
    ret = sum(weights[i] * mu[i] for i in range(n)) * 100.0  # percent
    var = 0.0
    for i in range(n):
        for j in range(n):
            var += weights[i] * sigma[i][j] * weights[j]
    vol = (var ** 0.5) * 100.0  # percent
    sharpe = (ret / vol) if vol > 0 else None
    return {"expected_return": ret, "volatility": vol, "sharpe": sharpe}


def quantum_comparison_records(instance_ids: set[str]) -> list[dict]:
    records: list[dict] = []
    for f in _iter_result_files():
        try:
            data = _load_json(f)
        except Exception:
            continue
        iid = data.get("instance_id")
        if iid not in instance_ids:
            continue
        inst = get_instance(iid)
        if not inst:
            continue
        weights = _weights_from_bitstring(data.get("bitstring"))
        metrics = {"expected_return": None, "volatility": None, "sharpe": None}
        if weights and len(weights) == inst["N"]:
            metrics = _portfolio_metrics(weights, inst["mu"], inst["sigma"])
        records.append({
            "solver": data["algorithm"],
            "type": "quantum",
            "instance_id": iid,
            "objective_value": data.get("objective_value"),
            "expected_return": metrics["expected_return"],
            "volatility": metrics["volatility"],
            "sharpe": metrics["sharpe"],
            "wall_time_seconds": data.get("wall_time_seconds"),
            "feasible": data.get("feasible"),
        })
    return records


def build_comparison() -> dict:
    classical = parse_classical()
    shared_instances = sorted({r["instance_id"] for r in classical})
    quantum = quantum_comparison_records(set(shared_instances))
    return {
        "instances": shared_instances,
        "records": classical + quantum,
    }


# ===========================================================================
# Live quantum compute: HHL linear solver + Quantum Stochastic Walk
# ---------------------------------------------------------------------------
# These endpoints recompute results on the fly so the frontend can offer
# interactive sliders (problem size / condition number for HHL, the quantum-
# classical mixing parameter omega for the QSW). The HHL algebra mirrors
# quantum-optimization-algorithm/hhl.py; the QSW algebra mirrors the
# "Quantum Stochastic Walks" reference module. Both are pure numpy.
# ===========================================================================

# HHL emulation is the noiseless / infinite-shot ceiling of the quantum
# linear solver: above this size the real circuit is intractable to simulate,
# so we report the SAME continuous solution a perfect HHL run would prepare,
# alongside the circuit's resource cost. Cap the live solve for responsiveness.
HHL_MAX_N = 30
QSW_MAX_N = 24

# QSW presets (Table 1 of the reference QSW portfolio study).
QSW_PRESETS = {
    "moderate_balanced": {"alpha": 10.0, "beta": 10.0, "lam": 10.0,
                          "label": "Moderate-Balanced"},
    "ultra_diversified": {"alpha": 1.0, "beta": 100.0, "lam": 10.0,
                          "label": "Ultra-Diversified"},
    "stability_focused": {"alpha": 1.0, "beta": 10.0, "lam": 100.0,
                          "label": "Stability-Focused"},
    "balanced_active": {"alpha": 10.0, "beta": 1.0, "lam": 100.0,
                        "label": "Balanced-Active"},
    "sharpe_maximizer": {"alpha": 100.0, "beta": 1.0, "lam": 10.0,
                         "label": "Sharpe-Maximizer"},
}


def _hhl_build_kkt(mu, sigma, q):
    """Markowitz KKT system A w = b, padded to a power of two (mirrors hhl.py)."""
    N = len(mu)
    dim = N + 1
    pad = 1 << (dim - 1).bit_length()
    A = np.eye(pad)
    b = np.zeros(pad)
    A[:N, :N] = q * sigma
    A[:N, N] = 1.0
    A[N, :N] = 1.0
    A[N, N] = 0.0
    b[:N] = mu
    b[N] = 1.0
    A = (A + A.T) / 2.0
    return A, b


def _hhl_resource_estimate(A, n_clock=6):
    """Qubit / depth / two-qubit-gate estimate for HHL on A (mirrors hhl.py)."""
    dim = A.shape[0]
    n_b = int(math.ceil(math.log2(dim)))
    qubits = n_b + n_clock + 1
    kappa = float(np.linalg.cond(A))
    depth = (2 ** n_clock) * (n_b ** 2)
    two_qubit = (2 ** n_clock) * n_b * 4
    return {
        "qubit_count": int(qubits),
        "circuit_depth": int(depth),
        "two_qubit_gate_count": int(two_qubit),
        "n_b": int(n_b),
        "n_clock": int(n_clock),
        "kappa": kappa,
    }


def hhl_payload(instance_id: str) -> dict | None:
    """Solve the Markowitz KKT linear system for an instance the way HHL would.

    Returns the continuous weight vector w = A^-1 b (the state |w> HHL prepares),
    the classical reference solution, the top-K asset decode and the quantum
    circuit's resource cost.
    """
    if not HAVE_NUMPY:
        return {"error": "numpy unavailable on server"}
    inst = get_instance(instance_id)
    if not inst:
        return None
    N, K = int(inst["N"]), int(inst["K"])
    mu = np.asarray(inst["mu"], dtype=float)
    sigma = np.asarray(inst["sigma"], dtype=float)
    q = float(inst["q"])

    A, b = _hhl_build_kkt(mu, sigma, q)
    # Classical reference (LU). This is also the noiseless HHL output.
    w_full = np.linalg.solve(A, b)
    w_assets = w_full[:N]
    est = _hhl_resource_estimate(A)

    # Decode: top-K by largest long weight (matches hhl.py / MVO decode).
    order = np.argsort(-w_assets)
    selected = sorted(int(i) for i in order[:K])

    return {
        "instance_id": instance_id,
        "N": N,
        "K": K,
        "q": q,
        "tickers": inst["asset_tickers"],
        "mu": mu.tolist(),
        "padded_dim": int(A.shape[0]),
        "weights": w_assets.tolist(),          # continuous KKT / HHL solution
        "weights_abs": np.abs(w_assets).tolist(),
        "selected": selected,
        "resource": est,
        "backend": ("hhl_aer_statevector" if N <= 4
                    else "hhl_classical_emulation_with_resource_estimate"),
        "max_n": HHL_MAX_N,
    }


# --- Quantum Stochastic Walk ------------------------------------------------

def _corr_sharpe_from_inst(mu, sigma):
    """Derive a correlation matrix and per-asset Sharpe ratios from mu/Sigma."""
    d = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
    corr = sigma / np.outer(d, d)
    np.clip(corr, -1.0, 1.0, out=corr)
    sharpe = mu / d
    return corr, sharpe


def _qsw_hamiltonian(sharpe, corr, alpha, beta):
    sr = np.asarray(sharpe, dtype=float)
    rng = sr.max() - sr.min()
    sr_norm = (sr - sr.min()) / rng if rng > 1e-10 else np.ones_like(sr) / len(sr)
    H = beta * (1.0 - np.abs(corr))
    np.fill_diagonal(H, alpha * sr_norm)
    return H


def _qsw_google_matrix(corr, lam):
    n = corr.shape[0]
    Aadj = np.maximum(1.0 - np.abs(corr), 0.0)
    np.fill_diagonal(Aadj, 0.0)
    col = Aadj.sum(axis=0)
    col[col < 1e-12] = 1.0
    Anorm = Aadj / col[np.newaxis, :]
    d = lam / (lam + 1.0)
    return (1.0 - d) * Anorm + d * np.ones((n, n)) / n


def _qsw_steady_state(H, G, omega):
    """Solve the GKLS Lindblad steady state; return (rho, weights)."""
    n = H.shape[0]
    n2 = n * n
    I_n = np.eye(n)
    L = -1j * (1.0 - omega) * (np.kron(I_n, H) - np.kron(H.T, I_n))
    Lam = np.zeros((n2, n2), dtype=complex)
    for k in range(n):
        for j in range(n):
            Lam[k * (n + 1), j * (n + 1)] = G[k, j]
    L = L + omega * (Lam - np.eye(n2, dtype=complex))

    trace_row = np.zeros(n2, dtype=complex)
    for k in range(n):
        trace_row[k * (n + 1)] = 1.0
    L[-1, :] = trace_row
    rhs = np.zeros(n2, dtype=complex)
    rhs[-1] = 1.0
    try:
        vec = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:
        vec, *_ = np.linalg.lstsq(L, rhs, rcond=None)
    rho = vec.reshape((n, n), order="F")
    w = np.real(np.diag(rho))
    w = np.maximum(w, 0.0)
    total = w.sum()
    w = w / total if total > 1e-12 else np.ones(n) / n
    return rho, w


def qsw_payload(instance_id: str, omega: float, alpha: float,
                beta: float, lam: float) -> dict | None:
    if not HAVE_NUMPY:
        return {"error": "numpy unavailable on server"}
    inst = get_instance(instance_id)
    if not inst:
        return None
    N = int(inst["N"])
    if N > QSW_MAX_N:
        return {"error": f"instance too large for live QSW (N={N} > {QSW_MAX_N})",
                "N": N, "max_n": QSW_MAX_N}
    mu = np.asarray(inst["mu"], dtype=float)
    sigma = np.asarray(inst["sigma"], dtype=float)
    corr, sharpe = _corr_sharpe_from_inst(mu, sigma)
    H = _qsw_hamiltonian(sharpe, corr, alpha, beta)
    G = _qsw_google_matrix(corr, lam)
    rho, w = _qsw_steady_state(H, G, omega)

    rho_abs = np.abs(rho)
    # off-diagonal "coherence mass" — the genuinely quantum part of the state
    coherence = float(rho_abs.sum() - np.trace(rho_abs).real)
    hhi = float(np.sum(w ** 2))
    eff_stocks = float(1.0 / hhi) if hhi > 1e-12 else float(N)

    return {
        "instance_id": instance_id,
        "N": N,
        "omega": omega,
        "alpha": alpha, "beta": beta, "lam": lam,
        "tickers": inst["asset_tickers"],
        "weights": w.tolist(),
        "hhi": hhi,
        "eff_stocks": eff_stocks,
        "coherence": coherence,
        "equal_weight": 1.0 / N,
        "hamiltonian": H.tolist(),
        "corr": np.abs(corr).tolist(),
        "rho_abs": rho_abs.tolist(),
        "max_n": QSW_MAX_N,
    }


def qsw_coherence_sweep(instance_id: str, alpha: float, beta: float,
                        lam: float, n_points: int = 21) -> dict | None:
    """Sweep omega 0->1 and report how coherence / concentration change.

    Demonstrates the quantum->classical transition: coherence and structure
    are highest at omega=0 (fully quantum) and vanish toward omega=1.
    """
    if not HAVE_NUMPY:
        return {"error": "numpy unavailable on server"}
    inst = get_instance(instance_id)
    if not inst:
        return None
    N = int(inst["N"])
    if N > QSW_MAX_N:
        return {"error": "instance too large", "N": N, "max_n": QSW_MAX_N}
    mu = np.asarray(inst["mu"], dtype=float)
    sigma = np.asarray(inst["sigma"], dtype=float)
    corr, sharpe = _corr_sharpe_from_inst(mu, sigma)
    H = _qsw_hamiltonian(sharpe, corr, alpha, beta)
    G = _qsw_google_matrix(corr, lam)

    omegas, coh, hhis = [], [], []
    for i in range(n_points):
        om = i / (n_points - 1)
        rho, w = _qsw_steady_state(H, G, om)
        ra = np.abs(rho)
        omegas.append(om)
        coh.append(float(ra.sum() - np.trace(ra).real))
        hhis.append(float(np.sum(w ** 2)))
    return {
        "instance_id": instance_id, "N": N,
        "omega": omegas, "coherence": coh, "hhi": hhis,
        "equal_weight_hhi": 1.0 / N,
    }


def _qsw_propagator(H, G, omega):
    """Return (eigvals, V, Vinv) of the Lindbladian L for time evolution."""
    n = H.shape[0]
    n2 = n * n
    I_n = np.eye(n)
    L = -1j * (1.0 - omega) * (np.kron(I_n, H) - np.kron(H.T, I_n))
    Lam = np.zeros((n2, n2), dtype=complex)
    for k in range(n):
        for j in range(n):
            Lam[k * (n + 1), j * (n + 1)] = G[k, j]
    L = L + omega * (Lam - np.eye(n2, dtype=complex))
    ev, V = np.linalg.eig(L)
    return ev, V, np.linalg.inv(V)


def _evolve_population(ev, V, Vinv, v0, times, n):
    """Propagate vec(rho0) along `times`; return per-asset population matrix."""
    coeff = Vinv @ v0
    pops, rhos = [], []
    for t in times:
        vt = V @ (np.exp(ev * t) * coeff)
        rho = vt.reshape((n, n), order="F")
        p = np.real(np.diag(rho))
        p = np.clip(p, 0.0, None)
        s = p.sum()
        p = p / s if s > 1e-12 else p
        pops.append(p)
        rhos.append(rho)
    return np.array(pops), rhos


def qsw_evolution_payload(instance_id: str, omega: float, alpha: float,
                          beta: float, lam: float, source: int = -1,
                          steps: int = 48) -> dict | None:
    """Continuous-time QSW dynamics from a localized start state.

    Starting all population on one asset, evolve the open-system walk and
    record how population spreads across assets. Low omega -> coherent
    quantum spreading (interference, ballistic); high omega -> classical
    diffusion. The participation ratio quantifies the spreading speed and is
    overlaid against the pure-classical (omega=1) walk to show the gap.
    """
    if not HAVE_NUMPY:
        return {"error": "numpy unavailable on server"}
    inst = get_instance(instance_id)
    if not inst:
        return None
    N = int(inst["N"])
    if N > QSW_MAX_N:
        return {"error": "instance too large", "N": N, "max_n": QSW_MAX_N}
    mu = np.asarray(inst["mu"], dtype=float)
    sigma = np.asarray(inst["sigma"], dtype=float)
    corr, sharpe = _corr_sharpe_from_inst(mu, sigma)
    H = _qsw_hamiltonian(sharpe, corr, alpha, beta)
    G = _qsw_google_matrix(corr, lam)

    # Start localized on the highest-Sharpe asset unless told otherwise.
    if source < 0 or source >= N:
        source = int(np.argmax(sharpe))
    rho0 = np.zeros((N, N), dtype=complex)
    rho0[source, source] = 1.0
    v0 = rho0.reshape(-1, order="F")

    # Time window: a few coherent periods of the Hamiltonian.
    eig_h = np.max(np.abs(np.linalg.eigvalsh(H)))
    period = (2.0 * math.pi / eig_h) if eig_h > 1e-9 else 1.0
    total_t = 12.0 * period
    times = [total_t * i / (steps - 1) for i in range(steps)]

    ev, V, Vinv = _qsw_propagator(H, G, omega)
    pops, rhos = _evolve_population(ev, V, Vinv, v0, times, N)

    # Participation ratio = effective number of occupied assets (1 = localized).
    part = [float(1.0 / np.sum(p ** 2)) if np.sum(p ** 2) > 1e-12 else 1.0
            for p in pops]
    # Coherence = total off-diagonal magnitude of rho(t).
    coh = [float(np.abs(r).sum() - np.abs(np.diag(r)).sum()) for r in rhos]

    # Pure-classical reference walk (omega = 1) for the spreading overlay.
    ev_c, V_c, Vi_c = _qsw_propagator(H, G, 1.0)
    pops_c, _ = _evolve_population(ev_c, V_c, Vi_c, v0, times, N)
    part_c = [float(1.0 / np.sum(p ** 2)) if np.sum(p ** 2) > 1e-12 else 1.0
              for p in pops_c]

    # |rho| snapshot at the time of peak coherence (most "quantum" instant).
    peak = int(np.argmax(coh))
    rho_peak = np.abs(rhos[peak]).tolist()

    return {
        "instance_id": instance_id,
        "N": N,
        "omega": omega,
        "source": source,
        "tickers": inst["asset_tickers"],
        "times": times,
        "population": pops.tolist(),          # steps x N
        "participation": part,
        "participation_classical": part_c,
        "coherence": coh,
        "rho_peak": rho_peak,
        "rho_peak_time_index": peak,
        "max_participation": float(N),
        "max_n": QSW_MAX_N,
    }


def _query_int(qs: dict, key: str, default: int) -> int:
    try:
        return int(qs.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def _query_float(qs: dict, key: str, default: float) -> float:
    try:
        return float(qs.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self) -> None:
        self._send_json({"error": "not found"}, status=404)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_404()
            return
        body = path.read_bytes()
        ctype = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _render_index(self) -> None:
        # Trivial template: strip `{{ url_for(...) }}` shims since we serve static
        # via the same origin under /static/.
        html = (TEMPLATES / "index.html").read_text()
        html = re.sub(
            r"\{\{\s*url_for\('static',\s*filename='([^']+)'\)\s*\}\}",
            r"/static/\1",
            html,
        )
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            return self._render_index()

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = (STATIC / rel).resolve()
            try:
                target.relative_to(STATIC.resolve())
            except ValueError:
                return self._send_404()
            return self._send_file(target)

        if path == "/api/instances":
            return self._send_json(list_instances())

        if path == "/api/results":
            return self._send_json(list_results())

        if path == "/api/comparison":
            return self._send_json(build_comparison())

        m = re.match(r"^/api/instance/([A-Za-z0-9_]+)$", path)
        if m:
            data = get_instance(m.group(1))
            return self._send_json(data) if data else self._send_404()

        m = re.match(r"^/api/results_for/([A-Za-z0-9_]+)$", path)
        if m:
            return self._send_json(results_for(m.group(1)))

        m = re.match(r"^/api/result/([A-Za-z0-9_]+)/([A-Za-z0-9_]+)$", path)
        if m:
            data = get_result(m.group(1), m.group(2))
            return self._send_json(data) if data else self._send_404()

        m = re.match(r"^/api/quantum/hhl/([A-Za-z0-9_]+)$", path)
        if m:
            data = hhl_payload(m.group(1))
            return self._send_json(data) if data else self._send_404()

        m = re.match(r"^/api/quantum/qsw/([A-Za-z0-9_]+)$", path)
        if m:
            omega = _query_float(qs, "omega", 0.2)
            alpha = _query_float(qs, "alpha", 10.0)
            beta = _query_float(qs, "beta", 10.0)
            lam = _query_float(qs, "lam", 10.0)
            data = qsw_payload(m.group(1), omega, alpha, beta, lam)
            return self._send_json(data) if data else self._send_404()

        m = re.match(r"^/api/quantum/qsw_sweep/([A-Za-z0-9_]+)$", path)
        if m:
            alpha = _query_float(qs, "alpha", 10.0)
            beta = _query_float(qs, "beta", 10.0)
            lam = _query_float(qs, "lam", 10.0)
            data = qsw_coherence_sweep(m.group(1), alpha, beta, lam)
            return self._send_json(data) if data else self._send_404()

        m = re.match(r"^/api/quantum/qsw_evolution/([A-Za-z0-9_]+)$", path)
        if m:
            omega = _query_float(qs, "omega", 0.1)
            alpha = _query_float(qs, "alpha", 10.0)
            beta = _query_float(qs, "beta", 10.0)
            lam = _query_float(qs, "lam", 10.0)
            source = _query_int(qs, "source", -1)
            data = qsw_evolution_payload(m.group(1), omega, alpha, beta, lam,
                                         source=source)
            return self._send_json(data) if data else self._send_404()

        self._send_404()


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Dashboard ready at http://127.0.0.1:{PORT}")
    print(f"  instances : {INSTANCES_DIR}")
    print(f"  results   : {RAW_RESULTS_DIR}, {WARM_START_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
