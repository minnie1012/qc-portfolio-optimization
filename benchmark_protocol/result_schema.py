"""Strict result schema. Every benchmark run produces one BenchmarkResult JSON."""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass
class BenchmarkResult:
    # identity
    algorithm: str
    algorithm_version: str
    instance_id: str
    seed: int

    # quality
    objective_value: float
    bitstring: list[int]
    feasible: bool
    approx_ratio: float | None  # null if exact unknown

    # runtime
    wall_time_seconds: float
    optimizer_iters: int | None
    num_circuit_evaluations: int | None
    convergence_history: list[float]

    # quantum-only (None for classical)
    qubit_count: int | None
    circuit_depth: int | None
    two_qubit_gate_count: int | None
    t_gate_count: int | None
    total_gate_count: int | None
    shots: int | str | None  # int, or "statevector"
    backend: str | None
    transpile_optimization_level: int | None

    # parameters used
    hyperparameters: dict[str, Any]
    initial_params: list[float] | None
    final_params: list[float] | None

    # bookkeeping
    env: dict[str, str] = field(default_factory=dict)
    timestamp_utc: str = ""
    git_commit: str = "n/a"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()
        if not self.env:
            self.env = default_env()

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=_json_default)
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkResult":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


REQUIRED_QUANTUM_FIELDS = (
    "qubit_count",
    "circuit_depth",
    "two_qubit_gate_count",
    "t_gate_count",
    "total_gate_count",
    "shots",
    "backend",
)


def validate(result: BenchmarkResult) -> list[str]:
    """Return a list of error strings; empty list means valid."""
    errors: list[str] = []

    if result.schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version={result.schema_version} != {SCHEMA_VERSION}")

    if result.feasible and len(result.bitstring) == 0:
        errors.append("feasible=True but bitstring empty")

    if result.approx_ratio is not None and result.approx_ratio < 0:
        errors.append(f"negative approx_ratio={result.approx_ratio}")

    if result.wall_time_seconds < 0:
        errors.append("negative wall_time_seconds")

    is_quantum = result.qubit_count is not None
    if is_quantum:
        for fld in REQUIRED_QUANTUM_FIELDS:
            if getattr(result, fld) is None:
                errors.append(f"quantum result missing {fld}")

    if not result.env:
        errors.append("env is empty")

    return errors


def default_env() -> dict[str, str]:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("qiskit", "numpy", "scipy", "qiskit_aer"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return env


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except ImportError:
        pass
    raise TypeError(f"not serializable: {type(obj)}")
