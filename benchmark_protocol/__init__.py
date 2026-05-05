"""Benchmark protocol — single source of truth for solver evaluation."""
from benchmark_protocol.result_schema import BenchmarkResult, validate, SCHEMA_VERSION
from benchmark_protocol.instances import ProblemInstance

__all__ = ["BenchmarkResult", "validate", "SCHEMA_VERSION", "ProblemInstance"]
