"""Opt-in native R3/R8 teacher and solver for the audited V2.1 finite-surface release.

The implementation, native build receipt and frozen physics profile are
distributed together in the versioned experiment. Existing untyped callers
can continue to pass the service's solver method as their Newton callback.
"""
from experiments.compact_pino_v21.stratum.shared_operator import SharedStratumOperator
from .v21_native import RawChart

__all__ = ['RawChart','SharedStratumOperator']
