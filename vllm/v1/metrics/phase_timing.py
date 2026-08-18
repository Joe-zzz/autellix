# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight phase timing for experiment instrumentation.

The measurement that motivated this module is the **scheduler duty cycle**,
on the hot path: under async scheduling the policy's Python work overlaps GPU
execution, so it is free until it approaches the step time. Timing the policy
phase against the inter-``schedule()`` interval gives the headroom directly
(see :func:`duty_cycle`).

Samples carry an optional weight so a timer can report cost per unit of work
rather than per event -- e.g. per KV block rather than per call -- which is what
makes a measurement extrapolate across context lengths and model sizes.

Because this instruments the very overhead it measures, it stays cheap: a
``perf_counter`` pair and a list append per record, with percentiles computed
only when a window is emitted. Windows are bounded by construction --
``reset()`` after each emit -- so memory does not grow.

Stdlib-only on purpose, so it is unit-testable on a CPU-only box without the
compiled vLLM stack.
"""

from __future__ import annotations

import math

_US_PER_S = 1e6
_MS_PER_S = 1e3


def _nearest_rank(sorted_samples: list[float], pct: float) -> float:
    """Nearest-rank percentile: ``index = ceil(pct/100 * n) - 1``.

    Chosen over interpolation because it always returns an observed sample,
    which keeps the reported numbers directly attributable to a real step.
    """
    n = len(sorted_samples)
    idx = math.ceil(pct / 100.0 * n) - 1
    return sorted_samples[min(max(idx, 0), n - 1)]


class PhaseTimer:
    """Accumulate durations for one named phase over a bounded window.

    Attributes:
        name: Identifier used when the summary is logged.
        emit_every: Window size; :meth:`should_emit` turns true once this many
            samples have been recorded since the last :meth:`reset`.
    """

    def __init__(self, name: str, emit_every: int = 1000) -> None:
        self.name = name
        self.emit_every = emit_every
        self._samples: list[float] = []
        self._total_weight: int = 0

    def record(self, seconds: float, weight: int = 0) -> None:
        """Add one observation.

        Args:
            seconds: Elapsed time. Negative values (monotonic-clock hiccups,
                suspended processes) are clamped to zero rather than corrupting
                the totals.
            weight: Optional payload size for this sample -- e.g. the number of
                KV blocks copied -- used to derive a per-unit cost.
        """
        self._samples.append(max(seconds, 0.0))
        self._total_weight += weight

    def should_emit(self) -> bool:
        """Whether a full window has accumulated since the last reset."""
        return len(self._samples) >= self.emit_every > 0

    def total_seconds(self) -> float:
        return math.fsum(self._samples)

    def summary(self) -> dict[str, float] | None:
        """Window statistics, or ``None`` if nothing has been recorded.

        ``per_unit_us`` is present only when weighted samples were recorded:
        total time divided by total weight, e.g. cost per KV block.
        """
        if not self._samples:
            return None

        ordered = sorted(self._samples)
        total = math.fsum(ordered)
        count = len(ordered)

        out: dict[str, float] = {
            "count": count,
            "total_s": total,
            "mean_ms": total / count * _MS_PER_S,
            "p50_ms": _nearest_rank(ordered, 50) * _MS_PER_S,
            "p95_ms": _nearest_rank(ordered, 95) * _MS_PER_S,
            "p99_ms": _nearest_rank(ordered, 99) * _MS_PER_S,
            "max_ms": ordered[-1] * _MS_PER_S,
        }
        if self._total_weight > 0:
            out["total_weight"] = self._total_weight
            out["per_unit_us"] = total / self._total_weight * _US_PER_S
        return out

    def reset(self) -> None:
        """Drop the current window. Call after emitting to bound memory."""
        self._samples.clear()
        self._total_weight = 0


def duty_cycle(numerator: PhaseTimer, denominator: PhaseTimer) -> float | None:
    """Fraction of the denominator's wall time spent in the numerator phase.

    Both timers must cover the same window. For the scheduler this is
    ``policy_time / inter-schedule interval``: near zero means the policy is
    fully hidden by async-scheduling overlap and optimising it buys nothing;
    approaching one means the policy has become the step's critical path.

    Returns:
        The ratio, or ``None`` if the denominator recorded no time.
    """
    denom = denominator.total_seconds()
    if denom <= 0.0:
        return None
    return numerator.total_seconds() / denom
