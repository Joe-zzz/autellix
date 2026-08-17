# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PhaseTimer (experiment instrumentation).

Deliberately stdlib-only (no torch import) so these run on a CPU-only box
without the compiled vLLM stack.
"""

from __future__ import annotations

import pytest

from vllm.v1.metrics.phase_timing import PhaseTimer, duty_cycle

MS = 1e-3


class TestEmptyTimer:
    def test_summary_is_none_when_no_samples(self):
        assert PhaseTimer("empty").summary() is None

    def test_should_not_emit_when_no_samples(self):
        assert PhaseTimer("empty", emit_every=1).should_emit() is False


class TestBasicStats:
    def test_count_and_total(self):
        t = PhaseTimer("p")
        t.record(1 * MS)
        t.record(3 * MS)
        s = t.summary()
        assert s["count"] == 2
        assert s["total_s"] == pytest.approx(4 * MS)

    def test_mean_is_reported_in_ms(self):
        t = PhaseTimer("p")
        t.record(1 * MS)
        t.record(3 * MS)
        assert t.summary()["mean_ms"] == pytest.approx(2.0)

    def test_max_is_reported_in_ms(self):
        t = PhaseTimer("p")
        for v in (1, 7, 3):
            t.record(v * MS)
        assert t.summary()["max_ms"] == pytest.approx(7.0)

    def test_negative_duration_is_clamped_to_zero(self):
        # A monotonic-clock hiccup must not produce a negative total.
        t = PhaseTimer("p")
        t.record(-5 * MS)
        assert t.summary()["total_s"] == 0.0


class TestPercentiles:
    """Nearest-rank: index = ceil(p/100 * n) - 1 over the sorted samples."""

    def test_p50_of_ten_samples(self):
        t = PhaseTimer("p")
        for v in range(1, 11):  # 1..10 ms
            t.record(v * MS)
        assert t.summary()["p50_ms"] == pytest.approx(5.0)

    def test_p95_and_p99_pick_the_top_sample_for_small_n(self):
        t = PhaseTimer("p")
        for v in range(1, 11):
            t.record(v * MS)
        s = t.summary()
        assert s["p95_ms"] == pytest.approx(10.0)
        assert s["p99_ms"] == pytest.approx(10.0)

    def test_single_sample_all_percentiles_equal(self):
        t = PhaseTimer("p")
        t.record(4 * MS)
        s = t.summary()
        assert s["p50_ms"] == pytest.approx(4.0)
        assert s["p95_ms"] == pytest.approx(4.0)
        assert s["max_ms"] == pytest.approx(4.0)

    def test_insertion_order_does_not_affect_percentiles(self):
        ordered, shuffled = PhaseTimer("a"), PhaseTimer("b")
        for v in (1, 2, 3, 4, 5):
            ordered.record(v * MS)
        for v in (4, 1, 5, 2, 3):
            shuffled.record(v * MS)
        assert ordered.summary()["p50_ms"] == shuffled.summary()["p50_ms"]


class TestWeighted:
    def test_per_unit_cost_reported_in_microseconds(self):
        # 10 ms moving 100 blocks -> 100 us per block.
        t = PhaseTimer("swap")
        t.record(10 * MS, weight=100)
        s = t.summary()
        assert s["total_weight"] == 100
        assert s["per_unit_us"] == pytest.approx(100.0)

    def test_per_unit_aggregates_across_samples(self):
        t = PhaseTimer("swap")
        t.record(10 * MS, weight=100)
        t.record(30 * MS, weight=100)
        # 40 ms / 200 blocks = 200 us per block
        assert t.summary()["per_unit_us"] == pytest.approx(200.0)

    def test_unweighted_summary_omits_per_unit(self):
        t = PhaseTimer("p")
        t.record(1 * MS)
        assert "per_unit_us" not in t.summary()

    def test_zero_total_weight_omits_per_unit(self):
        t = PhaseTimer("p")
        t.record(1 * MS, weight=0)
        assert "per_unit_us" not in t.summary()


class TestEmitCadence:
    def test_should_emit_only_on_the_boundary(self):
        t = PhaseTimer("p", emit_every=3)
        t.record(MS)
        assert t.should_emit() is False
        t.record(MS)
        assert t.should_emit() is False
        t.record(MS)
        assert t.should_emit() is True

    def test_reset_clears_samples_and_cadence(self):
        t = PhaseTimer("p", emit_every=2)
        t.record(MS)
        t.record(MS)
        assert t.should_emit() is True
        t.reset()
        assert t.summary() is None
        assert t.should_emit() is False

    def test_windows_are_independent_after_reset(self):
        t = PhaseTimer("p", emit_every=2)
        t.record(10 * MS)
        t.record(10 * MS)
        t.reset()
        t.record(2 * MS)
        assert t.summary()["mean_ms"] == pytest.approx(2.0)


class TestDutyCycle:
    def test_ratio_of_totals(self):
        policy, interval = PhaseTimer("policy"), PhaseTimer("interval")
        policy.record(1 * MS)
        policy.record(1 * MS)
        interval.record(10 * MS)
        interval.record(10 * MS)
        assert duty_cycle(policy, interval) == pytest.approx(0.1)

    def test_none_when_denominator_empty(self):
        policy = PhaseTimer("policy")
        policy.record(MS)
        assert duty_cycle(policy, PhaseTimer("interval")) is None

    def test_none_when_denominator_total_is_zero(self):
        policy, interval = PhaseTimer("policy"), PhaseTimer("interval")
        policy.record(MS)
        interval.record(0.0)
        assert duty_cycle(policy, interval) is None
