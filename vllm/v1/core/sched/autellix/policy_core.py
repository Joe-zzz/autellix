# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared quantum / demotion / promotion machinery for the Autellix schedulers.

Algorithm 1 of the paper runs the same per-call mechanics for every policy
(MLFQ baseline, PLAS, ATLAS): each call holds a discretized priority level with
a per-level quantum in **seconds of service** (D3); exhausting the quantum
demotes the call one level; a waiting call accrues wait, and once its
wait-to-service ratio reaches ``beta`` it is promoted back to the top level with
its *call-level* windows reset; and when the batch is full a strictly-better
waiting call proactively preempts the worst running call (bounded per step,
recompute-based). Attained service is measured as engine wall time: each
scheduled step charges its measured duration to every co-scheduled call,
prefill chunks included (the paper's execution-time definition, ``e_i``).

The policies differ only in the *windows* fed to the anti-starvation ratio:

* MLFQ (FastServe baseline) uses the call-level windows alone
  (``W_c / max(1, T_c)``) -- the paper's deliberately "naive" variant.
* PLAS / ATLAS use **program-level** starvation (paper §4.2.2):
  ``(W_p + W_c) / max(1, T_p + T_c)``, where ``W_p``/``T_p`` come from the
  process table (PLAS: service sum; ATLAS: critical-path max). Only the
  call-level windows reset on promotion, so a program's calls promote together
  and the program cannot be re-starved by its own promotion.

Priority encoding (shared by all three schedulers): ``request.priority`` is the
call's current queue level, an int in ``[0, K-1]`` with 0 the top queue, so
``Request.__lt__``'s int-priority semantics are untouched. The native
``PriorityRequestQueue`` orders by ``(priority, arrival_time, request_id)`` --
level primary, FCFS within a level. Program attained service influences only
the *arrival* level (PLAS/ATLAS bin the program scalar); demotion and promotion
then move the level in place, exactly as Algorithm 1 moves calls between
queues.

This module hosts the mechanics as a mixin over ``AsyncScheduler`` subclasses;
the schedulers keep their own arrival binning, program folding, and state
release. It relies on base-scheduler attributes (``waiting``,
``skipped_waiting``, ``running``, ``requests``, ``max_num_running_reqs``,
``_preempt_request``) and must precede ``AsyncScheduler`` in the MRO.
"""

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.sched.autellix.mlfq import MlfqBinner
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.phase_timing import PhaseTimer, duty_cycle
from vllm.v1.request import Request

logger = init_logger(__name__)

# Steps per instrumentation window. Large enough that the emit cost is
# negligible against the per-step work it measures.
_TIMER_WINDOW = 1000

_QUANTA_ENV = "AUTELLIX_QUEUE_QUANTA"
_AUTO_QUANTA_ENV = "AUTELLIX_AUTO_QUANTA"
# Calibration samples over a TIME window rather than a fixed count. A count-based
# window is self-defeating on a fast engine: 500 samples filled within seconds of
# load starting on the 8B PRM and produced p25=0.017s, six times below the 0.1s
# that was validated by hand, because it captured only the startup transient --
# small batches, no queueing, calls running unrepresentatively fast. A time
# window is also self-scaling across engines: a fast one contributes many samples
# and a slow one few, but both describe the same interval of real load.
# The two halves are constrained by different things, so they are sized
# differently. The skip must outlast the startup transient, whose duration is not
# known: at ~6s in the transient was still strong (p25=0.017s against a
# steady-state 0.066s) and 60s was clean, with nothing measured in between. Its
# failure mode is also silent -- calibrating onto the transient yields quanta too
# small, which over-demotes and degenerates to FCFS without showing up in
# throughput or latency -- so it stays conservative. The window only has to
# collect enough samples for a quantile, and is heavily over-provisioned at 60s:
# the 8B PRM completes ~4681 calls per 60s, where a few hundred suffice. Halving
# it costs no precision and keeps the total inside the shortest warmup that
# precedes measurement.
_AUTO_QUANTA_SKIP_S = float(os.getenv("AUTELLIX_AUTO_QUANTA_SKIP_S", "60"))
_AUTO_QUANTA_WINDOW_S = float(os.getenv("AUTELLIX_AUTO_QUANTA_WINDOW_S", "30"))
# Floor on samples before trusting the quantile, in case an engine is slow enough
# that the window yields very few completions.
_AUTO_QUANTA_MIN_SAMPLES = 50
# Q1 lands at this quantile of observed call service, and the ladder doubles from
# there. Calibrated by hand first: the engines' typical calls measured ~0.3 s
# (8B PRM), ~2 s (1B generation) and ~16-32 s (RAG), and ladders starting near a
# quarter of those put every engine into a healthy spread, where both a ladder
# 3x too high (zero demotions, MLFQ inert) and one 20x too low (calls sink to the
# bottom level immediately) had failed.
_AUTO_QUANTA_QUANTILE = 0.25
_MIN_QUANTUM_S = 1e-3


def quanta_from_env(default: tuple[float, ...]) -> tuple[float, ...]:
    """Read the per-queue quantum ladder from the environment, else ``default``.

    The ladder has to be scaled to the workload: measured on beam-search, the 1B
    generation engine put 84287 of 155425 completed calls under the first
    1-second quantum while the 8B PRM put *all* 124025 of its calls there and was
    never demoted, leaving PLAS, MLFQ and FCFS equivalent on that engine. RAG's
    generation model sits at the other extreme, with its first two levels empty.
    One ladder cannot serve all three, and the paper fixes the MLFQ structure
    without stating any numeric quanta, so these are calibration rather than
    fidelity.

    An environment variable rather than a CLI flag because vLLM instantiates the
    scheduler with a fixed kwargs set (``v1/engine/core.py:150``) and offers no
    way to pass constructor arguments through ``--scheduler-cls``. Per-model
    values are then reachable via ``ModelConfig.with_env``.

    Args:
        default: Ladder to use when the variable is unset or empty.

    Returns:
        The parsed ladder, or ``default``.

    Raises:
        ValueError: If the variable is set but not a comma-separated list of
            positive, strictly increasing floats of the same length as
            ``default`` -- failing loudly beats silently scheduling on a ladder
            the caller did not intend.
    """
    raw = os.getenv(_QUANTA_ENV, "").strip()
    if not raw:
        return default
    try:
        values = tuple(float(x) for x in raw.split(","))
    except ValueError as exc:
        raise ValueError(
            f"{_QUANTA_ENV}={raw!r} is not a comma-separated float list"
        ) from exc
    if len(values) != len(default):
        raise ValueError(
            f"{_QUANTA_ENV} must have {len(default)} entries, got {len(values)}"
        )
    if any(v <= 0 for v in values):
        raise ValueError(f"{_QUANTA_ENV} entries must be positive, got {values}")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{_QUANTA_ENV} must be strictly increasing, got {values}")
    return values


@dataclass
class CallQueueState:
    """Mutable per-call queue bookkeeping, keyed by ``request_id``.

    All windows are in **seconds of engine wall time** (D3): each scheduled
    step charges its measured duration, so quanta and windows are robust to
    variable step durations (long prefill steps, big batches, MPS interference)
    and transfer across token lengths.

    Attributes:
        queue_index: The call's current queue; equals ``request.priority``
            (queue 0 is Q1, the highest priority).
        quantum_remaining: Seconds of service left before the call is demoted a
            level.
        wait_window: Seconds the call has waited (``W_c``); reset to 0 on
            anti-starvation promotion.
        service_window: Seconds the call has been served (``T_c``); reset to 0
            on anti-starvation promotion.
    """

    queue_index: int
    quantum_remaining: float
    wait_window: float
    service_window: float
    # Lifetime service, unlike service_window which anti-starvation promotion
    # resets. Used only to histogram completed calls against the quantum ladder.
    total_service: float = 0.0


class QuantumMlfqMixin:
    """Per-call quantum demotion, beta promotion, and proactive preemption.

    Mix into an ``AsyncScheduler`` subclass *before* the base class. The host
    scheduler must call :meth:`_init_policy_core` from its ``__init__`` and
    :meth:`_register_call_state` for every genuinely new call, and should
    release state via :meth:`_release_call_state` (or pop entries itself) on
    completion/abort. Hooks:

    * :meth:`_program_windows` supplies the program-level ``(W_p, T_p)`` added
      to the call windows in the anti-starvation ratio (default ``(0, 0)`` --
      call-level-only, the MLFQ baseline).
    * :meth:`_on_service_step` is called once per charged decode step so
      program-aware hosts can accrue attained service without a second pass.
    """

    def _init_policy_core(
        self,
        num_queues: int,
        queue_quanta: tuple[float, ...],
        beta: float,
        max_proactive_preemptions_per_step: int,
        binner: MlfqBinner,
    ) -> None:
        """Install the shared constants and per-call state map.

        Args:
            num_queues: Number of feedback queues ``K``.
            queue_quanta: Per-queue quantum in **seconds of service**; must have
                length ``num_queues``.
            beta: Anti-starvation wait-to-service ratio threshold.
            max_proactive_preemptions_per_step: Upper bound on proactive
                preemptions per ``schedule()`` call (0 disables it).
            binner: The binner providing demote / anti-starvation / outranks.

        Raises:
            ValueError: If ``len(queue_quanta) != num_queues``.
        """
        if len(queue_quanta) != num_queues:
            raise ValueError(
                f"queue_quanta must have length num_queues = {num_queues}, "
                f"got {len(queue_quanta)}"
            )
        self.num_queues = num_queues
        self.queue_quanta = tuple(queue_quanta)
        self.beta = beta
        self.max_proactive_preemptions_per_step = max_proactive_preemptions_per_step
        self.binner = binner
        self._call_state: dict[str, CallQueueState] = {}
        self.proactive_preemption_count = 0
        # Every preemption, whichever path triggered it (see _preempt_request).
        # Cumulative and never reset, so the last emitted line is the run total.
        self.preemption_count = 0
        # MLFQ activity. The queue structure only does work if calls actually
        # move between levels: if quanta are far larger than a typical call's
        # service time nothing is ever demoted, and if they are far smaller
        # everything sinks to the bottom queue immediately -- both degenerate to
        # FCFS with extra bookkeeping. The paper fixes the structure but states
        # no numeric quanta, so these counters are how we tell whether the
        # values we inherited are calibrated for this workload at all.
        self.demotion_count = 0
        self.promotion_count = 0
        # Completed calls bucketed against the quantum ladder, to answer what
        # the quanta should be rather than guessing. Bucket i counts calls whose
        # total service fell in [ladder[i-1], ladder[i]); the last bucket is the
        # overflow past the final quantum. If nearly everything lands in bucket
        # 0 the ladder sits above the workload and the MLFQ is inert; if nearly
        # everything overflows it sits below and every call sinks to the bottom
        # queue. Either way the fix is a ladder scaled to this distribution.
        self._service_ladder = tuple(self.queue_quanta)
        self.completed_service_hist = [0] * (len(self._service_ladder) + 1)

        # Auto-calibration. Opt-in, so runs that do not ask for it keep the
        # ladder they were configured with and stay comparable to earlier
        # results. Calibrating once and then freezing -- rather than tracking
        # continuously -- keeps the measurement phase deterministic and avoids
        # the feedback loop a continuous controller would have, since quanta
        # affect scheduling, which affects service times, which would feed back
        # into the quanta.
        self._auto_quanta = os.getenv(_AUTO_QUANTA_ENV, "").strip() not in ("", "0")
        self._auto_samples: list[float] = []
        self._auto_calibrated = False
        # Set on the first completion, so the window is measured from when load
        # actually starts rather than from engine construction, which precedes
        # model loading and graph capture by minutes.
        self._auto_first_completion_ts: float | None = None

        # Instrumentation. Under async scheduling the policy's Python work
        # overlaps GPU execution, so it costs nothing until it approaches the
        # step time; ``_policy_timer`` against ``_interval_timer`` reports that
        # headroom as a duty cycle (see PhaseTimer.duty_cycle). Emitted
        # together so both windows cover the same steps.
        self._policy_timer = PhaseTimer("autellix_policy", emit_every=_TIMER_WINDOW)
        self._interval_timer = PhaseTimer("schedule_interval", emit_every=_TIMER_WINDOW)
        self._last_schedule_ts: float | None = None

        # D1: one scheduler-side clock. ``schedule()`` is called once per engine
        # step, so the interval between calls IS the batch step wall time when
        # the engine is busy. ``_clock`` is injectable so tests can feed a
        # deterministic step sequence; production uses ``time.monotonic``.
        self._clock: Callable[[], float] = time.monotonic
        self._last_step_ts: float | None = None
        self._step_dt: float = 0.0
        # Clamp guards idle gaps / pauses: an engine idle for seconds must not
        # charge its first busy step a huge dt. Busy steps are tens of ms, well
        # under the cap, so it only ever bounds the post-idle step.
        self._max_step_dt: float = 1.0

    def _register_call_state(self, request: Request, queue_index: int) -> None:
        """Enter a new call at ``queue_index`` with that queue's full quantum."""
        request.priority = queue_index
        self._call_state[request.request_id] = CallQueueState(
            queue_index=queue_index,
            quantum_remaining=self.queue_quanta[queue_index],
            wait_window=0.0,
            service_window=0.0,
        )

    def _program_windows(self, request: Request) -> tuple[float, float]:
        """Return the program-level ``(W_p, T_p)`` for the starvation ratio.

        The default is call-level-only behaviour (the MLFQ / FastServe
        baseline); program-aware schedulers override this with process-table
        windows.
        """
        return (0.0, 0.0)

    def _on_service_step(self, req_id: str, amount: float) -> None:
        """Hook called once per charged step with its duration (default: no-op).

        Args:
            req_id: The scheduled request charged this step.
            amount: Seconds of service charged (the measured step wall time).
        """

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Run the policy bookkeeping, then delegate to the base loop.

        The bookkeeping pass (anti-starvation promotion + proactive preemption)
        mutates ``request.priority`` and the waiting heap so that the base
        loop, which is unchanged, admits and evicts calls in policy order.

        Proactive preemptions happen before the base loop, so the base only
        reports its own KV-pressure preemptions in ``preempted_req_ids``; the
        proactive victims are merged in afterwards. The v2 model runner frees a
        request's persistent-batch slot only for ids in ``finished_req_ids`` or
        ``preempted_req_ids``, and a proactive preemption always happens at a
        full batch whose vacated slot is immediately re-admitted -- so an
        unreported victim overflows the worker's ``max_num_reqs`` slots
        ("No free indices"). The worker frees slots before adding new/resumed
        requests, so reporting a victim resumed in this same step is safe.
        """
        now = self._clock()
        entered = time.perf_counter()
        if self._last_schedule_ts is not None:
            self._interval_timer.record(entered - self._last_schedule_ts)
        self._last_schedule_ts = entered

        # Wall-clock attained service: the interval between schedule() calls is
        # the batch step duration when the engine is busy.
        if self._last_step_ts is None:
            self._step_dt = 0.0
        else:
            self._step_dt = min(max(now - self._last_step_ts, 0.0), self._max_step_dt)
        self._last_step_ts = now
        self._accrue_wait_and_promote()
        preempted_req_ids = self._proactively_preempt(now)
        self._policy_timer.record(time.perf_counter() - entered)
        self._emit_timing_if_due()

        scheduler_output = super().schedule(throttle_prefills)  # type: ignore[misc]
        if preempted_req_ids:
            if scheduler_output.preempted_req_ids is None:
                scheduler_output.preempted_req_ids = preempted_req_ids
            else:
                scheduler_output.preempted_req_ids |= preempted_req_ids
        return scheduler_output

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Count the preemption, then delegate to the base scheduler.

        Both preemption paths funnel through here -- this mixin's proactive
        policy preemption (:meth:`_proactively_preempt`) and vLLM's own
        memory-pressure preemption in the base scheduling loop -- so a single
        counter covers both, and it works identically under
        ``preemption_mode="recompute"``, where no swap-side instrumentation
        exists to observe them.
        """
        self.preemption_count += 1
        super()._preempt_request(request, timestamp)  # type: ignore[misc]

    def _emit_timing_if_due(self) -> None:
        """Log the policy's share of the step budget once per window.

        ``duty`` near zero means async scheduling hides the policy entirely and
        optimising it buys nothing; approaching one means it has become the
        step's critical path. Both timers are reset together so the ratio always
        compares the same window.

        The preemption counters are cumulative rather than per-window: they are
        the number the swap-vs-recompute comparison turns on, and reporting a
        running total means the final line carries the run total regardless of
        where the window boundaries fell.
        """
        if not self._policy_timer.should_emit():
            return
        # Snapshot of where live calls currently sit, alongside the cumulative
        # movement counters. A histogram concentrated in one level means the
        # MLFQ has collapsed to FCFS; a spread means the quanta are doing work.
        occupancy = [0] * self.num_queues
        for state in self._call_state.values():
            if 0 <= state.queue_index < self.num_queues:
                occupancy[state.queue_index] += 1
        logger.info(
            "autellix_timing policy=%s interval=%s duty=%s preempt_total=%d "
            "preempt_proactive=%d demotions=%d promotions=%d queue_occupancy=%s "
            "call_service_hist=%s ladder=%s",
            self._policy_timer.summary(),
            self._interval_timer.summary(),
            duty_cycle(self._policy_timer, self._interval_timer),
            self.preemption_count,
            self.proactive_preemption_count,
            self.demotion_count,
            self.promotion_count,
            occupancy,
            self.completed_service_hist,
            list(self._service_ladder),
        )
        self._policy_timer.reset()
        self._interval_timer.reset()

    def _accrue_wait_and_promote(self) -> None:
        """Accrue wait for waiting calls and promote the starving ones to Q1.

        Every call currently waiting (in either the main or the skipped queue)
        accrues this step's wall time as wait ``W_c`` (seconds, D3). A call
        outside Q1 whose ``(W_p + W_c) / max(1, T_p + T_c)`` has reached ``beta``
        (program windows from :meth:`_program_windows`) is promoted to Q1 with
        its call-level windows reset, and the queue is re-heapified so the base
        loop sees the new ordering.
        """
        for queue in (self.waiting, self.skipped_waiting):
            promoted: list[Request] = []
            for request in list(queue):
                # Every queued call was registered by add_request, so its state
                # is present; a missing entry is a real invariant violation.
                state = self._call_state[request.request_id]
                state.wait_window += self._step_dt
                if state.queue_index == 0:
                    continue
                program_wait, program_service = self._program_windows(request)
                if self.binner.anti_starvation(
                    program_wait + state.wait_window,
                    program_service + state.service_window,
                    self.beta,
                ):
                    self._promote_to_top(request, state)
                    promoted.append(request)
            if promoted:
                # Priorities changed in place; re-insert to restore heap order.
                queue.remove_requests(promoted)
                for request in promoted:
                    queue.add_request(request)

    def _promote_to_top(self, request: Request, state: CallQueueState) -> None:
        """Promote a starving call to Q1, resetting only its call-level windows.

        The program-level windows (``W_p``, ``T_p``) are deliberately left
        untouched (paper §4.2.2): resetting them would immediately re-starve
        the program's other calls.
        """
        self.promotion_count += 1
        state.queue_index = 0
        state.quantum_remaining = self.queue_quanta[0]
        state.wait_window = 0.0
        state.service_window = 0.0
        request.priority = 0

    def _proactively_preempt(self, now: float) -> set[str]:
        """Preempt the worst running call(s) so better waiting calls can run.

        Only fires when the batch is full. Each iteration preempts the current
        worst running call while the best waiting call *strictly* outranks it,
        up to ``max_proactive_preemptions_per_step`` times. Freed calls return
        to the waiting queue (recompute-based, KV blocks freed) so the base
        loop can admit the better waiting calls into the vacated slots. The
        strict-outrank guard makes preemption stop once the queues converge,
        and the per-step cap bounds recompute so it cannot thrash.

        Returns:
            The preempted request ids, to be merged into the step's
            ``SchedulerOutput.preempted_req_ids`` (see :meth:`schedule`).
        """
        preempted_req_ids: set[str] = set()
        if len(self.running) < self.max_num_running_reqs:
            return preempted_req_ids
        while (
            len(preempted_req_ids) < self.max_proactive_preemptions_per_step
            and self.waiting
            and self.running
        ):
            best_waiting = self.waiting.peek_request()
            worst_running = max(
                self.running, key=lambda r: (r.priority, r.arrival_time)
            )
            if not self.binner.outranks(best_waiting.priority, worst_running.priority):
                break
            self.running.remove(worst_running)
            self._preempt_request(worst_running, now)
            self.proactive_preemption_count += 1
            preempted_req_ids.add(worst_running.request_id)
        return preempted_req_ids

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Charge this step's wall time to every scheduled call, demote on empty.

        ``super()`` (``AsyncScheduler`` -> ``Scheduler``) runs first to reserve
        output placeholders. Then every request co-scheduled this step -- prefill
        chunks **included** (D2), unlike the step-count version which skipped
        them -- is charged the measured step duration ``self._step_dt`` (set in
        :meth:`schedule`) against both its quantum and its service window. This
        is execution-time accounting (the paper's ``e_i``): each co-scheduled
        request pays the full step wall time, not a GPU share. A call whose
        quantum is exhausted is demoted one level with its priority updated in
        place. Each charge is forwarded to :meth:`_on_service_step` so
        program-aware hosts accrue attained service in the same pass, keeping the
        quantum and the program fold in lockstep. The first step after start /
        idle has ``_step_dt == 0`` and charges nothing.
        """
        super()._update_after_schedule(scheduler_output)  # type: ignore[misc]
        step_dt = self._step_dt
        if step_dt <= 0.0:
            return
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is None:
                continue
            # A scheduled call is always registered (see add_request).
            state = self._call_state[req_id]
            state.quantum_remaining -= step_dt
            state.service_window += step_dt
            state.total_service += step_dt
            self._on_service_step(req_id, step_dt)
            if state.quantum_remaining <= 0.0:
                self.demotion_count += 1
                state.queue_index = self.binner.demote(state.queue_index)
                request.priority = state.queue_index
                state.quantum_remaining = self.queue_quanta[state.queue_index]

    def record_completed_call(self, state: CallQueueState | None) -> None:
        """Bucket a finished call's lifetime service against the quantum ladder.

        Called from every completion path: :meth:`_release_call_state` for the
        MLFQ baseline, and the program-aware schedulers' ``_fold_completed_calls``,
        which pop ``_call_state`` themselves.
        """
        if state is None:
            return
        self._maybe_calibrate_quanta(state.total_service)
        for i, bound in enumerate(self._service_ladder):
            if state.total_service < bound:
                self.completed_service_hist[i] += 1
                return
        self.completed_service_hist[-1] += 1

    def _maybe_calibrate_quanta(self, total_service: float) -> None:
        """Collect one sample and, once there are enough, fix the ladder.

        Runs only under ``AUTELLIX_AUTO_QUANTA``. The first
        ``_AUTO_QUANTA_SKIP_S`` seconds of completions are discarded as startup
        transient, the next ``_AUTO_QUANTA_WINDOW_S`` seconds are sampled, and Q1
        is then set to the ``_AUTO_QUANTA_QUANTILE`` quantile of that window with
        the rest of the ladder doubling from there; calibration then stops for
        the life of the engine. If the window yields fewer than
        ``_AUTO_QUANTA_MIN_SAMPLES`` completions the configured ladder is kept
        rather than fitted to noise.

        Calls already in flight keep the quantum they were issued, so the switch
        cannot retroactively demote anything; only calls registered afterwards
        see the new ladder.
        """
        if not self._auto_quanta or self._auto_calibrated:
            return
        now = self._clock()
        if self._auto_first_completion_ts is None:
            self._auto_first_completion_ts = now
            return
        elapsed = now - self._auto_first_completion_ts
        if elapsed < _AUTO_QUANTA_SKIP_S:
            return
        self._auto_samples.append(total_service)
        if elapsed < _AUTO_QUANTA_SKIP_S + _AUTO_QUANTA_WINDOW_S:
            return
        if len(self._auto_samples) < _AUTO_QUANTA_MIN_SAMPLES:
            # Too few completions to trust a quantile; keep the configured
            # ladder rather than calibrating onto noise.
            self._auto_calibrated = True
            logger.info(
                "autellix_auto_quanta skipped: only %d samples in %.0fs window",
                len(self._auto_samples),
                _AUTO_QUANTA_WINDOW_S,
            )
            self._auto_samples.clear()
            return

        ordered = sorted(self._auto_samples)
        idx = min(int(_AUTO_QUANTA_QUANTILE * len(ordered)), len(ordered) - 1)
        base = max(ordered[idx], _MIN_QUANTUM_S)
        self.queue_quanta = tuple(base * (2**i) for i in range(self.num_queues))
        self._service_ladder = tuple(self.queue_quanta)
        self.completed_service_hist = [0] * (len(self._service_ladder) + 1)
        self._auto_calibrated = True
        self._auto_samples.clear()
        logger.info(
            "autellix_auto_quanta calibrated: p%d=%.4fs over %d calls in a "
            "%.0fs window (after %.0fs skip) -> ladder=%s",
            int(_AUTO_QUANTA_QUANTILE * 100),
            base,
            len(ordered),
            _AUTO_QUANTA_WINDOW_S,
            _AUTO_QUANTA_SKIP_S,
            [round(q, 4) for q in self.queue_quanta],
        )

    def _release_call_state(self, req_ids: Iterable[str]) -> None:
        """Drop each finished call's per-call state.

        Idempotent per call (``pop`` with a default), so the two completion
        paths -- ``update_from_output`` for normal stops and
        ``finish_requests`` for aborts -- never error on an already-released
        id.
        """
        for req_id in req_ids:
            self.record_completed_call(self._call_state.pop(req_id, None))
