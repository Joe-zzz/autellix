# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request attained-service accrual for the Autellix policies.

Attained service is the Least-Attained-Service signal used by the schedulers:
the **seconds of engine wall time** a call (request) has been scheduled for
(D2/D3) -- each continuous-batching step charges its measured duration, so
long prefill steps and big batches are priced at their true cost rather than a
constant per-step unit. This tracker holds the live per-request accrual; on
completion the call's total is folded into its program via the process table.
The module is pure Python with no vLLM/torch dependency.
"""


class AttainedServiceTracker:
    """Live attained service per in-flight request, keyed by ``request_id``.

    Accrual only ever grows: it is deliberately *not* reset on preemption,
    because service recomputed after a recompute-based preemption was still
    served GPU time and should count toward the request's attained service.
    """

    def __init__(self) -> None:
        self._service: dict[str, float] = {}

    def record_step(self, request_id: str, amount: float = 1.0) -> None:
        """Accrue a scheduled step's service (seconds of engine wall time).

        Args:
            request_id: The scheduled request.
            amount: Seconds of service to add for this step.
        """
        self._service[request_id] = self._service.get(request_id, 0.0) + amount

    def get(self, request_id: str) -> float:
        """Return the request's accrued service (``0.0`` if unknown)."""
        return self._service.get(request_id, 0.0)

    def pop(self, request_id: str) -> float:
        """Remove and return the request's accrued service.

        Called when a call completes so its total can be folded into the program.

        Args:
            request_id: The completing request.

        Returns:
            The accrued service, or ``0.0`` if the request is unknown.
        """
        return self._service.pop(request_id, 0.0)
