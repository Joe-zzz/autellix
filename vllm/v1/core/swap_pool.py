# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host KV swap pool for swap-based preemption (``preemption_mode="swap"``).

When the scheduler preempts a request under swap mode, it parks the request's KV
cache in a pinned host buffer instead of dropping it (recompute). This module is
the **scheduler-side** allocator: it hands out and reclaims host KV *block slots*
(integer ids) and remembers which slots hold each parked request's blocks. It is
pure Python and holds no tensors — the pinned host KV mirror itself lives in the
worker/model-runner, which copies blocks GPU<->host by slot id via
``copy_kv_blocks`` (``vllm/distributed/kv_transfer/kv_connector/utils.py``).

Slots are **non-evictable** while a request is parked (unlike the hash-keyed
prefix-cache offload managers): a parked request keeps its host slots until it
resumes (swap-in) or is aborted. When the pool is exhausted, the caller falls
back to recompute for that request, so undersizing degrades gracefully rather
than blocking. One host block corresponds to one GPU block (same block size).
"""


def num_host_blocks_for(swap_space_gb: float, kv_bytes_per_block: int) -> int:
    """Number of host KV block slots that fit in the configured swap space.

    Args:
        swap_space_gb: Host memory reserved for the swap pool, in GiB.
        kv_bytes_per_block: Bytes of KV cache per block (all layers/groups),
            matching the GPU block layout.

    Returns:
        The block count (0 when swap space or block size is non-positive, which
        effectively disables swap — every preemption then falls back to
        recompute).
    """
    if swap_space_gb <= 0.0 or kv_bytes_per_block <= 0:
        return 0
    return int(swap_space_gb * (1024**3)) // kv_bytes_per_block


class HostSwapPool:
    """Fixed pool of host KV block slots + a per-request parked-blocks registry.

    Manages ``num_host_blocks`` slots as a free-list and maps each parked
    ``request_id`` to the ordered list of host slot ids holding its KV blocks
    (in logical block order, so slot ``i`` mirrors the request's ``i``-th GPU
    block). All bookkeeping is by integer id; the worker owns the tensors.
    """

    def __init__(self, num_host_blocks: int) -> None:
        """Build the pool.

        Args:
            num_host_blocks: Total host block slots (0 disables swap).
        """
        self.num_host_blocks = num_host_blocks
        self._free: list[int] = list(range(num_host_blocks))
        self._req_to_host: dict[str, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        """Host slots currently available."""
        return len(self._free)

    def can_fit(self, num_blocks: int) -> bool:
        """Whether ``num_blocks`` slots can be reserved right now."""
        return 0 <= num_blocks <= len(self._free)

    def is_parked(self, request_id: str) -> bool:
        """Whether ``request_id`` currently holds parked KV in the pool."""
        return request_id in self._req_to_host

    def alloc(self, request_id: str, num_blocks: int) -> list[int] | None:
        """Reserve ``num_blocks`` host slots for a request being parked.

        Args:
            request_id: The request being swapped out.
            num_blocks: Host slots needed (= the request's GPU block count).

        Returns:
            The reserved slot ids (logical block order), or ``None`` if the pool
            cannot fit them (caller falls back to recompute).

        Raises:
            ValueError: If ``request_id`` is already parked (double swap-out).
        """
        if request_id in self._req_to_host:
            raise ValueError(f"request {request_id} is already parked in the swap pool")
        if num_blocks > len(self._free):
            return None
        host_ids = [self._free.pop() for _ in range(num_blocks)]
        self._req_to_host[request_id] = host_ids
        return host_ids

    def get(self, request_id: str) -> list[int] | None:
        """Return a parked request's host slot ids, or ``None`` if not parked."""
        return self._req_to_host.get(request_id)

    def free(self, request_id: str) -> list[int]:
        """Release a request's host slots back to the pool.

        Called after swap-in retires (KV copied back to GPU) or when a parked
        request is aborted. Idempotent: releasing a request that holds no slots
        returns an empty list.

        Args:
            request_id: The request whose slots to release.

        Returns:
            The released slot ids (empty if the request was not parked).
        """
        host_ids = self._req_to_host.pop(request_id, [])
        self._free.extend(host_ids)
        return host_ids
