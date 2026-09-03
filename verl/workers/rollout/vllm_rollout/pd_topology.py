# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Physical placement helpers for vLLM PD replicas."""

from __future__ import annotations

import socket
from collections.abc import Sequence
from dataclasses import dataclass

from verl.utils.net_utils import is_valid_ipv6_address

__all__ = ["PDEnginePlacement", "PDNodePlacement", "PDPortReservation", "plan_pd_engine_placements"]


@dataclass(frozen=True)
class PDNodePlacement:
    """The local ranks of one logical P/D engine placed on one node."""

    node_rank: int
    worker_start: int
    worker_end: int
    node_id: str
    host_ip: str
    accelerator_ids: tuple[str, ...]

    @property
    def local_world_size(self) -> int:
        return self.worker_end - self.worker_start


@dataclass(frozen=True)
class PDEnginePlacement:
    """One logical P/D engine, potentially distributed across nodes."""

    role: str
    index: int
    tp_size: int
    dp_size: int
    nodes: tuple[PDNodePlacement, ...]

    @property
    def world_size(self) -> int:
        return self.tp_size * self.dp_size

    @property
    def worker_start(self) -> int:
        return self.nodes[0].worker_start

    @property
    def worker_end(self) -> int:
        return self.nodes[-1].worker_end

    @property
    def head(self) -> PDNodePlacement:
        return self.nodes[0]


def plan_pd_engine_placements(
    worker_infos: Sequence[tuple[str, str, str]],
    *,
    prefill_replicas: int,
    prefill_tp: int,
    prefill_dp: int,
    decode_replicas: int,
    decode_tp: int,
    decode_dp: int,
    gpus_per_node: int,
) -> tuple[list[PDEnginePlacement], list[PDEnginePlacement]]:
    """Split ordered workers into P/D engines and their node-rank groups.

    An engine may span nodes either through model parallelism (TP larger than
    one node, with DP=1) or through vLLM's external-DP path (node-local TP and
    DP>1). The current vLLM launcher cannot combine both forms in one engine.
    """

    expected_workers = prefill_replicas * prefill_tp * prefill_dp
    expected_workers += decode_replicas * decode_tp * decode_dp
    if len(worker_infos) != expected_workers:
        raise ValueError(f"PD topology expects {expected_workers} workers, got {len(worker_infos)}")

    placements: list[PDEnginePlacement] = []
    cursor = 0
    for role, replicas, tp_size, dp_size in (
        ("prefill", prefill_replicas, prefill_tp, prefill_dp),
        ("decode", decode_replicas, decode_tp, decode_dp),
    ):
        if tp_size > gpus_per_node and dp_size > 1:
            raise ValueError(
                f"{role} cannot combine cross-node TP (tp_size={tp_size}, "
                f"gpus_per_node={gpus_per_node}) with dp_size={dp_size} in one engine"
            )
        engine_world_size = tp_size * dp_size
        local_world_size = min(gpus_per_node, engine_world_size)
        model_parallel_crosses_nodes = tp_size > gpus_per_node
        invalid_model_parallel_layout = model_parallel_crosses_nodes and tp_size % local_world_size != 0
        invalid_external_dp_layout = not model_parallel_crosses_nodes and local_world_size % tp_size != 0
        if (
            engine_world_size % local_world_size != 0
            or invalid_model_parallel_layout
            or invalid_external_dp_layout
        ):
            raise ValueError(
                f"{role} engine world_size={engine_world_size} cannot be evenly placed on "
                f"{gpus_per_node}-device nodes with tp_size={tp_size}"
            )
        engine_nnodes = engine_world_size // local_world_size

        for index in range(replicas):
            node_placements: list[PDNodePlacement] = []
            for node_rank in range(engine_nnodes):
                start, end = cursor, cursor + local_world_size
                group = worker_infos[start:end]
                node_ids = {info[0] for info in group}
                host_ips = {info[2] for info in group}
                if len(node_ids) != 1 or len(host_ips) != 1:
                    raise ValueError(
                        f"{role}[{index}] node_rank={node_rank} workers[{start}:{end}] "
                        "span multiple physical nodes"
                    )
                accelerator_ids = tuple(info[1] for info in group)
                if len(set(accelerator_ids)) != len(accelerator_ids):
                    raise ValueError(
                        f"{role}[{index}] node_rank={node_rank} has duplicate accelerator ids: "
                        f"{accelerator_ids}"
                    )
                node_placements.append(
                    PDNodePlacement(
                        node_rank=node_rank,
                        worker_start=start,
                        worker_end=end,
                        node_id=next(iter(node_ids)),
                        host_ip=next(iter(host_ips)),
                        accelerator_ids=accelerator_ids,
                    )
                )
                cursor = end
            physical_node_ids = {node.node_id for node in node_placements}
            if len(physical_node_ids) != engine_nnodes:
                raise ValueError(
                    f"{role}[{index}] needs {engine_nnodes} distinct nodes, got "
                    f"{len(physical_node_ids)}"
                )
            placements.append(
                PDEnginePlacement(
                    role=role,
                    index=index,
                    tp_size=tp_size,
                    dp_size=dp_size,
                    nodes=tuple(node_placements),
                )
            )

    prefills = [placement for placement in placements if placement.role == "prefill"]
    decodes = [placement for placement in placements if placement.role == "decode"]
    return prefills, decodes


class PDPortReservation:
    """Reserve a TCP port on the physical node that will host a P/D engine."""

    def __init__(self, host_ip: str, requested_port: int | None = None, port_count: int = 1) -> None:
        if port_count < 1:
            raise ValueError(f"port_count must be positive, got {port_count}")
        self._family = socket.AF_INET6 if is_valid_ipv6_address(host_ip) else socket.AF_INET
        self._host_ip = host_ip
        self._sockets: list[socket.socket] = []
        if requested_port is not None:
            if requested_port < 1 or requested_port + port_count - 1 >= 65536:
                raise ValueError(
                    f"invalid port range [{requested_port}, {requested_port + port_count - 1}]"
                )
            self._port = requested_port
            self._bind_range(requested_port, port_count)
            return

        # The connector derives per-rank ports from one base. Ask the kernel
        # for a candidate base, then reserve the complete consecutive range.
        # Retry because an ephemeral candidate may sit near 65535 or overlap
        # another service on a following port.
        for _ in range(64):
            candidate_socket = self._new_socket()
            candidate_socket.bind((host_ip, 0))
            candidate_port = candidate_socket.getsockname()[1]
            candidate_socket.close()
            if candidate_port + port_count - 1 >= 65536:
                continue
            try:
                self._port = candidate_port
                self._bind_range(candidate_port, port_count)
                return
            except OSError:
                self.release()
        raise RuntimeError(f"unable to reserve {port_count} consecutive ports on {host_ip}")

    def _new_socket(self) -> socket.socket:
        reserved_socket = socket.socket(family=self._family, type=socket.SOCK_STREAM)
        reserved_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return reserved_socket

    def _bind_range(self, base_port: int, port_count: int) -> None:
        try:
            for offset in range(port_count):
                reserved_socket = self._new_socket()
                reserved_socket.bind((self._host_ip, base_port + offset))
                self._sockets.append(reserved_socket)
        except BaseException:
            self.release()
            raise

    def get_port(self) -> int:
        return self._port

    def release(self) -> None:
        for reserved_socket in self._sockets:
            reserved_socket.close()
        self._sockets = []
