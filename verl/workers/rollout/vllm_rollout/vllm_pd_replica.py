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
"""Multi-prefill/multi-decode vLLM PD replica with cross-node KV transfer.

Asymmetric TP/DP is supported. One logical P or D engine may span nodes either
through model parallelism or role-specific external DP. GPU keeps vLLM's native
NIXL/Mooncake connectors; Ascend uses vLLM-Ascend's MooncakeConnectorV1.
"""

import asyncio
import copy
import logging
import os
import uuid
from collections.abc import Mapping
from dataclasses import replace as _dc_replace
from typing import Any, Optional

import ray
from ray.actor import ActorHandle

from verl.utils.device import get_device_name, get_resource_name, is_torch_npu_available
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
from verl.workers.rollout.vllm_rollout.pd_routing import DecodeRoutingController
from verl.workers.rollout.vllm_rollout.pd_topology import (
    PDEnginePlacement,
    PDPortReservation,
    plan_pd_engine_placements,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _deep_merge_dict(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""
    merged = {key: copy.deepcopy(value) for key, value in base.items()}
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _drop_none_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Remove unset Hydra schema values before applying role overrides."""
    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            nested = _drop_none_values(value)
            if nested:
                result[key] = nested
        elif value is not None:
            result[key] = value
    return result


class vLLMPDReplica(vLLMReplica):
    """Replica that runs vLLM in prefill-decode disaggregated mode."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank,
            config,
            model_config,
            gpus_per_node,
            is_reward_model,
            is_teacher_model,
            name_suffix,
        )

        disagg = self.config.disaggregation
        assert disagg.enabled, "vLLMPDReplica requires rollout.disaggregation.enabled=True"

        if disagg.transfer_backend not in ("nixl", "mooncake"):
            raise NotImplementedError(
                f"vLLMPDReplica supports transfer_backend in ('nixl', 'mooncake') in this "
                f"revision; got {disagg.transfer_backend!r}. mori/ascend/fake are reserved "
                f"in DisaggregationConfig and will land in follow-ups."
            )
        self._n_prefill = disagg.prefill_replicas
        self._n_decode = disagg.decode_replicas
        self._prefill_dp = disagg.prefill_data_parallel_size
        self._decode_dp = disagg.decode_data_parallel_size

        self._prefill_tp = self.config.tensor_model_parallel_size
        # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
        self._decode_tp = (
            disagg.decode_tensor_model_parallel_size
            if disagg.decode_tensor_model_parallel_size is not None
            else self._prefill_tp
        )

        prefill_engine_world_size = self._prefill_tp * self._prefill_dp
        decode_engine_world_size = self._decode_tp * self._decode_dp
        pd_world_size = self._n_prefill * prefill_engine_world_size
        pd_world_size += self._n_decode * decode_engine_world_size
        invalid_cross_node_tp = (
            self._prefill_tp > gpus_per_node and self._prefill_dp > 1
        ) or (self._decode_tp > gpus_per_node and self._decode_dp > 1)
        if invalid_cross_node_tp:
            raise NotImplementedError(
                "one P/D engine cannot combine cross-node TP with external DP; "
                f"prefill=(tp={self._prefill_tp}, dp={self._prefill_dp}), "
                f"decode=(tp={self._decode_tp}, dp={self._decode_dp}), "
                f"gpus_per_node={gpus_per_node}"
            )
        if pd_world_size > gpus_per_node and pd_world_size % gpus_per_node != 0:
            raise ValueError(
                f"multi-node PD footprint ({pd_world_size} devices) must fill whole "
                f"{gpus_per_node}-device nodes in the current Ray resource-pool layout"
            )
        if self.config.data_parallel_size != 1:
            raise NotImplementedError(f"data_parallel_size=1 only (got {self.config.data_parallel_size})")
        if self.config.pipeline_model_parallel_size != 1:
            raise NotImplementedError(
                f"pipeline_model_parallel_size=1 only "
                f"(got {self.config.pipeline_model_parallel_size}); PD path does not model PP yet"
            )

        self.world_size = pd_world_size
        self.gpus_per_replica_node = min(self.gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0
        self.nnodes = self.world_size // self.gpus_per_replica_node

        if disagg.bootstrap_port is not None:
            last_port = disagg.bootstrap_port + (self.replica_rank + 1) * pd_world_size - 1
            if last_port >= 65536:
                raise ValueError(f"PD bootstrap port range exceeds 65535 (last port={last_port})")

        self._prefill_servers: list[ActorHandle] = []
        self._decode_servers: list[ActorHandle] = []
        self._prefill_engine_servers: list[list[ActorHandle]] = []
        self._decode_engine_servers: list[list[ActorHandle]] = []
        self._prefill_server_addresses: list[str] = []
        self._decode_server_addresses: list[str] = []
        self._decode_router: ActorHandle | None = None

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker count {len(self.workers)} != PD world size {self.world_size}"
        )
        use_ascend_mooncake_v1 = self._is_ascend_platform()
        transfer_backend = self.config.disaggregation.transfer_backend
        if use_ascend_mooncake_v1 and transfer_backend == "nixl":
            logger.warning(
                "NixlConnector is not supported for Ascend PD; falling back to "
                "MooncakeConnectorV1"
            )
            transfer_backend = "mooncake"
        if (
            self.nnodes > 1
            and not use_ascend_mooncake_v1
            and transfer_backend == "mooncake"
            and self.config.disaggregation.mooncake_protocol in ("local", "nvlink")
        ):
            raise ValueError(
                "cross-node Mooncake PD requires mooncake_protocol='rdma' or 'tcp'; "
                f"got {self.config.disaggregation.mooncake_protocol!r}"
            )

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                        ray.util.get_node_ip_address().strip("[]"),
                    )
                )
                for worker in self.workers
            ]
        )

        prefill_placements, decode_placements = plan_pd_engine_placements(
            worker_infos,
            prefill_replicas=self._n_prefill,
            prefill_tp=self._prefill_tp,
            prefill_dp=self._prefill_dp,
            decode_replicas=self._n_decode,
            decode_tp=self._decode_tp,
            decode_dp=self._decode_dp,
            gpus_per_node=self.gpus_per_node,
        )
        placements = [*prefill_placements, *decode_placements]
        side_channel_ports, port_reservations = await self._reserve_side_channel_ports(placements)
        engine_ids = [uuid.uuid4().hex for _ in placements]

        try:
            for ordinal, placement in enumerate(placements):
                kv_cfg = self._build_kv_transfer_config(
                    role=placement.role,
                    engine_id=engine_ids[ordinal],
                    transfer_backend=transfer_backend,
                    mooncake_protocol=self.config.disaggregation.mooncake_protocol,
                    use_ascend_mooncake_v1=use_ascend_mooncake_v1,
                    kv_port=side_channel_ports[ordinal],
                    prefill_tp=self._prefill_tp,
                    prefill_dp=self._prefill_dp,
                    decode_tp=self._decode_tp,
                    decode_dp=self._decode_dp,
                )
                engine_servers = [
                    self._spawn_pd_server(
                        role=placement.role,
                        pd_index=placement.index,
                        workers=self.workers[node.worker_start : node.worker_end],
                        node_id=node.node_id,
                        node_rank=node.node_rank,
                        cuda_visible_devices=",".join(node.accelerator_ids),
                        tp=placement.tp_size,
                        dp=placement.dp_size,
                        nnodes=len(placement.nodes),
                        kv_transfer_config=kv_cfg,
                        side_channel_host=node.host_ip,
                        side_channel_port=side_channel_ports[ordinal],
                        mooncake_bootstrap_port=side_channel_ports[ordinal],
                        actor_name=self._pd_actor_name(placement, node.node_rank),
                        zmq_base_trainer_rank=node.worker_start,
                    )
                    for node in placement.nodes
                ]
                if placement.role == "prefill":
                    self._prefill_engine_servers.append(engine_servers)
                else:
                    self._decode_engine_servers.append(engine_servers)

            # Force server actor construction while ports are still reserved,
            # then release immediately before vLLM binds its side channels.
            engine_masters = await asyncio.gather(
                *[engine_servers[0].get_master_address.remote() for engine_servers in self._all_pd_engine_servers]
            )
            await asyncio.gather(
                *[server.get_master_address.remote() for server in self._all_pd_node_servers]
            )
            await asyncio.gather(*[reservation.release.remote() for reservation in port_reservations])

            await asyncio.gather(
                *[
                    self._launch_pd_engine(engine_servers, master)
                    for engine_servers, master in zip(
                        self._all_pd_engine_servers,
                        engine_masters,
                        strict=True,
                    )
                ]
            )
        finally:
            for reservation in port_reservations:
                ray.kill(reservation, no_restart=True)

        self._prefill_servers = [servers[0] for servers in self._prefill_engine_servers]
        self._decode_servers = [servers[0] for servers in self._decode_engine_servers]
        decode_addresses = await asyncio.gather(
            *[server.get_server_address.remote() for server in self._decode_servers]
        )
        decode_peer_ids = [self._format_address(host, port, scheme=True) for host, port in decode_addresses]
        decode_router_cls = ray.remote(DecodeRoutingController)
        self._decode_router = decode_router_cls.options(num_cpus=0).remote(
            self.config.disaggregation.decode_policy,
            decode_peer_ids,
        )
        await asyncio.gather(
            *[
                server.set_pd_peer.remote(
                    decode_peers=self._decode_servers,
                    prefill_side_channel_host=placement.head.host_ip,
                    prefill_side_channel_port=side_channel_ports[index],
                    prefill_engine_id=engine_ids[index],
                    decode_router=self._decode_router,
                )
                for index, (server, placement) in enumerate(zip(self._prefill_servers, prefill_placements, strict=True))
            ]
        )

        self.servers = self._all_pd_node_servers
        prefill_addresses = await asyncio.gather(
            *[server.get_server_address.remote() for server in self._prefill_servers]
        )
        self._prefill_server_addresses = [self._format_address(host, port) for host, port in prefill_addresses]
        self._decode_server_addresses = [self._format_address(host, port) for host, port in decode_addresses]
        self._server_handle = self._prefill_servers[0]
        self._server_address = self._prefill_server_addresses[0]

        logger.info(
            "vLLMPDReplica rank=%s launched: prefills=%s, decodes=%s",
            self.replica_rank,
            self._prefill_server_addresses,
            self._decode_server_addresses,
        )

    @property
    def _all_pd_engine_servers(self) -> list[list[ActorHandle]]:
        return [*self._prefill_engine_servers, *self._decode_engine_servers]

    @property
    def _all_pd_node_servers(self) -> list[ActorHandle]:
        return [server for engine_servers in self._all_pd_engine_servers for server in engine_servers]

    @property
    def _all_pd_servers(self) -> list[ActorHandle]:
        """Backward-compatible alias for every node-rank server actor."""
        return self._all_pd_node_servers

    def _requested_port(self, ordinal: int) -> int | None:
        base_port = self.config.disaggregation.bootstrap_port
        if base_port is None:
            return None
        prefill_engine_world_size = self._prefill_tp * self._prefill_dp
        decode_engine_world_size = self._decode_tp * self._decode_dp
        if ordinal < self._n_prefill:
            engine_offset = ordinal * prefill_engine_world_size
        else:
            engine_offset = self._n_prefill * prefill_engine_world_size
            engine_offset += (ordinal - self._n_prefill) * decode_engine_world_size
        return base_port + self.replica_rank * self.world_size + engine_offset

    async def _reserve_side_channel_ports(
        self,
        placements: list[PDEnginePlacement],
    ) -> tuple[list[int], list[ActorHandle]]:
        """Reserve every connector port range on the node that will bind it."""
        reservation_cls = ray.remote(PDPortReservation)
        reservations: list[ActorHandle] = []
        try:
            head_reservations: list[ActorHandle] = []
            for ordinal, placement in enumerate(placements):
                reservation = reservation_cls.options(
                    num_cpus=0,
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=placement.head.node_id,
                        soft=False,
                    ),
                ).remote(
                    placement.head.host_ip,
                    self._requested_port(ordinal),
                    placement.world_size,
                )
                head_reservations.append(reservation)
                reservations.append(reservation)
            base_ports = await asyncio.gather(
                *[reservation.get_port.remote() for reservation in head_reservations]
            )

            secondary_reservations: list[ActorHandle] = []
            for placement, base_port in zip(placements, base_ports, strict=True):
                for node in placement.nodes[1:]:
                    node_port = base_port + node.node_rank * node.local_world_size
                    reservation = reservation_cls.options(
                        num_cpus=0,
                        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                            node_id=node.node_id,
                            soft=False,
                        ),
                    ).remote(node.host_ip, node_port, node.local_world_size)
                    secondary_reservations.append(reservation)
                    reservations.append(reservation)
            if secondary_reservations:
                await asyncio.gather(
                    *[reservation.get_port.remote() for reservation in secondary_reservations]
                )
            return base_ports, reservations
        except BaseException:
            for reservation in reservations:
                ray.kill(reservation, no_restart=True)
            raise

    @staticmethod
    async def _launch_pd_engine(
        engine_servers: list[ActorHandle],
        master: tuple[str, int, int],
    ) -> None:
        master_address, master_port, dp_rpc_port = master
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address,
                    master_port=master_port,
                    dp_rpc_port=dp_rpc_port,
                )
                for server in engine_servers
            ]
        )

    def _pd_actor_name(self, placement: PDEnginePlacement, node_rank: int) -> str:
        return (
            f"vllm_server_{placement.role}_{self.replica_rank}_{placement.index}_{node_rank}"
            f"{self.name_suffix}"
        )

    @staticmethod
    def _format_address(host: str, port: int, *, scheme: bool = False) -> str:
        authority = f"[{host}]:{port}" if is_valid_ipv6_address(host) else f"{host}:{port}"
        return f"http://{authority}" if scheme else authority

    def get_request_server_endpoints(self) -> list[tuple[str, ActorHandle]]:
        """Expose every prefill server to the session-aware request router."""
        if not self._prefill_servers or len(self._prefill_server_addresses) != len(self._prefill_servers):
            raise RuntimeError("PD prefill servers have not been launched")
        return list(zip(self._prefill_server_addresses, self._prefill_servers, strict=True))

    async def sleep(self):
        """Drain PD requests, then sleep all P/D servers."""
        await asyncio.gather(
            *[server.wait_for_requests_to_drain.remote() for _, server in self.get_request_server_endpoints()]
        )
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    def get_metrics_server_endpoints(self) -> list[tuple[str, dict[str, Any]]]:
        """Expose all P/D vLLM metrics endpoints without changing request routing."""
        if len(self._prefill_server_addresses) != len(self._prefill_servers):
            raise RuntimeError("PD prefill metrics endpoints are not ready")
        if len(self._decode_server_addresses) != len(self._decode_servers):
            raise RuntimeError("PD decode metrics endpoints are not ready")
        return [
            *[
                (
                    address,
                    {"request_endpoint": index, "pd_role": "prefill", "pd_index": index},
                )
                for index, address in enumerate(self._prefill_server_addresses)
            ],
            *[
                (
                    address,
                    {"request_endpoint": -1, "pd_role": "decode", "pd_index": index},
                )
                for index, address in enumerate(self._decode_server_addresses)
            ],
        ]

    @staticmethod
    def _is_ascend_platform() -> bool:
        """Follow Verl's existing NPU availability convention."""
        return is_torch_npu_available(check_device=False)

    @staticmethod
    def _build_kv_transfer_config(
        role: str,
        engine_id: str,
        transfer_backend: str,
        mooncake_protocol: Optional[str] = None,
        use_ascend_mooncake_v1: bool = False,
        kv_port: Optional[int] = None,
        prefill_tp: Optional[int] = None,
        prefill_dp: Optional[int] = None,
        decode_tp: Optional[int] = None,
        decode_dp: Optional[int] = None,
    ) -> dict:
        """Assemble vLLM's ``--kv-transfer-config`` payload."""
        role_to_kv_role = {
            "prefill": "kv_producer",
            "decode": "kv_consumer",
        }
        if use_ascend_mooncake_v1:
            if transfer_backend != "mooncake":
                raise ValueError("Ascend PD requires transfer_backend='mooncake'")
            if any(value is None for value in (kv_port, prefill_tp, prefill_dp, decode_tp, decode_dp)):
                raise ValueError(
                    "MooncakeConnectorV1 requires kv_port plus prefill/decode TP and DP sizes"
                )
            connector = "MooncakeConnectorV1"
        else:
            connector = {
                "nixl": "NixlConnector",
                "mooncake": "MooncakeConnector",
            }[transfer_backend]
        cfg: dict = {
            "kv_connector": connector,
            "kv_role": role_to_kv_role[role],
            "engine_id": engine_id,
            "kv_buffer_device": get_device_name(),
        }
        if use_ascend_mooncake_v1:
            cfg["kv_port"] = kv_port
            cfg["kv_connector_extra_config"] = {
                "prefill": {"dp_size": prefill_dp, "tp_size": prefill_tp},
                "decode": {"dp_size": decode_dp, "tp_size": decode_tp},
            }
        elif transfer_backend == "mooncake" and mooncake_protocol:
            cfg["kv_connector_extra_config"] = {"mooncake_protocol": mooncake_protocol}
        return cfg

    def _build_pd_role_config(self, role: str, tp: int, dp: int) -> RolloutConfig:
        """Apply role-local vLLM settings without mutating the shared config."""
        if role not in ("prefill", "decode"):
            raise ValueError(f"unknown PD role: {role!r}")

        disagg = self.config.disaggregation
        role_gpu_memory_utilization = (
            disagg.prefill_gpu_memory_utilization
            if role == "prefill"
            else disagg.decode_gpu_memory_utilization
        )
        role_engine_kwargs = (
            disagg.prefill_engine_kwargs if role == "prefill" else disagg.decode_engine_kwargs
        )
        engine_kwargs = copy.deepcopy(self.config.engine_kwargs)
        global_vllm_kwargs = engine_kwargs.get("vllm", {}) or {}
        role_engine_kwargs = _drop_none_values(role_engine_kwargs or {})
        engine_kwargs["vllm"] = _deep_merge_dict(global_vllm_kwargs, role_engine_kwargs)

        return _dc_replace(
            self.config,
            tensor_model_parallel_size=tp,
            data_parallel_size=dp,
            gpu_memory_utilization=(
                role_gpu_memory_utilization
                if role_gpu_memory_utilization is not None
                else self.config.gpu_memory_utilization
            ),
            engine_kwargs=engine_kwargs,
        )

    def _spawn_pd_server(
        self,
        role: str,
        pd_index: int,
        workers: list[ActorHandle],
        node_id: str,
        node_rank: int,
        cuda_visible_devices: str,
        tp: int,
        dp: int,
        nnodes: int,
        kv_transfer_config: dict,
        side_channel_host: str,
        side_channel_port: int,
        mooncake_bootstrap_port: int,
        actor_name: str,
        zmq_base_trainer_rank: int = 0,
    ) -> ActorHandle:
        """Construct one PD ``vLLMHttpServer`` actor."""
        per_role_config = self._build_pd_role_config(role, tp, dp)

        env_vars = {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "NCCL_CUMEM_ENABLE": "0",
            "VLLM_NIXL_SIDE_CHANNEL_HOST": side_channel_host,
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel_port),
            "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(mooncake_bootstrap_port),
            # Avoid Mooncake TCP port exhaustion under validation concurrency.
            "MC_TCP_ENABLE_CONNECTION_POOL": os.environ.get("MC_TCP_ENABLE_CONNECTION_POOL", "1"),
            "VERL_ZMQ_BASE_TRAINER_RANK": str(zmq_base_trainer_rank),
            "VERL_RAY_JOB_ID": ray.get_runtime_context().get_job_id(),
        }

        return self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": env_vars},
            name=actor_name,
            max_concurrency=self.max_concurrency,
        ).remote(
            config=per_role_config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=workers,
            replica_rank=self.replica_rank,
            node_rank=node_rank,
            gpus_per_node=len(workers),
            nnodes=nnodes,
            cuda_visible_devices=cuda_visible_devices,
            disaggregation_role=role,
            disaggregation_index=pd_index,
            disaggregation_kv_transfer_config=kv_transfer_config,
        )
