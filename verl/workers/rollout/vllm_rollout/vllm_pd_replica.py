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
"""vLLM PD-disaggregated replica with M prefill and N decode servers.

Prefill servers are independent request ingress points and share the decode
pool. Asymmetric TP is supported. Each engine must fit on one node, while the
composite PD replica may span multiple nodes.
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
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

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


def _reserve_pd_port(worker) -> tuple[str, int]:
    """Reserve a side-channel port in the worker's node-local process."""
    host = ray.util.get_node_ip_address().strip("[]")
    port, sock = get_free_port(host, with_alive_sock=True)
    reservations = getattr(worker, "_verl_pd_port_reservations", [])
    reservations.append(sock)
    worker._verl_pd_port_reservations = reservations
    return host, port


def _release_pd_ports(worker) -> None:
    """Release ports reserved by :func:`_reserve_pd_port`."""
    for sock in getattr(worker, "_verl_pd_port_reservations", []):
        sock.close()
    worker._verl_pd_port_reservations = []


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

        self._prefill_tp = self.config.tensor_model_parallel_size
        # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
        self._decode_tp = (
            disagg.decode_tensor_model_parallel_size
            if disagg.decode_tensor_model_parallel_size is not None
            else self._prefill_tp
        )

        pd_world_size = self._n_prefill * self._prefill_tp + self._n_decode * self._decode_tp
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

        self._prefill_servers: list[ActorHandle] = []
        self._decode_servers: list[ActorHandle] = []
        self._prefill_server_addresses: list[str] = []
        self._decode_server_addresses: list[str] = []

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker count {len(self.workers)} != PD world size {self.world_size}"
        )
        use_ascend_mooncake_v1 = is_torch_npu_available(check_device=False)
        transfer_backend = self.config.disaggregation.transfer_backend
        cache_pool = self.config.cache_pool
        if cache_pool.enabled and not use_ascend_mooncake_v1:
            raise NotImplementedError(
                "PD cache_pool currently requires vLLM-Ascend with MooncakeConnectorV1"
            )
        cache_pool_config = {
            "enabled": cache_pool.enabled,
            "backend": cache_pool.backend,
            "consumer_is_to_put": cache_pool.consumer_is_to_put,
            "store_decode_kv": cache_pool.store_decode_kv,
            "consumer_is_to_load": cache_pool.consumer_is_to_load,
            "load_async": cache_pool.load_async,
            "use_layerwise": cache_pool.use_layerwise,
            "extra_config": dict(cache_pool.extra_config),
        }
        if use_ascend_mooncake_v1 and transfer_backend == "nixl":
            logger.warning(
                "NixlConnector is not supported for Ascend PD; falling back to "
                "MooncakeConnectorV1"
            )
            transfer_backend = "mooncake"

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

        def engine_location(start: int, end: int, role: str, index: int) -> tuple[str, str]:
            node_ids = {info[0] for info in worker_infos[start:end]}
            if len(node_ids) != 1:
                raise NotImplementedError(
                    f"{role} replica {index} spans multiple nodes; each PD engine must fit on one node"
                )
            return worker_infos[start][0], worker_infos[start][2]

        prefill_engine_ids: list[str] = []
        prefill_side_channel_hosts: list[str] = []
        prefill_side_channel_ports: list[int] = []

        reservation_workers: list[ActorHandle] = []
        reservations_released = False
        try:
            for i in range(self._n_prefill):
                start = i * self._prefill_tp
                end = start + self._prefill_tp
                prefill_workers = self.workers[start:end]
                prefill_node_id, prefill_host_ip = engine_location(start, end, "prefill", i)
                prefill_devs = self._collect_cuda_devices(worker_infos[start:end])
                prefill_engine_id = uuid.uuid4().hex
                reserved_host, prefill_side_channel_port = await prefill_workers[0].__ray_call__.remote(
                    _reserve_pd_port
                )
                if reserved_host != prefill_host_ip:
                    raise RuntimeError(
                        f"prefill replica {i} worker moved nodes during launch: "
                        f"expected {prefill_host_ip}, got {reserved_host}"
                    )
                reservation_workers.append(prefill_workers[0])
                prefill_engine_ids.append(prefill_engine_id)
                prefill_side_channel_hosts.append(prefill_host_ip)
                prefill_side_channel_ports.append(prefill_side_channel_port)
                prefill_kv_cfg = self._build_kv_transfer_config(
                    role="prefill",
                    engine_id=prefill_engine_id,
                    transfer_backend=transfer_backend,
                    mooncake_protocol=self.config.disaggregation.mooncake_protocol,
                    use_ascend_mooncake_v1=use_ascend_mooncake_v1,
                    kv_port=prefill_side_channel_port,
                    prefill_tp=self._prefill_tp,
                    decode_tp=self._decode_tp,
                    cache_pool_config=cache_pool_config,
                )
                self._prefill_servers.append(
                    self._spawn_pd_server(
                        role="prefill",
                        workers=prefill_workers,
                        node_id=prefill_node_id,
                        cuda_visible_devices=prefill_devs,
                        tp=self._prefill_tp,
                        kv_transfer_config=prefill_kv_cfg,
                        side_channel_host=prefill_host_ip,
                        side_channel_port=prefill_side_channel_port,
                        mooncake_bootstrap_port=prefill_side_channel_port,
                        actor_name=f"vllm_server_{self.replica_rank}_{i}{self.name_suffix}",
                        zmq_base_trainer_rank=start,
                    )
                )

            for i in range(self._n_decode):
                start = self._n_prefill * self._prefill_tp + i * self._decode_tp
                end = start + self._decode_tp
                workers_i = self.workers[start:end]
                node_id_i, decode_host_ip = engine_location(start, end, "decode", i)
                devs_i = self._collect_cuda_devices(worker_infos[start:end])

                reserved_host, decode_side_channel_port = await workers_i[0].__ray_call__.remote(_reserve_pd_port)
                if reserved_host != decode_host_ip:
                    raise RuntimeError(
                        f"decode replica {i} worker moved nodes during launch: "
                        f"expected {decode_host_ip}, got {reserved_host}"
                    )
                reservation_workers.append(workers_i[0])
                decode_kv_cfg = self._build_kv_transfer_config(
                    role="decode",
                    engine_id=uuid.uuid4().hex,
                    transfer_backend=transfer_backend,
                    mooncake_protocol=self.config.disaggregation.mooncake_protocol,
                    use_ascend_mooncake_v1=use_ascend_mooncake_v1,
                    kv_port=decode_side_channel_port,
                    prefill_tp=self._prefill_tp,
                    decode_tp=self._decode_tp,
                    cache_pool_config=cache_pool_config,
                )
                self._decode_servers.append(
                    self._spawn_pd_server(
                        role="decode",
                        workers=workers_i,
                        node_id=node_id_i,
                        cuda_visible_devices=devs_i,
                        tp=self._decode_tp,
                        kv_transfer_config=decode_kv_cfg,
                        side_channel_host=decode_host_ip,
                        side_channel_port=decode_side_channel_port,
                        mooncake_bootstrap_port=decode_side_channel_port,
                        actor_name=f"vllm_server_decode_{self.replica_rank}_{i}{self.name_suffix}",
                        zmq_base_trainer_rank=start,
                    )
                )

            await asyncio.gather(
                *[worker.__ray_call__.remote(_release_pd_ports) for worker in reservation_workers]
            )
            reservations_released = True
            await asyncio.gather(
                *[
                    server.launch_server.remote(master_address=None, master_port=None, dp_rpc_port=None)
                    for server in self._prefill_servers + self._decode_servers
                ]
            )
        finally:
            if not reservations_released:
                await asyncio.gather(
                    *[worker.__ray_call__.remote(_release_pd_ports) for worker in reservation_workers],
                    return_exceptions=True,
                )

        await asyncio.gather(
            *[
                prefill_server.set_pd_peer.remote(
                    decode_peers=self._decode_servers,
                    prefill_side_channel_port=prefill_side_channel_ports[index],
                    prefill_engine_id=prefill_engine_ids[index],
                )
                for index, prefill_server in enumerate(self._prefill_servers)
            ]
        )

        self.servers = list(self._prefill_servers) + list(self._decode_servers)
        prefill_addresses = await asyncio.gather(
            *[server.get_server_address.remote() for server in self._prefill_servers]
        )
        decode_addresses = await asyncio.gather(
            *[server.get_server_address.remote() for server in self._decode_servers]
        )
        self._prefill_server_addresses = [
            f"[{host}]:{port}" if is_valid_ipv6_address(host) else f"{host}:{port}"
            for host, port in prefill_addresses
        ]
        self._decode_server_addresses = [
            f"[{host}]:{port}" if is_valid_ipv6_address(host) else f"{host}:{port}"
            for host, port in decode_addresses
        ]
        self._server_handle = self._prefill_servers[0]
        self._server_address = self._prefill_server_addresses[0]

        logger.info(
            "vLLMPDReplica rank=%s launched: prefills=%s, decodes=%s",
            self.replica_rank,
            self._prefill_server_addresses,
            self._decode_server_addresses,
        )

    def get_request_server_endpoints(self) -> list[tuple[str, ActorHandle]]:
        """Expose every prefill server to the session-aware request router."""
        if not self._prefill_servers or len(self._prefill_server_addresses) != len(self._prefill_servers):
            raise RuntimeError("PD prefill servers have not been launched")
        return list(zip(self._prefill_server_addresses, self._prefill_servers, strict=True))

    async def sleep(self):
        """Drain PD requests, disconnect peers, then sleep all P/D servers."""
        await asyncio.gather(
            *[server.wait_for_requests_to_drain.remote() for _, server in self.get_request_server_endpoints()]
        )
        await asyncio.gather(
            *[
                server.collective_rpc.remote(method="disconnect_kv_transfer_peers")
                for server in self.servers
            ]
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
    def _collect_cuda_devices(worker_infos) -> str:
        return ",".join(worker_info[1] for worker_info in worker_infos)

    @staticmethod
    def _build_kv_transfer_config(
        role: str,
        engine_id: str,
        transfer_backend: str,
        mooncake_protocol: Optional[str] = None,
        use_ascend_mooncake_v1: bool = False,
        kv_port: Optional[int] = None,
        prefill_tp: Optional[int] = None,
        decode_tp: Optional[int] = None,
        cache_pool_config: Optional[dict] = None,
    ) -> dict:
        """Assemble vLLM's ``--kv-transfer-config`` payload."""
        role_to_kv_role = {
            "prefill": "kv_producer",
            "decode": "kv_consumer",
        }
        if use_ascend_mooncake_v1:
            if transfer_backend != "mooncake":
                raise ValueError("Ascend PD requires transfer_backend='mooncake'")
            if kv_port is None or prefill_tp is None or decode_tp is None:
                raise ValueError(
                    "MooncakeConnectorV1 requires kv_port, prefill_tp, and decode_tp"
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
                "prefill": {"dp_size": 1, "tp_size": prefill_tp},
                "decode": {"dp_size": 1, "tp_size": decode_tp},
            }
            if cache_pool_config and cache_pool_config.get("enabled", False):
                mooncake_cfg = dict(cfg)
                store_extra = dict(cache_pool_config.get("extra_config", {}))
                store_extra.update(
                    {
                        "backend": cache_pool_config.get("backend", "mooncake"),
                        # AscendStore uses this value as an IPC path suffix, not
                        # as a TCP listener. The engine id prevents collisions
                        # between multiple P/D engines on the same node.
                        "lookup_rpc_port": engine_id,
                        "consumer_is_to_put": bool(
                            role == "decode"
                            and cache_pool_config.get("consumer_is_to_put", False)
                        ),
                        # vLLM-Ascend 0.23 names Decode-side incremental
                        # writeback ``save_decode_cache``.
                        "save_decode_cache": bool(
                            role == "decode"
                            and cache_pool_config.get("store_decode_kv", False)
                        ),
                        "consumer_is_to_load": bool(
                            role == "decode"
                            and cache_pool_config.get("consumer_is_to_load", False)
                        ),
                        "load_async": bool(cache_pool_config.get("load_async", False)),
                        "use_layerwise": bool(
                            cache_pool_config.get("use_layerwise", False)
                        ),
                    }
                )
                store_cfg = {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": role_to_kv_role[role],
                    "kv_connector_extra_config": store_extra,
                }
                cfg = {
                    "kv_connector": "MultiConnector",
                    "kv_role": role_to_kv_role[role],
                    "engine_id": engine_id,
                    "kv_buffer_device": get_device_name(),
                    "kv_load_failure_policy": "recompute",
                    "kv_connector_extra_config": {
                        # Direct P-to-D transfer wins for the current turn;
                        # AscendStore supplies shared-prefix hits on later turns.
                        "connectors": [mooncake_cfg, store_cfg]
                    },
                }
        elif transfer_backend == "mooncake" and mooncake_protocol:
            cfg["kv_connector_extra_config"] = {"mooncake_protocol": mooncake_protocol}
        return cfg

    def _build_pd_role_config(self, role: str, tp: int) -> RolloutConfig:
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
        workers: list[ActorHandle],
        node_id: str,
        cuda_visible_devices: str,
        tp: int,
        kv_transfer_config: dict,
        side_channel_host: str,
        side_channel_port: int,
        mooncake_bootstrap_port: int,
        actor_name: str,
        zmq_base_trainer_rank: int = 0,
    ) -> ActorHandle:
        """Construct one PD ``vLLMHttpServer`` actor."""
        per_role_config = self._build_pd_role_config(role, tp)

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
            node_rank=0,
            gpus_per_node=self.gpus_per_replica_node,
            nnodes=1,
            cuda_visible_devices=cuda_visible_devices,
            disaggregation_role=role,
            disaggregation_kv_transfer_config=kv_transfer_config,
        )
