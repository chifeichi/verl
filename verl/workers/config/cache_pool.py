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
from dataclasses import dataclass, field
from typing import Any

from verl.base_config import BaseConfig

__all__ = ["KVCachePoolConfig"]

_ALLOWED_CACHE_POOL_BACKENDS = ("mooncake", "memcache", "yuanrong")


@dataclass
class KVCachePoolConfig(BaseConfig):
    """AscendStore-backed shared KV cache for rollout engines.

    Field names intentionally mirror ``AscendStoreConnector``. Backend-
    specific options are forwarded through ``extra_config`` without Verl
    translating or interpreting them.
    """

    enabled: bool = False
    backend: str = "mooncake"
    consumer_is_to_put: bool = False
    store_decode_kv: bool = False
    consumer_is_to_load: bool = False
    load_async: bool = False
    use_layerwise: bool = False
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enabled and self.backend not in _ALLOWED_CACHE_POOL_BACKENDS:
            raise ValueError(f"cache_pool.backend={self.backend!r} not in {_ALLOWED_CACHE_POOL_BACKENDS}")
