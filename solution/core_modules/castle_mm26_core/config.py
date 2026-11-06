from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import canonical_json_hash, read_yaml


PROFILE_BLOCKS = ("recall_profiles", "scoring_profiles", "scoring_component_profiles", "threshold_profiles", "output_profiles")


@dataclass(frozen=True)
class AppConfig:
    path: Path
    data: dict[str, Any]
    repo_root: Path
    environment: dict[str, Any]
    model_profile: dict[str, Any]
    active_environment_profile: str = ""

    @property
    def mode(self) -> str:
        return str(self.data.get("mode"))

    @property
    def write_official_zip(self) -> bool:
        return bool(self.data.get("write_official_zip"))

    @property
    def sort_interval_rows_by_final_score(self) -> bool:
        submission = self.data.get("submission") or {}
        if not isinstance(submission, dict):
            return True
        return _config_bool(submission.get("sort_interval_rows_by_final_score"), default=True)

    @property
    def full_smoke_all_queries(self) -> bool:
        runtime = self.data.get("runtime") or {}
        if not isinstance(runtime, dict):
            return False
        return _config_bool(runtime.get("full_smoke_all_queries"), default=False)

    @property
    def profile_registry(self) -> dict[str, Any]:
        registry = self.data.get("profile_registry")
        if not isinstance(registry, dict):
            raise ValueError("config.profile_registry must be a mapping")
        return registry

    @property
    def profile_registry_sha256(self) -> str:
        return canonical_json_hash(self.profile_registry)

    @property
    def output_root(self) -> Path:
        return Path(str(self.environment["output_root"]))

    def input_path(self, key: str) -> Path:
        value = ((self.data.get("inputs") or {}).get(key) or "").strip()
        if not value:
            raise ValueError(f"missing inputs.{key}")
        return Path(self.resolve_env_value(value))

    def resolve_env_value(self, value: str) -> str:
        result = str(value)
        for match in re.finditer(r"\$\{environment\.([A-Za-z0-9_]+)\}", result):
            key = match.group(1)
            if key not in self.environment:
                raise ValueError(f"unknown environment profile key: {key}")
            result = result.replace(match.group(0), str(self.environment[key]))
        return result

    def resolve_profile(self, block: str, name: str) -> dict[str, Any]:
        if block not in PROFILE_BLOCKS:
            raise ValueError(f"unknown profile block: {block}")
        profiles = self.profile_registry.get(block)
        if not isinstance(profiles, dict):
            raise ValueError(f"profile_registry.{block} must be a mapping")
        profile = profiles.get(name)
        if not isinstance(profile, dict):
            raise ValueError(f"profile not found: {block}.{name}")
        return profile


def load_config(path: Path | str, *, active_environment: str | None = None) -> AppConfig:
    target = Path(path).resolve()
    data = read_yaml(target)
    required = [
        "schema_version",
        "mode",
        "backend",
        "allow_qdrant",
        "write_official_zip",
        "active_environment_profile",
        "runtime",
        "inputs",
        "environment_profiles",
        "model_profiles",
        "qa_mllm",
        "day_date_map",
        "preflight",
        "profile_registry",
        "submission",
        "official_sources",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"config missing required top-level keys: {missing}")
    if data.get("backend") != "parquet":
        raise ValueError("Fully-Automatic backend must be parquet")
    if data.get("allow_qdrant") is not False:
        raise ValueError("Fully-Automatic config must set allow_qdrant=false")
    if data.get("write_official_zip") is True and data.get("mode") != "fully_automatic":
        raise ValueError("official Fully-Automatic config must set mode=fully_automatic")
    if data.get("write_official_zip") is True and (data.get("qa_mllm") or {}).get("model") != "openai/gpt-5.5":
        raise ValueError("official Fully-Automatic config must set qa_mllm.model=openai/gpt-5.5")
    env_name = active_environment or str(data.get("active_environment_profile"))
    env_profiles = data.get("environment_profiles") or {}
    model_profiles = data.get("model_profiles") or {}
    environment = env_profiles.get(env_name)
    model_profile = model_profiles.get(env_name)
    if not isinstance(environment, dict):
        raise ValueError(f"environment profile not found: {env_name}")
    if not isinstance(model_profile, dict):
        raise ValueError(f"model profile not found: {env_name}")
    repo_root = Path(str(environment.get("repo_root", target.parents[2]))).resolve()
    config = AppConfig(
        path=target,
        data=data,
        repo_root=repo_root,
        environment=environment,
        model_profile=model_profile,
        active_environment_profile=env_name,
    )
    _validate_profile_registry(config)
    return config


def parse_devices(value: str | None, *, default: str = "0") -> list[str]:
    text = (value or default).strip()
    if not text:
        raise ValueError("device list must not be empty")
    devices = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("cuda:"):
            devices.append(item)
        elif item.isdigit():
            devices.append(f"cuda:{item}")
        elif item == "cpu":
            devices.append("cpu")
        else:
            raise ValueError(f"invalid device item: {item!r}")
    if not devices:
        raise ValueError("device list must not be empty")
    if len(devices) > 3:
        raise ValueError("this project supports at most three explicit GPUs")
    return devices


def _config_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _validate_profile_registry(config: AppConfig) -> None:
    registry = config.profile_registry
    for block in PROFILE_BLOCKS:
        if not isinstance(registry.get(block), dict) or not registry[block]:
            raise ValueError(f"profile_registry.{block} must be non-empty")
    if config.write_official_zip:
        text = str(registry)
        if "{}" in text:
            raise ValueError("official config cannot use empty profile placeholders")


def run_output_root(config: AppConfig, run_id: str) -> Path:
    if not run_id:
        raise ValueError("run id is required")
    return config.output_root / run_id
