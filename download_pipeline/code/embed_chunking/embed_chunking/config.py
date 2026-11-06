from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    min_token_estimate: int = 80
    target_min_token_estimate: int = 150
    target_max_token_estimate: int = 400
    soft_max_token_estimate: int = 600
    hard_max_token_estimate: int = 800

    min_duration_sec: float = 15.0
    target_min_duration_sec: float = 30.0
    target_max_duration_sec: float = 90.0
    soft_max_duration_sec: float = 120.0
    hard_max_duration_sec: float = 180.0

    overlap_tokens: int = 0
    overlap_seconds: float = 0.0
    chunking_version: str = "1.0"

    def validate(self) -> None:
        if self.hard_max_token_estimate <= 0:
            raise ValueError("hard_max_token_estimate must be positive")
        if self.target_max_token_estimate > self.hard_max_token_estimate:
            raise ValueError("target_max_token_estimate cannot exceed hard_max_token_estimate")
        if self.overlap_tokens > 100:
            raise ValueError("overlap_tokens must be <= 100")
        if self.overlap_seconds > 30:
            raise ValueError("overlap_seconds must be <= 30")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
