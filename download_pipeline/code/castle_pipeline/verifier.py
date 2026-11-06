from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .manifests import artifact_tree_contains_processing_tmp
from .progress import atomic_write_json
from .time_utils import utc_now_iso


@dataclass
class VerifyResult:
    passed: bool
    frame_count: int = 0
    expected_frame_count: int | None = None
    tolerance: int = 0
    checked_at: str = field(default_factory=utc_now_iso)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return sum(1 for _ in reader)


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _try_open_sample_images(frames: list[Path], result: VerifyResult, sample_count: int = 5) -> None:
    try:
        from PIL import Image
    except ImportError:
        result.warnings.append("PIL is not installed; skipped sample image decode check")
        return
    sample = frames if len(frames) <= sample_count else random.sample(frames, sample_count)
    for frame in sample:
        try:
            with Image.open(frame) as image:
                image.verify()
        except Exception as exc:
            result.errors.append(f"frame image decode failed: {frame.name}: {exc}")


def _check_sample_image_width(frames: list[Path], expected_width: int, result: VerifyResult, sample_count: int = 5) -> None:
    try:
        from PIL import Image
    except ImportError:
        result.warnings.append("PIL is not installed; skipped sample image width check")
        return
    sample = frames if len(frames) <= sample_count else random.sample(frames, sample_count)
    for frame in sample:
        try:
            with Image.open(frame) as image:
                if image.width != expected_width:
                    result.errors.append(f"frame width check failed: {frame.name}: {image.width} != {expected_width}")
                    return
        except Exception as exc:
            result.errors.append(f"frame image width check failed: {frame.name}: {exc}")
            return


def verify_processed_video_dir(
    video_dir: Path,
    expected_frame_count: int | None,
    tolerance: int = 2,
    require_verify_result: bool = False,
    sample_images: bool = True,
    expected_frame_width: int | None = None,
) -> VerifyResult:
    result = VerifyResult(passed=False, expected_frame_count=expected_frame_count, tolerance=tolerance)
    frames_dir = video_dir / "frames"
    if not video_dir.exists():
        result.errors.append(f"video output dir missing: {video_dir}")
        return result
    if not frames_dir.exists():
        result.errors.append(f"frames dir missing: {frames_dir}")
        return result
    frames = sorted(frames_dir.glob("*.jpg"))
    result.frame_count = len(frames)
    if not frames:
        result.errors.append("no extracted frames found")
    empty = [frame.name for frame in frames if frame.stat().st_size <= 0]
    if empty:
        result.errors.append(f"empty frame files found: {empty[:10]}")
    if expected_frame_count is not None:
        if abs(len(frames) - expected_frame_count) > tolerance:
            result.errors.append(
                f"frame_count {len(frames)} outside tolerance {tolerance} for expected {expected_frame_count}"
            )
    csv_path = video_dir / "frame_manifest.csv"
    jsonl_path = video_dir / "frame_manifest.jsonl"
    summary_path = video_dir / "video_processing_summary.json"
    ffmpeg_cmd_path = video_dir / "ffmpeg_command.txt"
    for required in [csv_path, jsonl_path, summary_path, ffmpeg_cmd_path]:
        if not required.exists():
            result.errors.append(f"required file missing: {required.name}")
    if csv_path.exists():
        try:
            rows = _count_csv_rows(csv_path)
            if rows != len(frames):
                result.errors.append(f"csv manifest row count {rows} != frame count {len(frames)}")
        except Exception as exc:
            result.errors.append(f"csv manifest read failed: {exc}")
    if jsonl_path.exists():
        try:
            rows = _count_jsonl_rows(jsonl_path)
            if rows != len(frames):
                result.errors.append(f"jsonl manifest row count {rows} != frame count {len(frames)}")
        except Exception as exc:
            result.errors.append(f"jsonl manifest read failed: {exc}")
    if csv_path.exists() and frames:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    frame_abs = row.get("frame_abs_path", "")
                    frame_rel = row.get("frame_rel_path", "")
                    if frame_abs and not Path(frame_abs).exists():
                        if ".__processing_tmp__" in frame_abs:
                            result.errors.append(f"manifest frame_abs_path stale_processing_tmp_path: {frame_abs}")
                        else:
                            result.errors.append(f"manifest frame_abs_path missing: {frame_abs}")
                        break
                    if frame_rel and not _frame_rel_path_exists(video_dir, frame_rel):
                        result.errors.append(f"manifest frame_rel_path missing: {frame_rel}")
                        break
        except Exception as exc:
            result.errors.append(f"manifest path validation failed: {exc}")
    if not video_dir.name.endswith(".__processing_tmp__") and artifact_tree_contains_processing_tmp(video_dir):
        result.errors.append("final video artifacts contain stale .__processing_tmp__ paths")
    if sample_images and frames:
        _try_open_sample_images(frames, result)
    if expected_frame_width is not None and frames:
        _check_sample_image_width(frames, expected_frame_width, result)
    verify_result_path = video_dir / "verify_result.json"
    if require_verify_result:
        if not verify_result_path.exists():
            result.errors.append("verify_result.json missing")
        else:
            try:
                payload = json.loads(verify_result_path.read_text(encoding="utf-8"))
                if payload.get("passed") is not True:
                    result.errors.append("verify_result.json does not contain passed=true")
            except Exception as exc:
                result.errors.append(f"verify_result.json unreadable: {exc}")
    result.passed = not result.errors
    return result


def write_verify_result(video_dir: Path, result: VerifyResult) -> None:
    atomic_write_json(video_dir / "verify_result.json", result.to_dict())


def _frame_rel_path_exists(video_dir: Path, frame_rel: str) -> bool:
    rel = Path(frame_rel)
    for candidate_root in [video_dir, *video_dir.parents]:
        if candidate_root.name == "processed_data" and (candidate_root / rel).exists():
            return True
    return (video_dir / "frames" / rel.name).exists()
