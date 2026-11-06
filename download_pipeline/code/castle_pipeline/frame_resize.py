from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .ffmpeg_utils import (
    build_ffmpeg_resize_frame_command,
    probe_image,
)
from .progress import atomic_write_text
from .time_utils import utc_now_iso


OLD_FRAMES_DIR_NAME = "old_frames"
FRAME_RESIZE_DISABLED_STATUS = "disabled"
FRAME_RESIZE_PENDING_STATUS = "pending"


def frame_resize_enabled(cfg: Any) -> bool:
    return bool(getattr(cfg.processing, "resize_extracted_frames", False)) and expected_frame_width(cfg) is not None


def expected_frame_width(cfg: Any) -> int | None:
    if not bool(getattr(cfg.processing, "resize_extracted_frames", False)):
        return None
    value = int(getattr(cfg.processing, "resized_frame_width", 0) or 0)
    return value if value > 0 else None


def expected_frame_width_for_video(cfg: Any, video_dir: Path) -> int | None:
    target_width = expected_frame_width(cfg)
    if target_width is None:
        return None
    if _metadata_says_resize_disabled(_read_summary(video_dir)):
        return None
    return target_width


def with_resize_pending_metadata(cfg: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    target_width = expected_frame_width(cfg)
    if not frame_resize_enabled(cfg) or target_width is None:
        return _with_resize_disabled_metadata(metadata)
    source_width = metadata.get("width") or getattr(cfg.processing, "frame_width", "")
    updated = dict(metadata)
    updated.update(
        {
            "frame_resize_enabled": True,
            "frame_resize_status": FRAME_RESIZE_PENDING_STATUS,
            "frame_resize_source_width": source_width,
            "frame_resize_target_width": target_width,
            "frame_width_after_resize": "",
            "frame_resize_command": "",
            "frame_resize_started_at": "",
            "frame_resize_finished_at": "",
            "frame_resize_original_frames_dir": "",
            "frame_resize_verified_at": "",
        }
    )
    return updated


def frame_resize_recovery_available(video_dir: Path) -> bool:
    return (video_dir / OLD_FRAMES_DIR_NAME).is_dir()


def processed_frames_need_resize(cfg: Any, video_dir: Path) -> bool:
    target_width = expected_frame_width(cfg)
    if not frame_resize_enabled(cfg) or target_width is None:
        return False
    if frame_resize_recovery_available(video_dir):
        return True
    frames_dir = video_dir / "frames"
    if not frames_dir.is_dir():
        return False
    required_artifacts = [
        video_dir / "frame_manifest.csv",
        video_dir / "frame_manifest.jsonl",
        video_dir / "video_processing_summary.json",
        video_dir / "verify_result.json",
    ]
    if any(not path.exists() for path in required_artifacts):
        return False
    summary = _read_summary(video_dir)
    if _metadata_says_resize_disabled(summary):
        return False
    if _metadata_says_resize_complete(summary, target_width) and _sample_width_ok(cfg, frames_dir, target_width):
        return False
    return True


def ensure_video_frames_resized(
    cfg: Any,
    video_dir: Path,
    remote_path: str,
    metadata: dict[str, Any],
    logger: Any,
    progress_callback: Callable[[int, str | None, float], None] | None = None,
) -> dict[str, Any]:
    target_width = expected_frame_width(cfg)
    if not frame_resize_enabled(cfg) or target_width is None:
        return _with_resize_disabled_metadata(metadata)

    frames_dir = video_dir / "frames"
    source_dir = video_dir / OLD_FRAMES_DIR_NAME
    started_at = utc_now_iso()
    source_width: int | None = None
    command_text = str(metadata.get("frame_resize_command") or "")

    if source_dir.exists():
        source_frames = _jpgs(source_dir)
        if not source_frames:
            raise RuntimeError(f"old frame source dir has no jpg files: {source_dir}")
        source_width = _sample_width(cfg, source_frames)
        command_text = command_text or _read_resize_command(video_dir) or _resize_command_for_source_frames(
            cfg.tools.ffmpeg,
            source_dir,
            frames_dir,
            target_width,
            source_frames,
        )
        if frames_dir.exists():
            if _folders_match_by_name(source_dir, frames_dir) and _sample_width_ok(cfg, frames_dir, target_width):
                try:
                    _copy_non_jpg_entries(source_dir, frames_dir)
                    _verify_resized_folder(cfg, source_dir, frames_dir, target_width)
                    if command_text:
                        atomic_write_text(video_dir / "frame_resize_command.txt", command_text + "\n")
                    return _with_resize_metadata(
                        metadata,
                        status="completed",
                        source_width=source_width,
                        target_width=target_width,
                        command_text=command_text,
                        started_at=str(metadata.get("frame_resize_started_at") or started_at),
                        finished_at=str(metadata.get("frame_resize_finished_at") or utc_now_iso()),
                        source_dir=source_dir,
                    )
                except RuntimeError as exc:
                    logger.warning("existing generated resize output failed verification and will be regenerated: %s", exc)
            _remove_incomplete_generated_frames(frames_dir, source_dir, logger)
    else:
        if not frames_dir.exists():
            raise FileNotFoundError(f"frames dir missing before resize: {frames_dir}")
        current_frames = _jpgs(frames_dir)
        if not current_frames:
            raise RuntimeError(f"frames dir has no jpg files before resize: {frames_dir}")
        source_width = _sample_width(cfg, current_frames)
        if source_width == target_width:
            return _with_resize_metadata(
                metadata,
                status="already_target_width",
                source_width=source_width,
                target_width=target_width,
                command_text=command_text or _read_resize_command(video_dir),
                started_at=started_at,
                finished_at=utc_now_iso(),
                source_dir=None,
            )
        shutil.move(str(frames_dir), str(source_dir))
        logger.info("moved original frames for resize %s -> %s", frames_dir, source_dir)
        source_frames = _jpgs(source_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)
    source_frames = _jpgs(source_dir)
    command_text = _resize_frame_command_template(cfg.tools.ffmpeg, source_dir, frames_dir, target_width)
    _resize_frames_with_workers(cfg, source_frames, frames_dir, target_width, remote_path, logger, progress_callback)

    _copy_non_jpg_entries(source_dir, frames_dir)
    _verify_resized_folder(cfg, source_dir, frames_dir, target_width)
    finished_at = utc_now_iso()
    atomic_write_text(video_dir / "frame_resize_command.txt", command_text + "\n")
    return _with_resize_metadata(
        metadata,
        status="completed",
        source_width=source_width,
        target_width=target_width,
        command_text=command_text,
        started_at=started_at,
        finished_at=finished_at,
        source_dir=source_dir,
    )


def cleanup_frame_resize_source(cfg: Any, video_dir: Path, logger: Any) -> None:
    target_width = expected_frame_width(cfg)
    source_dir = video_dir / OLD_FRAMES_DIR_NAME
    frames_dir = video_dir / "frames"
    if not source_dir.exists():
        return
    if target_width is None:
        raise RuntimeError(f"refusing to remove {source_dir}; target width is not configured")
    _verify_resized_folder(cfg, source_dir, frames_dir, target_width)
    shutil.rmtree(source_dir)
    logger.info("removed verified pre-resize frames dir %s", source_dir)


def _read_summary(video_dir: Path) -> dict[str, Any]:
    path = video_dir / "video_processing_summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _metadata_says_resize_complete(metadata: dict[str, Any], target_width: int) -> bool:
    if metadata.get("frame_resize_enabled") is not True:
        return False
    if metadata.get("frame_resize_status") not in {"completed", "already_target_width"}:
        return False
    try:
        recorded_target = int(metadata.get("frame_resize_target_width") or 0)
        recorded_after = int(metadata.get("frame_width_after_resize") or 0)
    except (TypeError, ValueError):
        return False
    return recorded_target == target_width and recorded_after == target_width


def _metadata_says_resize_disabled(metadata: dict[str, Any]) -> bool:
    if metadata.get("frame_resize_enabled") is False:
        return True
    status = str(metadata.get("frame_resize_status") or "").strip().lower()
    return status == FRAME_RESIZE_DISABLED_STATUS


def _with_resize_disabled_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    updated = dict(metadata)
    updated.update(
        {
            "frame_resize_enabled": False,
            "frame_resize_status": FRAME_RESIZE_DISABLED_STATUS,
        }
    )
    return updated


def _with_resize_metadata(
    metadata: dict[str, Any],
    status: str,
    source_width: int | None,
    target_width: int,
    command_text: str,
    started_at: str,
    finished_at: str,
    source_dir: Path | None,
) -> dict[str, Any]:
    updated = dict(metadata)
    updated.update(
        {
            "frame_resize_enabled": True,
            "frame_resize_status": status,
            "frame_resize_source_width": source_width or "",
            "frame_resize_target_width": target_width,
            "frame_width_after_resize": target_width,
            "frame_resize_command": command_text,
            "frame_resize_started_at": started_at,
            "frame_resize_finished_at": finished_at,
            "frame_resize_original_frames_dir": str(source_dir) if source_dir else "",
            "frame_resize_verified_at": utc_now_iso(),
        }
    )
    return updated


def _jpgs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jpg"))


def _sample_width(cfg: Any, frames: list[Path]) -> int | None:
    for frame in _sample_frames(frames):
        width, _height = _image_size(cfg, frame)
        if width is not None:
            return width
    return None


def _sample_width_ok(cfg: Any, frames_dir: Path, target_width: int) -> bool:
    frames = _jpgs(frames_dir)
    if not frames:
        return False
    for frame in _sample_frames(frames):
        width, _height = _image_size(cfg, frame)
        if width != target_width:
            return False
    return True


def _sample_frames(frames: list[Path]) -> list[Path]:
    if len(frames) <= 3:
        return frames
    middle = frames[len(frames) // 2]
    return [frames[0], middle, frames[-1]]


def _image_size(cfg: Any, path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except ImportError:
        probe = probe_image(cfg.tools.ffprobe, path)
        return probe.width, probe.height


def _folders_match_by_name(source_dir: Path, output_dir: Path) -> bool:
    return [path.name for path in _jpgs(source_dir)] == [path.name for path in _jpgs(output_dir)]


def _verify_resized_folder(cfg: Any, source_dir: Path, output_dir: Path, target_width: int) -> None:
    source_frames = _jpgs(source_dir)
    output_frames = _jpgs(output_dir)
    if not source_frames:
        raise RuntimeError(f"resize source has no jpg files: {source_dir}")
    if [path.name for path in source_frames] != [path.name for path in output_frames]:
        raise RuntimeError(
            f"resized frames do not match source names: source={len(source_frames)} output={len(output_frames)}"
        )
    empty = [path.name for path in output_frames if path.stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"resized frame files are empty: {empty[:10]}")
    mismatched_widths: list[str] = []
    for frame in output_frames:
        width, _height = _image_size(cfg, frame)
        if width != target_width:
            mismatched_widths.append(f"{frame.name}:{width}")
    if mismatched_widths:
        raise RuntimeError(
            f"resized frame width is not {target_width}: {len(mismatched_widths)} mismatches in {output_dir}; "
            f"examples={mismatched_widths[:10]}"
        )
    _verify_preserved_non_jpg_entries(source_dir, output_dir)


def _copy_non_jpg_entries(source_dir: Path, output_dir: Path) -> None:
    for path in source_dir.rglob("*"):
        relative = path.relative_to(source_dir)
        target = output_dir / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if path.suffix.lower() == ".jpg":
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _verify_preserved_non_jpg_entries(source_dir: Path, output_dir: Path) -> None:
    missing: list[str] = []
    mismatched: list[str] = []
    for path in source_dir.rglob("*"):
        if path.is_dir() or path.suffix.lower() == ".jpg":
            continue
        relative = path.relative_to(source_dir)
        target = output_dir / relative
        if not target.exists():
            missing.append(str(relative))
            continue
        if not target.is_file() or target.stat().st_size != path.stat().st_size:
            mismatched.append(str(relative))
    if missing:
        raise RuntimeError(f"non-jpg frame-side files were not preserved: {missing[:10]}")
    if mismatched:
        raise RuntimeError(f"non-jpg frame-side files changed during resize: {mismatched[:10]}")


def _remove_incomplete_generated_frames(frames_dir: Path, source_dir: Path, logger: Any) -> None:
    if not source_dir.exists():
        raise RuntimeError(f"refusing to reset generated frames because source dir is missing: {source_dir}")
    if frames_dir.name != "frames":
        raise RuntimeError(f"refusing to reset unexpected generated frame dir: {frames_dir}")
    shutil.rmtree(frames_dir)
    logger.warning("removed incomplete generated resize output %s; preserved source %s", frames_dir, source_dir)


def partition_frames_for_resize_workers(frames: list[Path], worker_count: int) -> list[list[Path]]:
    workers = max(1, int(worker_count or 1))
    partitions: list[list[Path]] = [[] for _ in range(workers)]
    for index, frame in enumerate(frames):
        partitions[index % workers].append(frame)
    return partitions


def _resize_frames_with_workers(
    cfg: Any,
    source_frames: list[Path],
    output_dir: Path,
    target_width: int,
    remote_path: str,
    logger: Any,
    progress_callback: Callable[[int, str | None, float], None] | None,
) -> None:
    worker_count = max(1, int(getattr(cfg.processing, "frame_resize_workers", 6) or 6))
    partitions = partition_frames_for_resize_workers(source_frames, worker_count)
    log_every = max(1, int(getattr(cfg.logging, "command_progress_every_items", 25) or 25))

    def run_partition(worker_index: int, frames: list[Path]) -> int:
        for local_index, frame in enumerate(frames, start=1):
            output = output_dir / frame.name
            cmd = build_ffmpeg_resize_frame_command(cfg.tools.ffmpeg, frame, output, target_width)
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg frame resize failed for {remote_path} frame={frame.name}: {result.stderr or result.stdout}")
            if local_index % log_every == 0:
                logger.info(
                    "frame resize worker progress path=%s worker=%s completed=%s assigned=%s latest=%s",
                    remote_path,
                    worker_index,
                    local_index,
                    len(frames),
                    frame.name,
                )
        return len(frames)

    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="castle-frame-resize") as executor:
        futures = [
            executor.submit(run_partition, worker_index, frames)
            for worker_index, frames in enumerate(partitions)
            if frames
        ]
        for future in as_completed(futures):
            completed += future.result()
    logger.info("frame resize folder completed path=%s frames=%s workers=%s", remote_path, completed, worker_count)
    if progress_callback is not None:
        latest = source_frames[-1].name if source_frames else None
        progress_callback(completed, latest, 0.0)


def _resize_frame_command_template(ffmpeg_bin: str, source_dir: Path, output_dir: Path, target_width: int) -> str:
    cmd = build_ffmpeg_resize_frame_command(
        ffmpeg_bin,
        source_dir / "<frame>.jpg",
        output_dir / "<frame>.jpg",
        target_width,
    )
    return " ".join(str(part) for part in cmd)


def _resize_command_for_source_frames(
    ffmpeg_bin: str,
    source_dir: Path,
    output_dir: Path,
    target_width: int,
    source_frames: list[Path],
) -> str:
    return _resize_frame_command_template(ffmpeg_bin, source_dir, output_dir, target_width)


def _read_resize_command(video_dir: Path) -> str:
    path = video_dir / "frame_resize_command.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _safe_log_stem(remote_path: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in remote_path).strip("_")[:180]
