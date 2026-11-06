from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .disk_utils import dir_size_bytes
from .logging_utils import heartbeat


@dataclass(frozen=True)
class VideoProbe:
    duration_seconds: float
    width: int | None
    height: int | None
    avg_frame_rate: str | None
    codec_name: str | None


@dataclass(frozen=True)
class ImageProbe:
    width: int | None
    height: int | None


def probe_video(ffprobe_bin: str, video_path: Path) -> VideoProbe:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{result.stderr or result.stdout}")
    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    video_stream = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), {})
    return VideoProbe(
        duration_seconds=duration,
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        avg_frame_rate=video_stream.get("avg_frame_rate"),
        codec_name=video_stream.get("codec_name"),
    )


def expected_frames(duration_seconds: float, fps: int) -> int:
    if duration_seconds <= 0:
        return 0
    return max(1, int(round(duration_seconds * fps)))


def probe_image(ffprobe_bin: str, image_path: Path) -> ImageProbe:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        str(image_path),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {image_path}:\n{result.stderr or result.stdout}")
    data = json.loads(result.stdout)
    video_stream = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), {})
    return ImageProbe(width=video_stream.get("width"), height=video_stream.get("height"))


def build_ffmpeg_extract_command(ffmpeg_bin: str, input_path: Path, frames_dir: Path, fps: int, width: int, jpg_q: int) -> list[str]:
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"fps={fps},scale={width}:-2",
        "-q:v",
        str(jpg_q),
        "-start_number",
        "0",
        str(frames_dir / "%06d.jpg"),
    ]


def build_ffmpeg_resize_frame_command(ffmpeg_bin: str, input_path: Path, output_path: Path, width: int) -> list[str]:
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        str(input_path),
        "-map_metadata",
        "-1",
        "-vf",
        f"scale={width}:-1",
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output_path),
    ]


def run_ffmpeg_with_heartbeat(
    cmd: list[str],
    log_path: Path,
    frames_dir: Path,
    cfg: Any,
    remote_path: str,
    logger: Any,
    progress_callback: Callable[[int, str | None, float], None] | None = None,
    heartbeat_include_dir_sizes: bool = True,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        last_size = dir_size_bytes(frames_dir)
        last_progress = time.monotonic()
        while True:
            rc = proc.poll()
            now = time.monotonic()
            current_size = dir_size_bytes(frames_dir)
            if current_size > last_size:
                last_size = current_size
                last_progress = now
            frames = sorted(frames_dir.glob("*.jpg"))
            latest_frame = frames[-1].name if frames else None
            if progress_callback is not None:
                progress_callback(len(frames), latest_frame, now - last_progress)
            if now - last_progress > max(cfg.huggingface.stall_timeout_seconds, 60):
                proc.kill()
                raise RuntimeError(f"ffmpeg stalled for {remote_path}; see {log_path}")
            heartbeat(
                cfg.heartbeat_log,
                cfg.paths.raw_cache_dir,
                cfg.paths.processed_data_dir,
                remote_path,
                "processing",
                "ffmpeg still active",
                downloaded_bytes=current_size,
                frame_count=len(frames),
                latest_frame_name=latest_frame,
                last_progress_seconds_ago=now - last_progress,
                logger=logger,
                dir_size_refresh_interval_seconds=cfg.logging.heartbeat_dir_size_refresh_interval_seconds,
                include_dir_sizes=heartbeat_include_dir_sizes,
            )
            if rc is not None:
                break
            time.sleep(cfg.huggingface.heartbeat_interval_seconds)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {remote_path}; see {log_path}")
