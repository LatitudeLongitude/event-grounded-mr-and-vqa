from __future__ import annotations

REPO_ID = "CASTLE-Dataset/CASTLE2024"

FILE_PENDING = "pending"
FILE_DOWNLOADING = "downloading"
FILE_DOWNLOADED_VERIFIED = "downloaded_verified"
FILE_COPIED_TO_PROCESSED = "copied_to_processed"
FILE_PROCESSED_VERIFIED = "processed_verified"

VIDEO_PENDING = "pending"
VIDEO_DOWNLOADING = "downloading"
VIDEO_DOWNLOADED_UNVERIFIED = "downloaded_unverified"
VIDEO_DOWNLOADED_VERIFIED = "downloaded_verified"
VIDEO_PROCESSING = "processing"
VIDEO_RESIZING = "resizing"
VIDEO_PROCESSED_UNVERIFIED = "processed_unverified"
VIDEO_PROCESSED_VERIFIED = "processed_verified"
VIDEO_DELETING_RAW = "deleting_raw"
VIDEO_RAW_DELETED_VERIFIED = "raw_deleted_verified"
VIDEO_DONE = "done"
VIDEO_FAILED_RETRYABLE = "failed_retryable"
VIDEO_FAILED_NEED_MANUAL = "failed_need_manual"

VIDEO_STATUSES = {
    VIDEO_PENDING,
    VIDEO_DOWNLOADING,
    VIDEO_DOWNLOADED_UNVERIFIED,
    VIDEO_DOWNLOADED_VERIFIED,
    VIDEO_PROCESSING,
    VIDEO_RESIZING,
    VIDEO_PROCESSED_UNVERIFIED,
    VIDEO_PROCESSED_VERIFIED,
    VIDEO_DELETING_RAW,
    VIDEO_RAW_DELETED_VERIFIED,
    VIDEO_DONE,
    VIDEO_FAILED_RETRYABLE,
    VIDEO_FAILED_NEED_MANUAL,
}

NON_VIDEO_STATUSES = {
    FILE_PENDING,
    FILE_DOWNLOADING,
    VIDEO_DOWNLOADED_UNVERIFIED,
    FILE_DOWNLOADED_VERIFIED,
    FILE_COPIED_TO_PROCESSED,
    FILE_PROCESSED_VERIFIED,
    VIDEO_FAILED_RETRYABLE,
    VIDEO_FAILED_NEED_MANUAL,
}

PROGRESS_SCHEMA_VERSION = 1

DEFAULT_CSV_FIELDS = [
    "repo_id",
    "remote_video_path",
    "original_video_filename",
    "processed_video_dir",
    "frame_rel_path",
    "frame_abs_path",
    "day",
    "source",
    "video_hour_stem",
    "frame_index",
    "timestamp_second",
    "timestamp_hms",
    "fps",
    "width",
    "jpg_q",
    "ffmpeg_command",
    "video_duration_seconds",
    "video_size_bytes",
    "download_started_at",
    "download_finished_at",
    "processing_started_at",
    "processing_finished_at",
    "raw_deleted_at",
    "sha256_or_size_check",
    "status",
]

FRAME_RESIZE_CSV_FIELDS = [
    "frame_resize_enabled",
    "frame_resize_status",
    "frame_resize_source_width",
    "frame_resize_target_width",
    "frame_width_after_resize",
    "frame_resize_command",
    "frame_resize_started_at",
    "frame_resize_finished_at",
    "frame_resize_original_frames_dir",
    "frame_resize_verified_at",
]
