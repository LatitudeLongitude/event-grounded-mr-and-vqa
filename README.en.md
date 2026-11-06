<p align="right"><a href="./README.zh.md"><kbd>简体中文</kbd></a> · <a href="./README.en.md"><kbd>English</kbd></a></p>

# Event-Evidence-Grounded Multimodal Retrieval and Video Question Answering

This project contains the companion code for the paper **“An Evidence Framework for Event-Grounded Multimodal Retrieval and Video Question Answering”** and was developed for the [Fully-Automatic Track of the 2nd CASTLE Multimodal Analytics Challenge](https://castle-dataset.github.io/challenges/mm26/#gear-fully-automatic-track).

## Project Overview

This project addresses event instance search (EIS), object instance search (OIS), and video question answering (QA) on the long-duration, multi-view [CASTLE2024](https://huggingface.co/datasets/CASTLE-Dataset/CASTLE2024) video dataset. The system organizes video frames and the speech transcripts supplied with the dataset into traceable evidence, preserving source-view and time information to locate events and objects and answer questions.

The overall workflow is: download and frame extraction → transcript cleaning and chunking → three-way feature encoding → L0–L5 hierarchical event construction → retrieval, reranking, and question answering. TIPSv2 provides the visual representation, while Qwen3 dense vectors and BGE-M3 sparse vectors provide the text representations. At query time, three-way retrieval, visual local-feature reranking, and text reranking select evidence. The event hierarchy then helps determine time intervals or organize QA context. For QA, the selected text and image evidence is sent to a multimodal model through OpenRouter, and the answer and its evidence citations are validated. The event hierarchy is used for evidence association and temporal localization.

| Directory | Function |
|---|---|
| [download_pipeline/code](download_pipeline/code/) | Downloads and verifies data, extracts frames, cleans transcripts, and chunks text. |
| [embedding_pipeline](embedding_pipeline/) | Separately generates TIPSv2, Qwen3, and BGE-M3 features. |
| [event_pipeline](event_pipeline/) | Builds micro-events, single-source events, cross-source events, activity episodes, and long sessions from frame and text atoms. |
| [solution](solution/) | Performs retrieval, reranking, question answering, and submission validation for the 36 challenge tasks; produces three EIS, OIS, and QA CSV files and a packaged ZIP. |

## Processing a Large Dataset with Less Storage

[download_pipeline/code](download_pipeline/code/) provides a “download, extract frames, then delete” pipeline. It processes one video at a time and overlaps the download, frame-extraction, and resizing stages. Once the extracted frames pass validation, the corresponding original video is deleted. The frame images and provenance records needed for later retrieval and question answering are retained, so the entire original-video collection need not be stored at once.

Using this approach, we completed the download and frame extraction for this large dataset with only about **600 GB of storage**. The current configuration extracts frames at 1 FPS, resizes images to 1920 pixels in width, and reduces image quality. We hope this approach is useful to researchers who want to use the CASTLE dataset but do not have enough storage to retain all the original videos.

## Where Effective Parameters Are Defined

All paths below are relative to this directory. The parameters that actually affect the project are distributed across YAML files, command-line options, and Python source code. A configuration field's name alone does not establish whether that field participates in a computation.

### Data Preparation and Event Construction

| Stage | Parameter location and effective scope |
|---|---|
| Download and frame extraction | [download_pipeline/code/config.yaml](download_pipeline/code/config.yaml) recursively overrides the defaults in [castle_pipeline/config.py](download_pipeline/code/castle_pipeline/config.py). It controls the dataset, paths, FFmpeg, frame rate, image dimensions, JPEG quality, concurrency, proxy, and `delete_raw_video_after_verified`. `run --no-resize-frames` can temporarily disable resizing. |
| Transcript cleaning | [transcript_cleaning/config.py](download_pipeline/code/transcript_cleaning/transcript_cleaning/config.py) defines the quality-flag thresholds for timestamp duration, which default to 1–120 seconds. Input and output paths come from this module's CLI; it does not read the downloader YAML. |
| Text chunking | [embed_chunking/config.py](download_pipeline/code/embed_chunking/embed_chunking/config.py) defines token and duration thresholds. The CLI can override `target_min_token_estimate`, `target_max_token_estimate`, and `hard_max_token_estimate`; their defaults are 150, 400, and 800, respectively. The target duration is 30–90 seconds, with a hard limit of 180 seconds. |
| TIPSv2 image encoding | [tipsv2_image_cls/code/config.yaml](embedding_pipeline/tipsv2_image_cls/code/config.yaml) controls the model path `paths.model_root`, image size (448), precision (FP32), per-GPU batch size (24), GPU selection, decoding, and output parameters. `--gpu-ids` and `--decode-workers` can override the corresponding settings; there is no dedicated environment-variable override for the model path. |
| Qwen3 text encoding | [qwen3_text_dense/code/config.yaml](embedding_pipeline/qwen3_text_dense/code/config.yaml) controls the model path, maximum length (32768), precision (float16), batch size (2), and runtime parameters. `QWEN3_MODEL_ROOT` takes precedence only when `model_root: auto`; if the environment variable is unset, the platform default path is used. |
| BGE-M3 text encoding | [bgem3_text_sparse/code/config.yaml](embedding_pipeline/bgem3_text_sparse/code/config.yaml) controls the model path, maximum length (8192), `use_fp16`, batch size (4), and runtime parameters. `BGEM3_MODEL_ROOT` takes precedence only when `model_root: auto`; if the environment variable is unset, the platform default path is used. GPU selection for both text encoders can be overridden with `--gpu-ids`, while batch size and encoding parameters for normal runs still come from YAML. |
| Event scale and view metadata | [event_pipeline/code/config.yaml](event_pipeline/code/config.yaml) controls input and output paths, read batch size, `max_event_frames`, `sparse_top_n`, the minimum L2 duration, and the maximum L2–L5 spans. In [view_meta.yaml](event_pipeline/config/view_meta.yaml), the effective fields are the view's role, `location_hint`, and `fusion_defaults.cross_source_time_slack_sec.l3_global`. |
| Event algorithms | In [event_pipeline/code/event_pipeline](event_pipeline/code/event_pipeline/), `visual_features.py`, `text_features.py`, `sparse_features.py`, and `boundary_projection.py` control boundary features and projection; the actual calculations in `fusion.py` define fusion weights; `segments_l1_l2.py`, `group_l3.py`, and `aggregate_l4_l5.py` control the construction of each level; and `representative_frames.py` controls representative-frame selection. The corresponding weights and decision thresholds must be changed in the source code. |

The chunking settings `soft_max_*` and `overlap_*` do not participate in the current chunking calculation. `runtime.max_retries` does not drive a retry loop for Qwen3 or BGE-M3, and BGE-M3's `runtime.records_per_shard` does not determine actual sharding. The event configuration's `rows_per_parquet_group`, weight hints in the view configuration, and L1/L2/L4/L5 time slack values are not current tuning controls for the computation. Although L4/L5 fusion boundary scores are saved, higher-level events are actually built by the continuity-based merging logic in `aggregate_l4_l5.py`.

### Retrieval, Scoring, and Question Answering

Every `config.yaml` in the table below refers to [solution/fully_automatic_track/config.yaml](solution/fully_automatic_track/config.yaml).

| Parameter category | Effective location |
|---|---|
| Environment, models, and API | `active_environment_profile` selects the `environment_profiles` and `model_profiles` with the same name, determining data paths, output paths, and local model paths. `qa_mllm` controls the OpenRouter model, endpoint, timeout, retries, response format, and evidence-count limits. `day_date_map` and `official_sources.challenge_url` are used for preflight checks. |
| Queries and per-task policies | [queries.yaml](solution/fully_automatic_track/queries.yaml) defines queries, positive and negative prompts, policy-profile selection, and time, person, activity, and counting rules. Task-type defaults are merged first, then fields from each task override them. |
| Retrieval and interval output | `profile_registry.recall_profiles` controls top-K, per-day/source/hour candidate reserves, text-reranking caps, and QA visual probes. `profile_registry.output_profiles` controls NMS IoU and the base merge gap, while the solver also applies per-task merging rules. `submission.sort_interval_rows_by_final_score` controls CSV row order. |
| Final EIS/OIS scoring | `INTERVAL_SCORE_SPECS` in [solver.py](solution/core_modules/castle_mm26_core/solver.py) defines per-task visual, positive-text, and negative-text weights, raw visual/text score floors, and required modalities. In `_calibrated_threshold`, both the final submission threshold and hard-rejection threshold for the combined score are 0; the main filters are raw-score floors and per-task rules. |
| QA evidence selection | `profile_registry.scoring_profiles.qa_evidence.evidence_weights` defines the base reranking, visual, and negative-text weights (0.75/0.25/0.50). `QA_TEXT_FLOORS`, `QA_KEEP_TOP_LIMITS`, and related functions in [qa.py](solution/core_modules/castle_mm26_core/qa.py) define per-task floors, keep limits, and additional policy weights. |
| Query-time model inference | [runtimes.py](solution/core_modules/castle_mm26_core/runtimes.py) defines inference defaults: TIPSv2 uses 448/FP32, Qwen3 query encoding uses 8192/BF16, BGE-M3 query encoding uses 8192/FP16, and Qwen3 reranking uses a maximum length of 8192, padding length of 1536, and BF16. `_execution_limits` in [pipeline.py](solution/core_modules/castle_mm26_core/pipeline.py) controls formal/smoke limits and batch sizes; formal text/visual reranking batch sizes are 4/8. `assign_model_devices` assigns devices according to the CLI `--devices` option. |

In particular, the interval-scoring fields `final_weights`, `object_track_weights`, and `scoring_component_profiles` in YAML do not determine the current final EIS/OIS score. Nor do `submit_quantile`, `min_submit_floor`, and `tail_audit_floor` determine the final submission threshold; `hard_floor` is used to decide whether to expand the reranking candidate cap. Some retained fields still participate in the configuration hash or behavior selected by policy name, so “does not affect the final score” does not mean they can be removed arbitrarily. OpenRouter requests do not set `temperature` or `max_tokens`; both use server defaults.

## Before You Start

We recommend Python 3.11, FFmpeg/FFprobe, and a CUDA-compatible PyTorch installation. Install dependencies according to each module's `requirements.txt` or `pyproject.toml`; the three feature encoders should use separate environments. Data, model weights, and runtime outputs are not included with the code. You must first prepare complete local snapshots of TIPSv2, Qwen3-Embedding-8B, BGE-M3, and Qwen3-Reranker-8B. QA additionally requires `OPENROUTER_API_KEY`.

Before running, update the machine-specific absolute paths retained in the downloader, TIPSv2, and solution configurations to match your system. The downloader's default proxy points to `127.0.0.1:7897`. If you do not use a proxy, explicitly set `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` to empty strings in the downloader YAML; simply removing the keys restores the code defaults.

Complete data preparation, all three encoding pipelines, and event-corpus construction in sequence, and check their validation reports. Then use [solution/fully_automatic_track/run.py](solution/fully_automatic_track/run.py) to run preflight checks, EIS, OIS, QA evidence export, QA answering, and merge/package. Use consistent configuration and run IDs across stages. After changing code, queries, models, or effective parameters, use a new run ID to regenerate the affected results.

The challenge rules allow participants to “receive queries in advance and generate results using any method they choose” and state that “participants receive queries in advance and produce their results by whatever process they see fit.” Because the challenge places few restrictions on the solution method, we optimized for particular tasks at the expense of generalizability. As a result, this system is not well suited to general-purpose questions.