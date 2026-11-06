from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one file from Hugging Face with env prepared before import.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--mode", choices=["xet", "http"], default="xet")
    parser.add_argument("--download-timeout", type=int, default=300)
    parser.add_argument("--etag-timeout", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_home
    os.environ["HF_HUB_CACHE"] = args.cache_dir
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.download_timeout)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(args.etag_timeout)
    if args.mode == "http":
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    else:
        os.environ.pop("HF_HUB_DISABLE_XET", None)

    from huggingface_hub import hf_hub_download

    print(f"[WORKER] start mode={args.mode} filename={args.filename}", flush=True)
    local_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        filename=args.filename,
        local_dir=args.local_root,
        cache_dir=args.cache_dir,
        token=os.getenv(args.token_env),
    )
    path = Path(local_path)
    print(f"[WORKER] done path={path} size_bytes={path.stat().st_size if path.exists() else -1}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
