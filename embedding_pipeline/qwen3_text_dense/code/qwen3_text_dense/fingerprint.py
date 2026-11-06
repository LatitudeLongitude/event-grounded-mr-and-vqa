from __future__ import annotations

from pathlib import Path
from typing import Dict

from .encoder import sha256_file


KEY_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
    "modules.json",
    "config_sentence_transformers.json",
)


def model_fingerprint(model_root: Path | str, *, hash_weights: bool = False) -> dict:
    root = Path(model_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Model root is missing: {root}")
    files: Dict[str, dict] = {}
    required_present = False
    for name in KEY_MODEL_FILES:
        path = root / name
        if not path.exists():
            files[name] = {"missing": True}
            continue
        if name in {"config.json", "tokenizer.json"}:
            required_present = True
        stat = path.stat()
        files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}
    if not required_present:
        raise FileNotFoundError("Required Qwen3 config/tokenizer files are missing")
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No safetensors weights found under {root}")
    for path in weights:
        stat = path.stat()
        item = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if hash_weights:
            item["sha256"] = sha256_file(path)
        files[path.name] = item
    return {"model_root": str(root), "files": files}
