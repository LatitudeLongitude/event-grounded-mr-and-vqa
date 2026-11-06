from __future__ import annotations

import base64
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import time
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from .io_utils import sha256_file


def _from_pretrained_without_meta(factory, model_root: Path | str, **kwargs):
    """Load with real tensors so a later .to(device) cannot hit meta tensors."""
    try:
        return factory.from_pretrained(str(model_root), low_cpu_mem_usage=False, **kwargs)
    except TypeError as exc:
        if "low_cpu_mem_usage" not in str(exc):
            raise
        return factory.from_pretrained(str(model_root), **kwargs)


def release_runtime_state(runtime: object) -> None:
    torch_module = getattr(runtime, "torch", None)
    for attr in ("model", "tokenizer"):
        if hasattr(runtime, attr):
            setattr(runtime, attr, None)
    for attr in ("yes_token_id", "no_token_id"):
        if hasattr(runtime, attr):
            setattr(runtime, attr, None)
    if hasattr(runtime, "torch"):
        setattr(runtime, "torch", None)
    gc.collect()
    if torch_module is None:
        return
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
            if hasattr(torch_module.cuda, "ipc_collect"):
                torch_module.cuda.ipc_collect()
    except Exception:
        return


def configure_offline_hf_cache(cache_dir: Path) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "HF_HOME": cache_dir / "huggingface",
        "HF_HUB_CACHE": cache_dir / "huggingface" / "hub",
        "HF_MODULES_CACHE": cache_dir / "huggingface" / "modules",
        "TORCH_HOME": cache_dir / "torch",
        "PIP_CACHE_DIR": cache_dir / "pip",
        "TMPDIR": cache_dir / "tmp",
        "TEMP": cache_dir / "tmp",
        "TMP": cache_dir / "tmp",
    }
    evidence = {}
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
        evidence[name] = str(path)
    flags = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "USE_TF": "0",
        "TRANSFORMERS_NO_TF": "1",
        "USE_FLAX": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, value in flags.items():
        os.environ[key] = value
    evidence.update(flags)
    return evidence


def configure_torch_precision(torch) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(values).all():
        raise ValueError("cannot normalize non-finite or zero vector")
    return (values / norms).astype(np.float32, copy=False)


def load_image_tensor(path: Path | str, *, image_size: int = 448) -> np.ndarray:
    target = Path(path)
    try:
        with Image.open(target) as image:
            rgb = image.convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            resized = rgb.resize((image_size, image_size), resample=resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"unable to decode image {target}: {exc}") from exc
    return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)


class TipsV2Runtime:
    def __init__(self, model_root: Path | str, *, device: str, cache_dir: Path):
        self.model_root = Path(model_root)
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.torch = None
        self.environment: dict[str, str] = {}

    def load(self) -> "TipsV2Runtime":
        self.environment = configure_offline_hf_cache(self.cache_dir)
        import torch
        from transformers import AutoModel

        configure_torch_precision(torch)
        self.torch = torch
        self._set_current_cuda_device()
        self.model = _from_pretrained_without_meta(AutoModel, self.model_root, trust_remote_code=True, local_files_only=True)
        self.model.eval().to(self.device)
        self._assert_no_meta_tensors()
        return self

    def _ensure(self):
        if self.model is None:
            self.load()
        return self.torch, self.model

    def ensure_loaded(self) -> "TipsV2Runtime":
        self._ensure()
        self._assert_no_meta_tensors()
        return self

    def _set_current_cuda_device(self) -> None:
        torch = self.torch
        if torch is None or not str(self.device).startswith("cuda"):
            return
        try:
            torch.cuda.set_device(self.device)
        except Exception:
            return

    def _assert_no_meta_tensors(self) -> None:
        model = self.model
        if model is None:
            return
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if getattr(tensor, "is_meta", False):
                raise RuntimeError(f"TIPSv2 model on {self.device} has meta tensor: {name}")

    def encode_text(self, prompts: Sequence[str]) -> np.ndarray:
        torch, model = self._ensure()
        self._set_current_cuda_device()
        with torch.inference_mode():
            raw = model.encode_text(list(prompts))
        if raw.dtype != torch.float32:
            raw = raw.float()
        values = raw.detach().cpu().numpy().astype(np.float32)
        if values.ndim != 2:
            raise ValueError(f"TIPSv2 text embeddings must be 2D, got {values.shape}")
        return l2_normalize(values)

    def encode_image_with_patches(self, image_paths: Sequence[Path | str]) -> dict[str, np.ndarray]:
        torch, model = self._ensure()
        self._set_current_cuda_device()
        batch = np.stack([load_image_tensor(path) for path in image_paths]).astype(np.float32)
        tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            output = model.encode_image(tensor)
        if not hasattr(output, "patch_tokens"):
            raise ValueError("TIPSv2 encode_image output does not expose native patch_tokens")
        if not hasattr(output, "cls_token"):
            raise ValueError("TIPSv2 encode_image output does not expose cls_token")
        cls = output.cls_token[:, 0, :].float()
        patches = output.patch_tokens.float()
        cls_np = l2_normalize(cls.detach().cpu().numpy().astype(np.float32))
        patch_np = patches.detach().cpu().numpy().astype(np.float32)
        if patch_np.ndim != 3:
            raise ValueError(f"patch tokens must be 3D, got {patch_np.shape}")
        patch_np = l2_normalize(patch_np, axis=-1)
        return {"cls": cls_np, "patch_tokens": patch_np}

    def fingerprint(self) -> dict[str, Any]:
        files = ["config.json", "model.safetensors", "modeling_tips.py", "text_encoder.py", "tokenizer.model"]
        records = []
        for name in files:
            path = self.model_root / name
            if path.exists() and path.is_file():
                records.append({"file": name, "sha256": sha256_file(path), "size": path.stat().st_size})
        return {
            "model_id": "google/tipsv2-g14",
            "model_root": self.model_root.as_posix(),
            "files": records,
            "fingerprint_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            "preprocess_id": "tipsv2_rgb_resize448_bilinear_antialias_v1",
        }

    def preflight(self, image_path: Path, prompt: str) -> dict[str, Any]:
        text = self.encode_text([prompt])
        encoded = self.encode_image_with_patches([image_path])
        cls = encoded["cls"]
        patches = encoded["patch_tokens"]
        if text.shape[-1] != cls.shape[-1] or text.shape[-1] != patches.shape[-1]:
            raise ValueError(f"TIPSv2 text/cls/patch dimensions differ: {text.shape}, {cls.shape}, {patches.shape}")
        if text.shape[-1] != 1536:
            raise ValueError(f"TIPSv2 embedding dimension must be 1536, got {text.shape[-1]}")
        return {
            "status": "completed",
            "device": self.device,
            "text_shape": list(text.shape),
            "cls_shape": list(cls.shape),
            "patch_tokens_shape": list(patches.shape),
            "dtype": "float32",
            "image_path": image_path.as_posix(),
            "image_sha256": sha256_file(image_path),
            "fingerprint": self.fingerprint(),
        }


class Qwen3DenseRuntime:
    def __init__(self, model_root: Path | str, *, device: str, cache_dir: Path, max_length: int = 8192, dtype: str = "bfloat16"):
        self.model_root = Path(model_root)
        self.device = device
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.torch = None

    def load(self) -> "Qwen3DenseRuntime":
        configure_offline_hf_cache(self.cache_dir)
        import torch
        import torch.nn.functional as F  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_root), padding_side="left", trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = _from_pretrained_without_meta(AutoModel, self.model_root, trust_remote_code=True, local_files_only=True, dtype=dtype_map[self.dtype])
        self.model.eval().to(self.device)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            self.load()
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        import torch.nn.functional as F

        batch = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with self.torch.inference_mode():
            outputs = self.model(**batch)
            pooled = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            normalized = F.normalize(pooled.float(), p=2, dim=1)
        values = normalized.detach().cpu().numpy().astype(np.float32)
        if values.shape[1] != 4096:
            raise ValueError(f"Qwen3 dense dimension must be 4096, got {values.shape[1]}")
        return values


def _last_token_pool(last_hidden_states, attention_mask):
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[range(batch_size), sequence_lengths]


@dataclass(frozen=True)
class SparseVector:
    indices: np.ndarray
    values: np.ndarray


class BGEM3SparseRuntime:
    def __init__(self, model_root: Path | str, *, devices: Sequence[str], cache_dir: Path, max_length: int = 8192, use_fp16: bool = True):
        self.model_root = Path(model_root)
        self.devices = tuple(devices)
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model = None

    def load(self) -> "BGEM3SparseRuntime":
        configure_offline_hf_cache(self.cache_dir)
        from FlagEmbedding import BGEM3FlagModel

        device_arg = list(self.devices) if len(self.devices) > 1 else self.devices[0]
        runtime_root = prepare_bgem3_runtime_root(self.model_root, self.cache_dir / "model_mirror")
        self.model = BGEM3FlagModel(str(runtime_root), use_fp16=self.use_fp16, devices=device_arg)
        return self

    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> list[SparseVector]:
        if self.model is None:
            self.load()
        output = self.model.encode(list(texts), batch_size=batch_size, max_length=self.max_length, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        lexical = output.get("lexical_weights") or output.get("sparse_vecs")
        if lexical is None:
            raise ValueError("BGE-M3 output does not contain lexical sparse vectors")
        return [parse_sparse_mapping(item, tokenizer=getattr(self.model, "tokenizer", None)) for item in lexical]


def parse_sparse_mapping(mapping: Mapping[Any, Any], *, tokenizer=None) -> SparseVector:
    parsed: dict[int, float] = {}
    for key, value in mapping.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            if tokenizer is None:
                raise
            index = int(tokenizer.convert_tokens_to_ids(str(key)))
        score = float(np.asarray(value).astype(np.float32))
        if index >= 0:
            parsed[index] = max(parsed.get(index, 0.0), score)
    indices = np.asarray(sorted(parsed), dtype=np.int32)
    values = np.asarray([parsed[int(index)] for index in indices], dtype=np.float32)
    validate_sparse_vector(indices, values)
    return SparseVector(indices=indices, values=values)


def decode_sparse_binary(indices_bytes: bytes, values_bytes: bytes) -> SparseVector:
    indices = np.frombuffer(indices_bytes, dtype="<i4").astype(np.int32)
    values = np.frombuffer(values_bytes, dtype="<f2").astype(np.float32)
    validate_sparse_vector(indices, values)
    return SparseVector(indices=indices, values=values)


def validate_sparse_vector(indices: np.ndarray, values: np.ndarray) -> None:
    if indices.ndim != 1 or values.ndim != 1 or len(indices) != len(values):
        raise ValueError("sparse indices and values must be same-length 1D arrays")
    if len(indices) == 0:
        raise ValueError("sparse vector must be non-empty")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("sparse indices must be unique")
    if np.any(indices < 0) or not np.isfinite(values).all() or np.all(values == 0):
        raise ValueError("invalid sparse vector values")


def prepare_bgem3_runtime_root(model_root: Path, mirror_root: Path) -> Path:
    if (model_root / "model.safetensors").exists() or (model_root / "model.safetensors.index.json").exists():
        return model_root
    bin_path = model_root / "pytorch_model.bin"
    if not bin_path.exists():
        return model_root
    mirror_root.mkdir(parents=True, exist_ok=True)
    for source in model_root.iterdir():
        if not source.is_file() or source.name == "pytorch_model.bin":
            continue
        target = mirror_root / source.name
        if target.exists():
            continue
        try:
            os.symlink(source, target)
        except OSError:
            shutil.copy2(source, target)
    safe_path = mirror_root / "model.safetensors"
    if not safe_path.exists():
        import torch
        from safetensors.torch import save_file

        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        tmp_path = safe_path.with_name(f"{safe_path.name}.tmp.{os.getpid()}")
        save_file(state, tmp_path, metadata={"format": "pt", "source": bin_path.as_posix()})
        os.replace(tmp_path, safe_path)
    return mirror_root


class Qwen3RerankerRuntime:
    def __init__(self, model_root: Path | str, *, device: str, cache_dir: Path, max_length: int = 8192, padding_length: int = 1536, dtype: str = "bfloat16", attention: str = "sdpa", seed: int = 20260626):
        self.model_root = Path(model_root)
        self.device = device
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.padding_length = padding_length
        self.dtype = dtype
        self.attention = attention
        self.seed = seed
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.yes_token_id = None
        self.no_token_id = None

    def load(self) -> "Qwen3RerankerRuntime":
        configure_offline_hf_cache(self.cache_dir)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_root), padding_side="left", local_files_only=True)
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        if not isinstance(self.no_token_id, int) or not isinstance(self.yes_token_id, int) or self.no_token_id == self.yes_token_id:
            raise ValueError("Qwen3 reranker tokenizer must provide distinct yes/no token ids")
        self.model = _from_pretrained_without_meta(AutoModelForCausalLM, self.model_root, dtype=dtype_map[self.dtype], attn_implementation=self.attention, local_files_only=True)
        self.model.eval().to(self.device)
        return self

    def score(self, pairs: Sequence[tuple[str, str, str]], *, instruction: str, batch_size: int = 4) -> list[dict[str, float | int]]:
        if self.model is None:
            self.load()
        assert self.model is not None and self.tokenizer is not None and self.torch is not None
        results = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                prompts = [format_reranker_prompt(instruction, query, document) for _, query, document in batch]
                encoded = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
                token_counts = [len(ids) for ids in encoded["input_ids"]]
                if any(count > self.max_length for count in token_counts):
                    raise ValueError(f"Qwen3 reranker token length exceeds max_length={self.max_length}: {token_counts}")
                encoded = self.tokenizer(prompts, padding="max_length", max_length=max(self.padding_length, max(token_counts)), truncation=False, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**inputs).logits[:, -1, :]
                no_logits = logits[:, self.no_token_id].float()
                yes_logits = logits[:, self.yes_token_id].float()
                margins = yes_logits - no_logits
                scores = self.torch.sigmoid(margins)
                for score, margin, token_count in zip(scores.detach().cpu().tolist(), margins.detach().cpu().tolist(), token_counts):
                    results.append({"reranker_score": float(score), "reranker_raw_score": float(margin), "token_count": int(token_count)})
        return results

    def evidence(self) -> dict[str, Any]:
        return {
            "model_id": "Qwen/Qwen3-Reranker-8B",
            "model_root": self.model_root.as_posix(),
            "device": self.device,
            "torch_dtype": self.dtype,
            "attention_implementation": self.attention,
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
        }


def format_reranker_prompt(instruction: str, query: str, document: str) -> str:
    return (
        "<|im_start|>system\n"
        "Judge whether the document answers the query. Answer yes or no.<|im_end|>\n"
        "<|im_start|>user\n"
        f"instruction:\n{instruction}\n\nquery:\n{query}\n\ndocument:\n{document}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class OpenRouterClient:
    def __init__(self, *, endpoint: str, model: str, api_key_env: str, referer_env: str, title: str, timeout_sec: int = 90, max_retries: int = 2):
        self.endpoint = endpoint
        self.model = model
        self.api_key_env = api_key_env
        self.referer_env = referer_env
        self.title = title
        self.timeout_sec = timeout_sec
        self.max_retries = max(0, int(max_retries))

    def api_key_present(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def model_metadata(self) -> dict[str, Any]:
        import requests

        response = self._request_with_retries(requests.get, "https://openrouter.ai/api/v1/models")
        payload = response.json()
        for item in payload.get("data", []):
            if item.get("id") == self.model:
                return item
        raise ValueError(f"OpenRouter model not found: {self.model}")

    def answer(
        self,
        *,
        question: str,
        evidence: Sequence[dict[str, Any]],
        schema: dict[str, Any],
        answer_policy: Mapping[str, Any] | None = None,
        max_images: int = 2,
    ) -> dict[str, Any]:
        import requests

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        prompt_evidence, image_paths = _prepare_prompt_evidence(evidence, max_images=max_images)
        prompt_text = _format_answer_prompt(
            question=question,
            evidence=prompt_evidence,
            answer_policy=answer_policy or {},
            schema=schema,
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt_text,
            }
        ]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "response_format": schema,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get(self.referer_env, "https://castle-dataset.github.io/challenges/mm26/"),
            "X-OpenRouter-Title": self.title,
        }
        response = self._request_with_retries(requests.post, self.endpoint, headers=headers, json=body)
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
        return json.loads(text)

    def _request_with_retries(self, method, url: str, **kwargs):
        import requests

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = method(url, timeout=self.timeout_sec, **kwargs)
                retryable = response.status_code in {408, 409, 425, 429} or 500 <= response.status_code <= 599
                if retryable:
                    response.raise_for_status()
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                response = getattr(exc, "response", None)
                if response is not None:
                    retryable = response.status_code in {408, 409, 425, 429} or 500 <= response.status_code <= 599
                    if not retryable:
                        break
                time.sleep(min(2 ** attempt, 8))
        assert last_exc is not None
        raise last_exc


def _prepare_prompt_evidence(evidence: Sequence[dict[str, Any]], *, max_images: int) -> tuple[list[dict[str, Any]], list[Path]]:
    prompt_evidence: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    image_count = 0
    prompt_keys = {
        "task_id",
        "evidence_id",
        "day_num",
        "source",
        "source_role",
        "start_second",
        "end_second",
        "text_atom_ids",
        "event_ids",
        "frame_atom_ids",
        "text_excerpt",
        "entity_mentions",
        "relation_candidates",
        "time_filter_fit",
        "reranker_score",
        "visual_probe_score",
        "qa_evidence_score",
    }
    for item in evidence:
        sanitized = {key: item[key] for key in prompt_keys if key in item}
        labels: list[str] = []
        for image_path in item.get("original_image_refs") or []:
            if image_count >= max_images:
                break
            path = Path(str(image_path))
            if path.exists() and path.is_file():
                image_count += 1
                labels.append(f"image_{image_count}")
                image_paths.append(path)
        if labels:
            sanitized["attached_image_refs"] = labels
        prompt_evidence.append(sanitized)
    return prompt_evidence, image_paths


def prompt_image_attachment_summary(evidence: Sequence[dict[str, Any]], *, max_images: int) -> dict[str, Any]:
    prompt_evidence, image_paths = _prepare_prompt_evidence(evidence, max_images=max_images)
    attached_refs: list[str] = []
    evidence_with_images = 0
    original_image_ref_count = 0
    existing_image_ref_count = 0
    missing_image_ref_count = 0
    for item in evidence:
        for image_path in item.get("original_image_refs") or []:
            original_image_ref_count += 1
            path = Path(str(image_path))
            if path.exists() and path.is_file():
                existing_image_ref_count += 1
            else:
                missing_image_ref_count += 1
    for item in prompt_evidence:
        labels = list(item.get("attached_image_refs") or [])
        if labels:
            evidence_with_images += 1
            attached_refs.extend(str(label) for label in labels)
    return {
        "max_prompt_images": int(max_images),
        "prompt_evidence_count": len(prompt_evidence),
        "original_image_ref_count": original_image_ref_count,
        "existing_image_ref_count": existing_image_ref_count,
        "missing_image_ref_count": missing_image_ref_count,
        "attached_image_count": len(image_paths),
        "evidence_with_attached_images": evidence_with_images,
        "attached_image_refs": attached_refs,
    }


def _format_answer_prompt(
    *,
    question: str,
    evidence: Sequence[dict[str, Any]],
    answer_policy: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> str:
    policy = _prepare_prompt_answer_policy(answer_policy)
    output_schema = _prepare_prompt_schema(schema)
    ordering_note = _ordering_prompt_note(policy)
    return (
        "Answer using only the candidate Evidence JSON and attached images. "
        "Return strict JSON matching the Output Schema JSON. "
        "Do not invent source, time, names, objects, counts, or relations. "
        "The JSON answer field must contain only the final answer value; put explanations only in rationale_short. "
        "Attached images follow attached_image_refs order.\n"
        "Use the Answer Policy JSON as deterministic reducer constraints: obey expected_answer_type; "
        "for count_unit/dedupe_policy count only supported unique units; "
        "for relation_direction require subject -> object direction; "
        "for ordering and extraction use the provided policy fields and candidate evidence only.\n"
        "The supporting_evidence_ids field must contain distinct ids from Evidence JSON. "
        "Include at least min_supporting_evidence ids. If your confidence is below 0.5, "
        "include at least low_confidence_min_supporting_evidence ids when that policy value is present; "
        "do not lower confidence below 0.5 when only one evidence item supports the answer.\n"
        f"{ordering_note}"
        f"Official Question: {question}\n"
        f"Answer Policy JSON:\n{json.dumps(policy, ensure_ascii=False, sort_keys=True)}\n"
        f"Evidence JSON:\n{json.dumps(list(evidence), ensure_ascii=False)}\n"
        f"Output Schema JSON:\n{json.dumps(output_schema, ensure_ascii=False, sort_keys=True)}"
    )


def _ordering_prompt_note(policy: Mapping[str, Any]) -> str:
    ordering = policy.get("ordering") if isinstance(policy, MappingABC) else None
    if not isinstance(ordering, MappingABC) or not ordering:
        return ""
    sort_by = str(ordering.get("sort_by") or "end_second")
    direction = str(ordering.get("direction") or "ascending").lower()
    extreme = "latest/highest" if direction == "descending" else "earliest/lowest"
    return (
        "Ordering rule: compare evidence timestamps as (day_num, "
        f"{sort_by}), not by {sort_by} alone. "
        f"For {direction} ordering, choose the {extreme} tuple after this cross-day comparison. "
        "For person answers with require_actor, the answer must be a named actor/person supported by that ordered evidence, "
        "and supporting_evidence_ids should include the ordered evidence item when available. "
        "If keyword_frames.positive is present, first discard evidence that does not locally connect a named person to those positive terms; "
        "do not infer the actor from source/camera name alone.\n"
    )


def _prepare_prompt_answer_policy(answer_policy: Mapping[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "reducer",
        "expected_answer_type",
        "min_supporting_evidence",
        "low_confidence_min_supporting_evidence",
        "keyword_frames",
        "time_filters",
        "count_unit",
        "dedupe_policy",
        "relation_direction",
        "ordering",
        "extraction",
        "actor_resolution",
        "activity_phrase_policy",
        "state_subject_policy",
        "role_policy",
        "visual_probe_terms",
        "entity_aliases",
        "multilingual_aliases",
        "answer_template",
    }
    return {key: _json_safe(value) for key, value in answer_policy.items() if key in allowed_keys}


def _prepare_prompt_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    json_schema = schema.get("json_schema")
    if isinstance(json_schema, MappingABC) and isinstance(json_schema.get("schema"), MappingABC):
        return dict(json_schema["schema"])
    return dict(schema)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, MappingABC):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    return str(value)


def image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def package_versions() -> dict[str, str]:
    result = {}
    for name in ("numpy", "pyarrow", "Pillow", "torch", "torchvision", "transformers", "FlagEmbedding", "safetensors", "requests"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result
