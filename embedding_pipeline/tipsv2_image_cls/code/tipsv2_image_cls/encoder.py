from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .schema import EMBEDDING_DIM, PARQUET_EMBEDDING_VALUE_TYPE


COMPUTE_PRECISION_POLICY = "fp32_dtype_tf32_allowed_no_autocast_no_fp16_no_bf16"


def _numpy_dtype_name(values: np.ndarray) -> str:
    dtype = np.asarray(values).dtype
    return f"np.{dtype.name}"


def configure_fp32_tf32(torch) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def _autocast_enabled(torch, device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:
        return bool(torch.is_autocast_enabled())


def precision_diagnostics(torch=None) -> Dict[str, object]:
    diagnostics: Dict[str, object] = {"compute_precision_policy": COMPUTE_PRECISION_POLICY}
    if torch is None:
        return diagnostics
    cuda_backend = getattr(torch.backends, "cuda", None)
    cuda_matmul = getattr(cuda_backend, "matmul", None)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    diagnostics["torch_float32_matmul_precision"] = (
        torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else "unavailable"
    )
    diagnostics["torch_backends_cuda_matmul_allow_tf32"] = bool(getattr(cuda_matmul, "allow_tf32", False))
    diagnostics["torch_backends_cudnn_allow_tf32"] = bool(getattr(cudnn_backend, "allow_tf32", False))
    diagnostics["torch_autocast_cuda_enabled"] = _autocast_enabled(torch, "cuda")
    diagnostics["torch_autocast_cpu_enabled"] = _autocast_enabled(torch, "cpu")
    return diagnostics


def extract_cls_token(output: Any):
    if not hasattr(output, "cls_token"):
        raise ValueError("TIPSv2 output does not expose cls_token")
    cls_token = output.cls_token
    if len(cls_token.shape) != 3 or cls_token.shape[1] < 1:
        raise ValueError(f"Unexpected cls_token shape: {cls_token.shape}")
    return cls_token[:, 0, :]


def extract_model_cls(model: Any, pixel_values: Any):
    """Use the official image-only API; outer forward wraps this output."""
    return extract_cls_token(model.encode_image(pixel_values))


class TIPSv2Encoder:
    def __init__(
        self,
        model_path: Path | str,
        *,
        device: str,
        precision: str = "fp32",
        embedding_dim: int = EMBEDDING_DIM,
        max_memory_fraction: float | None = None,
    ):
        if precision != "fp32":
            raise ValueError(
                "TIPSv2Encoder only supports fp32 tensor dtype with TF32 allowed; "
                "16-bit precision and autocast are forbidden"
            )
        self.model_path = Path(model_path).resolve()
        self.device = device
        self.precision = precision
        self.embedding_dim = embedding_dim
        self.max_memory_fraction = max_memory_fraction
        self.model_id = "google/tipsv2-g14"
        self.model_code_sha256 = sha256_file(self.model_path / "modeling_tips.py")
        self._torch = None
        self.model = None
        self.last_timings = {"h2d_seconds": 0.0, "forward_seconds": 0.0, "postprocess_seconds": 0.0}
        self.last_peak_memory_mib = 0.0

    def load(self) -> "TIPSv2Encoder":
        import torch
        from transformers import AutoModel

        self._torch = torch
        configure_fp32_tf32(torch)
        if self.max_memory_fraction is not None and str(self.device).startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(self.max_memory_fraction, device=self.device)
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        self.model.to(self.device)
        return self

    def _ensure_loaded(self):
        if self.model is None or self._torch is None:
            self.load()
        return self._torch

    def _require_fp32_tensor(self, values, label: str):
        torch = self._ensure_loaded()
        if not hasattr(values, "dtype"):
            raise ValueError(f"{label} does not expose a dtype")
        if values.dtype != torch.float32:
            raise ValueError(f"{label} must be torch.float32 before normalization, got {values.dtype}")
        return values

    def _normalize_torch_tensor(self, values, label: str):
        torch = self._ensure_loaded()
        tensor = self._require_fp32_tensor(values, label)
        if tensor.ndim != 2 or tensor.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embedding shape (N, {self.embedding_dim}), got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("Encoder output contains NaN or Inf")
        raw_norms = torch.linalg.vector_norm(tensor, dim=1)
        self._require_fp32_tensor(raw_norms, f"{label} raw norm")
        if torch.any(raw_norms == 0):
            raise ValueError("Encoder output contains a zero vector")
        normalized = tensor / raw_norms[:, None]
        self._require_fp32_tensor(normalized, f"normalized {label}")
        return normalized, raw_norms

    def _normalize_torch(self, values, label: str) -> Tuple[np.ndarray, np.ndarray]:
        normalized, raw_norms = self._normalize_torch_tensor(values, label)
        embedding_np = normalized.detach().cpu().numpy().astype(np.float32, copy=False)
        raw_norms_np = raw_norms.detach().cpu().numpy().astype(np.float32, copy=False)
        if embedding_np.dtype != np.float32:
            raise ValueError(f"Normalized {label} numpy dtype must be np.float32")
        return embedding_np, raw_norms_np

    def encode_images(self, pixel_values) -> Tuple[np.ndarray, np.ndarray]:
        import time

        torch = self._ensure_loaded()
        if str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        tensor = torch.as_tensor(pixel_values, dtype=torch.float32, device=self.device)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        h2d_done = time.perf_counter()
        with torch.inference_mode():
            cls = extract_model_cls(self.model, tensor)
        self._require_fp32_tensor(cls, "out.cls_token[:, 0, :]")
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        forward_done = time.perf_counter()
        result = self._normalize_torch(cls, "out.cls_token[:, 0, :]")
        finished = time.perf_counter()
        self.last_timings = {
            "h2d_seconds": h2d_done - started,
            "forward_seconds": forward_done - h2d_done,
            "postprocess_seconds": finished - forward_done,
        }
        if str(self.device).startswith("cuda"):
            self.last_peak_memory_mib = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        return result

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._ensure_loaded()
        with torch.inference_mode():
            values = self.model.encode_text(list(texts))
        self._require_fp32_tensor(values, "model.encode_text(...)")
        return self._normalize_torch(values, "model.encode_text(...)")[0]

    def smoke_check(self, pixel_values, text: str) -> Tuple[float, np.ndarray, Dict[str, object]]:
        torch = self._ensure_loaded()
        repeated = np.stack([pixel_values, pixel_values])
        tensor = torch.as_tensor(repeated, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            cls = extract_model_cls(self.model, tensor)
            self._require_fp32_tensor(cls, "out.cls_token[:, 0, :]")
            image_normalized, _ = self._normalize_torch_tensor(cls, "out.cls_token[:, 0, :]")
            text_raw = self.model.encode_text([text])
            self._require_fp32_tensor(text_raw, "model.encode_text(...)")
            text_normalized, _ = self._normalize_torch_tensor(text_raw, "model.encode_text(...)")
            repeat_similarity = image_normalized[0] @ image_normalized[1]
            similarity = image_normalized[:1] @ text_normalized.T
        self._require_fp32_tensor(repeat_similarity, "repeat image similarity")
        self._require_fp32_tensor(similarity, "image-text similarity")
        image_np = image_normalized.detach().cpu().numpy().astype(np.float32, copy=False)
        text_np = text_normalized.detach().cpu().numpy().astype(np.float32, copy=False)
        if image_np.dtype != np.float32 or text_np.dtype != np.float32:
            raise ValueError("Smoke normalized numpy outputs must be np.float32")
        evidence = {
            "cls_token_dtype": str(cls.dtype),
            "normalized_image_embedding_dtype": str(image_normalized.dtype),
            "raw_text_embedding_dtype": str(text_raw.dtype),
            "normalized_text_embedding_dtype": str(text_normalized.dtype),
            "similarity_dtype": str(similarity.dtype),
            "numpy_embedding_before_save_dtype": _numpy_dtype_name(image_np),
            "parquet_embedding_value_type": PARQUET_EMBEDDING_VALUE_TYPE,
            **precision_diagnostics(torch),
        }
        return float(repeat_similarity.detach().cpu().item()), text_np, evidence


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def resolved_versions() -> Dict[str, str]:
    packages = ("torch", "torchvision", "transformers", "pyarrow", "pandas", "Pillow", "PyYAML")
    result: Dict[str, str] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    try:
        import torch

        result["cuda"] = str(torch.version.cuda)
    except ImportError:
        result["cuda"] = "not-installed"
    return result
