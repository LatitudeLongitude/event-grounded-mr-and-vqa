from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .inventory import ChunkRecord
from .schema import EMBEDDING_DIM, validate_embedding_array


def sha256_file(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_versions() -> dict:
    versions = {}
    for module_name in ("numpy", "pyarrow", "torch", "transformers"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"unavailable: {type(exc).__name__}: {exc}"
    return versions


@dataclass(frozen=True)
class EncodeResult:
    embeddings: np.ndarray
    raw_norms: np.ndarray
    truncated_ids: tuple[str, ...] = ()
    token_lengths: tuple[int, ...] = ()
    dtype_evidence: dict | None = None


class Qwen3DenseEncoder:
    def __init__(
        self,
        model_root: Path | str,
        *,
        device: str,
        model_id: str,
        max_length: int,
        inference_dtype: str,
        model_fingerprint: str,
    ):
        self.model_root = Path(model_root).resolve()
        self.device = device
        self.model_id = model_id
        self.max_length = max_length
        self.inference_dtype = inference_dtype
        self.model_fingerprint = model_fingerprint
        self.tokenizer = None
        self.model = None
        self.dtype_evidence: dict = {}

    def load(self) -> "Qwen3DenseEncoder":
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_root,
            padding_side="left",
            local_files_only=True,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            self.model_root,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype_map[self.inference_dtype],
        )
        self.model.eval()
        self.model.to(self.device)
        self.dtype_evidence = {
            "encoder_backend": "qwen3",
            "document_instruction": "not-used",
            "padding_side": self.tokenizer.padding_side,
            "pooling": "last_token",
            "normalization": "torch.nn.functional.normalize_fp32",
            "inference_dtype": self.inference_dtype,
        }
        return self

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[range(batch_size), sequence_lengths]

    def encode_records(self, records: Sequence[ChunkRecord]) -> EncodeResult:
        if self.model is None or self.tokenizer is None:
            self.load()
        import torch
        import torch.nn.functional as F

        texts = [record.text_for_embedding for record in records]
        ids = [record.embedding_chunk_id for record in records]
        length_inputs = self.tokenizer(texts, padding=False, truncation=False, add_special_tokens=True)
        token_lengths = tuple(len(values) for values in length_inputs["input_ids"])
        truncated_ids = tuple(chunk_id for chunk_id, length in zip(ids, token_lengths) if length > self.max_length)
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
            pooled = self._last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            raw_norms = torch.linalg.vector_norm(pooled.float(), ord=2, dim=1)
            normalized = F.normalize(pooled.float(), p=2, dim=1)
        if normalized.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Qwen3 embeddings must have dimension {EMBEDDING_DIM}, got {normalized.shape[1]}")
        embeddings = normalized.detach().cpu().numpy().astype(np.float32)
        norms = raw_norms.detach().cpu().numpy().astype(np.float32)
        validate_embedding_array(embeddings, require_normalized=True)
        evidence = {
            **self.dtype_evidence,
            "last_hidden_state_dtype": str(outputs.last_hidden_state.dtype),
            "pooled_dtype": str(pooled.dtype),
            "normalized_dtype": str(normalized.dtype),
            "numpy_embedding_before_save_dtype": "np.float32",
            "output_dimension": int(normalized.shape[1]),
        }
        return EncodeResult(
            embeddings=embeddings,
            raw_norms=norms,
            truncated_ids=truncated_ids,
            token_lengths=token_lengths,
            dtype_evidence=evidence,
        )
