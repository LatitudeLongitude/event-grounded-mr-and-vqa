from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from .inventory import ChunkRecord
from .schema import SparseVector, parse_lexical_weights


def sha256_file(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_versions() -> dict:
    versions = {}
    for module_name in ("numpy", "pyarrow", "torch", "transformers", "FlagEmbedding"):
        try:
            if module_name == "FlagEmbedding":
                versions[module_name] = version(module_name)
                continue
            module = __import__(module_name)
            module_version = getattr(module, "__version__", None)
            if module_version:
                versions[module_name] = module_version
            else:
                versions[module_name] = version(module_name)
        except PackageNotFoundError:
            versions[module_name] = "unknown"
        except Exception as exc:
            versions[module_name] = f"unavailable: {type(exc).__name__}: {exc}"
    return versions


@dataclass(frozen=True)
class SparseEncodeResult:
    vectors: tuple[SparseVector, ...]
    truncated_ids: tuple[str, ...] = ()
    token_lengths: tuple[int, ...] = ()
    dtype_evidence: dict | None = None


class BGEM3SparseEncoder:
    def __init__(
        self,
        model_root: Path | str,
        *,
        devices: Sequence[int],
        model_id: str,
        max_length: int,
        use_fp16: bool,
        model_fingerprint: str,
    ):
        self.model_root = Path(model_root).resolve()
        self.devices = tuple(devices)
        self.model_id = model_id
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model_fingerprint = model_fingerprint
        self.model = None
        self.tokenizer = None
        self.dtype_evidence: dict = {}

    def _device_argument(self):
        cuda_devices = [f"cuda:{gpu_id}" for gpu_id in self.devices]
        if len(cuda_devices) == 1:
            return cuda_devices[0]
        return cuda_devices

    def load(self) -> "BGEM3SparseEncoder":
        from FlagEmbedding import BGEM3FlagModel

        kwargs = {
            "use_fp16": self.use_fp16,
            "devices": self._device_argument(),
        }
        self.model = BGEM3FlagModel(str(self.model_root), **kwargs)
        self.tokenizer = getattr(self.model, "tokenizer", None)
        self.dtype_evidence = {
            "encoder_backend": "bge-m3",
            "model_class": type(self.model).__name__,
            "document_instruction": "not-used",
            "return_dense": False,
            "return_sparse": True,
            "return_colbert_vecs": False,
            "use_fp16": self.use_fp16,
            "devices": list(self.devices),
            "output_type": "learned_sparse",
        }
        return self

    def encode_records(self, records: Sequence[ChunkRecord], *, batch_size: int | None = None) -> SparseEncodeResult:
        if self.model is None:
            self.load()
        assert self.model is not None
        texts = [record.text_for_embedding for record in records]
        ids = [record.embedding_chunk_id for record in records]
        output = self.model.encode(
            texts,
            batch_size=batch_size or len(texts),
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        if not isinstance(output, dict):
            raise ValueError("BGEM3FlagModel.encode must return a dict-like output")
        for forbidden in ("dense_vecs", "dense_embeddings", "colbert_vecs", "colbert_embeddings"):
            value = output.get(forbidden)
            if value is not None:
                raise ValueError(f"Forbidden BGE-M3 output was returned: {forbidden}")
        lexical_weights = output.get("lexical_weights") or output.get("sparse_vecs")
        if lexical_weights is None:
            raise ValueError("BGEM3FlagModel output did not include lexical_weights")
        if len(lexical_weights) != len(records):
            raise ValueError("BGEM3 lexical_weights count does not match input batch size")
        vectors = tuple(parse_lexical_weights(item, tokenizer=self.tokenizer) for item in lexical_weights)
        token_lengths = self._token_lengths(texts)
        truncated_ids = tuple(chunk_id for chunk_id, length in zip(ids, token_lengths) if length > self.max_length)
        key_formats = sorted(set(vector.key_format for vector in vectors))
        raw_dtypes = sorted(set(vector.raw_value_dtype for vector in vectors))
        evidence = {
            **self.dtype_evidence,
            "lexical_weights_count": len(vectors),
            "lexical_weight_key_formats": key_formats,
            "raw_sparse_value_dtypes": raw_dtypes,
            "sparse_index_dtype": "int32",
            "sparse_value_validation_dtype": "float32",
            "sparse_value_storage_dtype": "float16",
            "dense_saved": False,
            "colbert_saved": False,
        }
        return SparseEncodeResult(
            vectors=vectors,
            truncated_ids=truncated_ids,
            token_lengths=token_lengths,
            dtype_evidence=evidence,
        )

    def _token_lengths(self, texts: Sequence[str]) -> tuple[int, ...]:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return ()
        try:
            encoded = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True)
            return tuple(len(values) for values in encoded["input_ids"])
        except Exception:
            return ()
