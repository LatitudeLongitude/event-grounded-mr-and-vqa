from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageDecodeError(RuntimeError):
    pass


def load_image_tensor(path: Path | str, *, image_size: int = 448) -> np.ndarray:
    target = Path(path)
    try:
        with Image.open(target) as image:
            rgb = image.convert("RGB")
            resized = rgb.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageDecodeError(f"Unable to decode image {target}: {exc}") from exc
    return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)


def load_image_batch(paths: Sequence[Path], *, image_size: int, decode_workers: int) -> np.ndarray:
    if decode_workers <= 1:
        images = [load_image_tensor(path, image_size=image_size) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=decode_workers, thread_name_prefix="jpeg-decode") as executor:
            images = list(executor.map(lambda path: load_image_tensor(path, image_size=image_size), paths))
    return np.stack(images).astype(np.float32, copy=False)


class ImageBatchLoader:
    def __init__(self, *, image_size: int, decode_workers: int):
        self.image_size = image_size
        self.decode_workers = decode_workers
        self._executor = (
            ThreadPoolExecutor(max_workers=decode_workers, thread_name_prefix="jpeg-decode")
            if decode_workers > 1
            else None
        )

    def load(self, paths: Sequence[Path]) -> np.ndarray:
        if self._executor is None:
            images = [load_image_tensor(path, image_size=self.image_size) for path in paths]
        else:
            images = list(
                self._executor.map(
                    lambda path: load_image_tensor(path, image_size=self.image_size),
                    paths,
                )
            )
        return np.stack(images).astype(np.float32, copy=False)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
