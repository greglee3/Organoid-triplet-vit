from __future__ import annotations

import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Sequence

from PIL import Image
from torch.utils.data import Dataset, Sampler


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class AsanoDataset(Dataset):
    """Read the Asano A/B/C images directly from a ZIP or an extracted folder."""

    def __init__(
        self,
        source: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.source = Path(source).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.class_names = ["A", "B", "C"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self._zip: zipfile.ZipFile | None = None
        self._zip_pid: int | None = None

        if self.source.is_file() and self.source.suffix.lower() == ".zip":
            self.storage = "zip"
            self.samples = self._scan_zip()
        elif self.source.is_dir():
            self.storage = "directory"
            self.data_root = self._resolve_data_root()
            self.samples = self._scan_directory()
        else:
            raise FileNotFoundError(f"Data source not found: {self.source}")

        if not self.samples:
            raise RuntimeError(
                f"No images found for split={split!r} under {self.source}"
            )
        self.targets = [label for _, label in self.samples]

    def _scan_zip(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        with zipfile.ZipFile(self.source) as archive:
            for raw_name in archive.namelist():
                path = PurePosixPath(raw_name)
                parts = path.parts
                if (
                    not parts
                    or parts[0] == "__MACOSX"
                    or path.suffix.lower() not in IMAGE_SUFFIXES
                ):
                    continue

                for index, part in enumerate(parts[:-1]):
                    if (
                        part == self.split
                        and index + 1 < len(parts)
                        and parts[index + 1] in self.class_to_idx
                    ):
                        label = self.class_to_idx[parts[index + 1]]
                        samples.append((raw_name, label))
                        break
        return sorted(samples, key=lambda item: item[0])

    def _resolve_data_root(self) -> Path:
        for candidate in (self.source, self.source / "data"):
            if (candidate / self.split).is_dir():
                return candidate
        raise FileNotFoundError(
            f"Expected <root>/{self.split}/A,B,C or "
            f"<root>/data/{self.split}/A,B,C under {self.source}"
        )

    def _scan_directory(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for class_name, label in self.class_to_idx.items():
            class_dir = self.data_root / self.split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for path in class_dir.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    samples.append((str(path), label))
        return sorted(samples, key=lambda item: item[0])

    def _get_zip(self) -> zipfile.ZipFile:
        process_id = os.getpid()
        if self._zip is None or self._zip_pid != process_id:
            if self._zip is not None:
                self._zip.close()
            self._zip = zipfile.ZipFile(self.source)
            self._zip_pid = process_id
        return self._zip

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_zip"] = None
        state["_zip_pid"] = None
        return state

    def __del__(self) -> None:
        archive = getattr(self, "_zip", None)
        if archive is not None:
            archive.close()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        payload, label = self.samples[index]
        if self.storage == "zip":
            archive = self._get_zip()
            with archive.open(payload) as stream:
                image = Image.open(stream).convert("RGB").copy()
        else:
            with Image.open(payload) as opened:
                image = opened.convert("RGB").copy()

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def stratified_train_val_indices(
    targets: Sequence[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(targets):
        grouped[int(label)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in grouped.values():
        rng.shuffle(indices)
        val_count = max(1, round(len(indices) * val_fraction))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


class PKBatchSampler(Sampler[list[int]]):
    """Create batches containing P classes and K examples per class."""

    def __init__(
        self,
        indices: Sequence[int],
        targets: Sequence[int],
        p: int = 3,
        k: int = 2,
        seed: int = 42,
    ) -> None:
        if k < 2:
            raise ValueError("k must be at least 2 so every anchor has a positive")

        self.by_class: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            self.by_class[int(targets[index])].append(int(index))
        self.classes = sorted(self.by_class)
        if p > len(self.classes):
            raise ValueError(f"p={p} exceeds the {len(self.classes)} available classes")

        self.p = p
        self.k = k
        self.seed = seed
        self.epoch = 0
        self.num_batches = max(1, len(indices) // (p * k))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.num_batches):
            selected_classes = rng.sample(self.classes, self.p)
            batch: list[int] = []
            for label in selected_classes:
                candidates = self.by_class[label]
                if len(candidates) >= self.k:
                    batch.extend(rng.sample(candidates, self.k))
                else:
                    batch.extend(rng.choices(candidates, k=self.k))
            rng.shuffle(batch)
            yield batch
