from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ImageWork:
    index: int
    image_path: Path
    image_id: int
    width: int
    height: int
    original_frame: Any | None = field(default=None, repr=False)

    def release_original(self) -> None:
        self.original_frame = None


@dataclass(frozen=True)
class WorkBlock:
    index: int
    image_works: tuple[ImageWork, ...]

    @classmethod
    def chunked(cls, image_works: Iterable[ImageWork], block_size: int) -> list["WorkBlock"]:
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        blocks = []
        batch = []
        for image_work in image_works:
            batch.append(image_work)
            if len(batch) == block_size:
                blocks.append(cls(index=len(blocks), image_works=tuple(batch)))
                batch = []

        if batch:
            blocks.append(cls(index=len(blocks), image_works=tuple(batch)))

        return blocks
