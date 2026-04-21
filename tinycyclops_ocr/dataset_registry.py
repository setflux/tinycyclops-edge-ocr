from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Iterable

from .runtime import IMAGE_EXTENSIONS, PROJECT_ROOT, display_path


DEFAULT_CCPD_PRESET_SIZE = 1000
MAGIC_VALIDATION_MARKER = "# magic_validated=true"


@dataclass(frozen=True)
class DatasetPreset:
    key: str
    label: str
    title: str
    preset_path: Path
    run_prefix: str
    source_path: Path | None = None
    index_path: Path | None = None
    supports_randomize: bool = False
    default_sample_size: int | None = None
    attribution: str | None = None

    @property
    def public_preset_path(self) -> str:
        return display_path(self.preset_path)

    @property
    def public_source_path(self) -> str | None:
        return display_path(self.source_path) if self.source_path is not None else None

    @property
    def public_index_path(self) -> str | None:
        return display_path(self.index_path) if self.index_path is not None else None


DEFAULT_ICDAR_IMAGES = PROJECT_ROOT / "data" / "icdar2015" / "test_images"
DEFAULT_CCPD_SOURCE = PROJECT_ROOT / "data" / "ccpd" / "source"
DEFAULT_CCPD_IMAGES = PROJECT_ROOT / "data" / "ccpd" / "preset_images"
DEFAULT_CCPD_INDEX = PROJECT_ROOT / "data" / "ccpd" / "ccpd2019_index.txt"


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "icdar2015": DatasetPreset(
        key="icdar2015",
        label="Preset Workload 1",
        title="'ICDAR 2015' Scene Text",
        preset_path=DEFAULT_ICDAR_IMAGES,
        run_prefix="web_icdar2015",
        attribution=(
            "ICDAR (International Conference on Document Analysis and Recognition) "
            "2015 Robust Reading Competition Challenge 4 test images and ground "
            "truth are provided by the official RRC downloads page."
        ),
    ),
    "ccpd": DatasetPreset(
        key="ccpd",
        label="Preset Workload 2",
        title="'CCPD' Car License Plates",
        source_path=DEFAULT_CCPD_SOURCE,
        preset_path=DEFAULT_CCPD_IMAGES,
        index_path=DEFAULT_CCPD_INDEX,
        run_prefix="web_ccpd",
        supports_randomize=True,
        default_sample_size=DEFAULT_CCPD_PRESET_SIZE,
        attribution=(
            "CCPD2019 (Chinese City Parking Dataset 2019) is distributed through "
            "Zenodo under CC-BY-4.0."
        ),
    ),
}


def get_dataset_preset(key: str) -> DatasetPreset:
    try:
        return DATASET_PRESETS[key]
    except KeyError as exc:
        raise KeyError(f"Unsupported preset: {key}") from exc


def list_dataset_presets() -> tuple[DatasetPreset, ...]:
    return tuple(DATASET_PRESETS.values())


def iter_supported_images(path: Path, *, recursive: bool = False) -> Iterable[Path]:
    if not path.is_dir():
        return ()

    candidates = path.rglob("*") if recursive else path.iterdir()
    return (
        item
        for item in candidates
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_images(path: Path, *, recursive: bool = False) -> int:
    return sum(1 for _ in iter_supported_images(path, recursive=recursive))


def has_preset_images(preset: DatasetPreset) -> bool:
    return any(iter_supported_images(preset.preset_path))


def has_supported_image_magic(path: Path) -> bool:
    try:
        with path.open("rb") as fp:
            header = fp.read(16)
    except OSError:
        return False

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8")
    if suffix == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix == ".bmp":
        return header.startswith(b"BM")
    return False


def is_supported_image_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and has_supported_image_magic(path)
    )


def is_magic_validated_index(index_path: Path) -> bool:
    try:
        with index_path.open("r", encoding="utf-8") as fp:
            for _ in range(16):
                line = fp.readline()
                if not line:
                    return False
                if line.strip() == MAGIC_VALIDATION_MARKER:
                    return True
    except OSError:
        return False
    return False


def read_image_index(index_path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def ccpd_category(relative_path: Path) -> str:
    for part in relative_path.parts[:-1]:
        if part.startswith("ccpd_"):
            return part
    if len(relative_path.parts) > 1:
        return relative_path.parts[0]
    return "_root"


def allocate_balanced_quotas(groups: dict[str, list[Path]], limit: int) -> dict[str, int]:
    quotas = {category: 0 for category in groups}
    active = sorted(groups)
    remaining = limit

    while remaining > 0 and active:
        progressed = False
        for category in list(active):
            if quotas[category] < len(groups[category]):
                quotas[category] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            else:
                active.remove(category)
        if not progressed:
            break

    if remaining:
        raise ValueError(f"Need {limit} images, but only {limit - remaining} indexed images are available.")
    return quotas


def choose_indexed_images(
    image_rel_paths: list[Path],
    *,
    limit: int,
    strategy: str,
    rng: random.Random,
) -> tuple[list[Path], dict[str, int]]:
    if len(image_rel_paths) < limit:
        raise ValueError(f"Need at least {limit} images, found {len(image_rel_paths)} in the index.")

    if strategy == "first":
        selected = image_rel_paths[:limit]
        category_counts: defaultdict[str, int] = defaultdict(int)
        for path in selected:
            category_counts[ccpd_category(path)] += 1
        return selected, dict(sorted(category_counts.items()))

    if strategy == "random":
        selected = rng.sample(image_rel_paths, limit)
        category_counts = defaultdict(int)
        for path in selected:
            category_counts[ccpd_category(path)] += 1
        return selected, dict(sorted(category_counts.items()))

    if strategy != "balanced":
        raise ValueError(f"Unsupported selection strategy: {strategy}")

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in image_rel_paths:
        groups[ccpd_category(path)].append(path)

    quotas = allocate_balanced_quotas(groups, limit)
    selected: list[Path] = []
    for category in sorted(groups):
        quota = quotas[category]
        if quota:
            selected.extend(rng.sample(groups[category], quota))
    rng.shuffle(selected)
    return selected, {category: quotas[category] for category in sorted(quotas) if quotas[category]}


def safe_materialized_image_name(index: int, source: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source.stem).strip("._-")
    safe_stem = safe_stem[:72] or "ccpd"
    return f"ccpd_{index:04d}_{safe_stem}{source.suffix.lower()}"


def materialize_symlink_workload(
    *,
    source_path: Path,
    index_path: Path,
    target_path: Path,
    limit: int,
    strategy: str,
    seed: str,
) -> dict[str, object]:
    if not source_path.is_dir():
        raise ValueError(f"CCPD source directory is not prepared: {display_path(source_path)}")
    if not index_path.is_file():
        raise ValueError(f"CCPD image index is not prepared: {display_path(index_path)}")
    if not is_magic_validated_index(index_path):
        raise ValueError(f"CCPD image index needs to be rebuilt with image validation: {display_path(index_path)}")

    image_rel_paths = read_image_index(index_path)
    rng = random.Random(seed)
    selected_rel_paths, category_counts = choose_indexed_images(
        image_rel_paths,
        limit=limit,
        strategy=strategy,
        rng=rng,
    )

    target_path.mkdir(parents=True, exist_ok=False)
    for index, relative_path in enumerate(selected_rel_paths, start=1):
        image_path = source_path / relative_path
        if not is_supported_image_file(image_path):
            raise ValueError(f"Indexed CCPD image is missing or invalid: {display_path(image_path)}")
        destination = target_path / safe_materialized_image_name(index, image_path)
        destination.symlink_to(image_path)

    manifest = {
        "source_path": display_path(source_path),
        "index_path": display_path(index_path),
        "target_path": display_path(target_path),
        "image_count": len(selected_rel_paths),
        "strategy": strategy,
        "seed": seed,
        "categories": category_counts,
        "images": [path.as_posix() for path in selected_rel_paths],
    }
    return manifest


def preset_health(preset: DatasetPreset) -> dict[str, object]:
    image_count = count_images(preset.preset_path)
    return {
        "key": preset.key,
        "label": preset.label,
        "title": preset.title,
        "path": preset.public_preset_path,
        "available": preset.preset_path.is_dir(),
        "image_count": image_count,
        "supports_randomize": preset.supports_randomize,
        "default_sample_size": preset.default_sample_size,
        "source_path": preset.public_source_path,
        "source_available": preset.source_path.is_dir() if preset.source_path is not None else None,
        "index_path": preset.public_index_path,
        "index_available": preset.index_path.is_file() if preset.index_path is not None else None,
        "attribution": preset.attribution,
    }
