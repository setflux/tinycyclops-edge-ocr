#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import random
import re
import secrets
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinycyclops_ocr.dataset_registry import get_dataset_preset  # noqa: E402
from tinycyclops_ocr.runtime import IMAGE_EXTENSIONS  # noqa: E402


CCPD_PRESET = get_dataset_preset("ccpd")
DEFAULT_SOURCE = CCPD_PRESET.source_path or PROJECT_ROOT / "data" / "ccpd" / "source"
DEFAULT_TARGET = CCPD_PRESET.preset_path
DEFAULT_INDEX = CCPD_PRESET.index_path or PROJECT_ROOT / "data" / "ccpd" / "ccpd2019_index.txt"
MAGIC_VALIDATION_MARKER = "# magic_validated=true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local CCPD preset directory for TinyCyclops."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Directory containing a licensed local CCPD image copy. Defaults to {DEFAULT_SOURCE}.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Preset output directory. Defaults to {DEFAULT_TARGET}.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help=f"Image index path. Defaults to {DEFAULT_INDEX}.",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Number of images to expose in the preset.")
    parser.add_argument(
        "--strategy",
        choices=("balanced", "random", "first"),
        default="balanced",
        help="Selection strategy. balanced samples evenly across CCPD category folders.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Random seed for reproducible random or balanced samples. Auto-generated when omitted.",
    )
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help="Rebuild the source image index even if it already exists.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks. Symlinks are the default to save disk space.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Remove the existing target directory before preparing the preset.",
    )
    return parser.parse_args()


def safe_name(index: int, source: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source.stem).strip("._-")
    safe_stem = safe_stem[:72] or "ccpd"
    return f"ccpd_{index:04d}_{safe_stem}{source.suffix.lower()}"


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


def discover_image_rel_paths(source: Path) -> list[Path]:
    return sorted(
        (
            path.relative_to(source)
            for path in source.rglob("*")
            if is_supported_image_file(path)
        ),
        key=lambda path: path.as_posix().lower(),
    )


def write_index(index_path: Path, source: Path, image_rel_paths: list[Path]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# TinyCyclops CCPD image index",
        f"# generated_at={generated_at}",
        f"# source={source}",
        MAGIC_VALIDATION_MARKER,
    ]
    lines.extend(path.as_posix() for path in image_rel_paths)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_index(index_path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


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


def load_or_create_index(source: Path, index_path: Path, *, refresh: bool) -> list[Path]:
    if refresh or not index_path.is_file() or not is_magic_validated_index(index_path):
        image_rel_paths = discover_image_rel_paths(source)
        write_index(index_path, source, image_rel_paths)
        return image_rel_paths
    return read_index(index_path)


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
        raise SystemExit(f"Need {limit} images, but only {limit - remaining} indexed images are available.")
    return quotas


def choose_images(
    image_rel_paths: list[Path],
    *,
    limit: int,
    strategy: str,
    rng: random.Random,
) -> tuple[list[Path], dict[str, int]]:
    if len(image_rel_paths) < limit:
        raise SystemExit(f"Need at least {limit} images, found {len(image_rel_paths)} in the index.")

    if strategy == "first":
        selected = image_rel_paths[:limit]
        category_counts: dict[str, int] = defaultdict(int)
        for path in selected:
            category_counts[ccpd_category(path)] += 1
        return selected, dict(sorted(category_counts.items()))

    if strategy == "random":
        selected = rng.sample(image_rel_paths, limit)
        category_counts = defaultdict(int)
        for path in selected:
            category_counts[ccpd_category(path)] += 1
        return selected, dict(sorted(category_counts.items()))

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


def prepare_target(target: Path, *, replace: bool) -> None:
    if target.exists() and replace:
        shutil.rmtree(target)
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"Target is not empty. Use --replace to rebuild it: {target}")
    target.mkdir(parents=True, exist_ok=True)


def materialize_preset(source: Path, target: Path, selected_rel_paths: list[Path], *, copy: bool) -> None:
    for index, relative_path in enumerate(selected_rel_paths, start=1):
        image = source / relative_path
        if not is_supported_image_file(image):
            raise SystemExit(f"Indexed image is missing or not decodable as an image: {image}")

        destination = target / safe_name(index, image)
        if copy:
            shutil.copy2(image, destination)
        else:
            destination.symlink_to(image)


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    target = args.target.expanduser().resolve()
    index_path = args.index.expanduser().resolve()

    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")
    if args.limit < 1:
        raise SystemExit("--limit must be greater than zero")

    image_rel_paths = load_or_create_index(source, index_path, refresh=args.refresh_index)
    seed = args.seed if args.seed is not None else secrets.token_hex(8)
    rng = random.Random(seed)
    selected_rel_paths, category_counts = choose_images(
        image_rel_paths,
        limit=args.limit,
        strategy=args.strategy,
        rng=rng,
    )

    prepare_target(target, replace=args.replace)
    materialize_preset(source, target, selected_rel_paths, copy=args.copy)

    print(f"prepared={len(selected_rel_paths)}")
    print(f"strategy={args.strategy}")
    print(f"seed={seed}")
    print(f"indexed={len(image_rel_paths)}")
    print(f"source={source}")
    print(f"target={target}")
    print(f"index={index_path}")
    print("mode=copy" if args.copy else "mode=symlink")
    print(
        "categories="
        + ",".join(f"{category}:{count}" for category, count in category_counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
