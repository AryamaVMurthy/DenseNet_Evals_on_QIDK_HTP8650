#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image


def load_imagenet_index(path: Path | None) -> Dict[int, Tuple[str, str]]:
    if path and path.exists():
        j = json.loads(path.read_text())
        return {int(k): (v[0], v[1]) for k, v in j.items()}
    # Try mirrors
    import urllib.request

    for url in (
        "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
        "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
    ):
        try:
            with urllib.request.urlopen(url) as r:
                j = json.load(r)
            return {int(k): (v[0], v[1]) for k, v in j.items()}
        except Exception:
            continue
    raise RuntimeError("Unable to load ImageNet class index mapping; pass --imagenet_index_json")


def main():
    ap = argparse.ArgumentParser(description="Export a subset of ImageNet-V2 to a folder structure (WNID subfolders)")
    ap.add_argument("--variant", choices=["matched-frequency", "threshold-0.7", "top-images"], default="matched-frequency")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--count", type=int, default=200, help="Number of images to export")
    ap.add_argument("--imagenet_index_json", type=Path, default=None)
    args = ap.parse_args()

    # Lazy import to avoid dependency unless requested
    try:
        from imagenetv2_pytorch import ImageNetV2Dataset
    except Exception as e:
        raise SystemExit(
            "Please install the loader first: pip install git+https://github.com/modestyachts/ImageNetV2_pytorch"
        )

    idx_map = load_imagenet_index(args.imagenet_index_json)

    ds = ImageNetV2Dataset(args.variant)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.count, len(ds)) if args.count > 0 else len(ds)
    exported = 0
    for i in range(n):
        img, target = ds[i]  # img: PIL.Image, target: 0..999
        wnid, name = idx_map.get(int(target), (str(int(target)), "?"))
        # Ensure RGB
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        out_sub = args.out_dir / wnid
        out_sub.mkdir(parents=True, exist_ok=True)
        out_path = out_sub / f"img_{i:05d}.jpg"
        img.save(out_path, format="JPEG", quality=95)
        exported += 1
    print(f"Exported {exported} images to {args.out_dir}")


if __name__ == "__main__":
    main()

