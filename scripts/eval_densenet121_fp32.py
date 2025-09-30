#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.datasets as dsets
import torchvision.models as models


def load_imagenet_index(path: Path | None) -> dict[str, int]:
    """Return mapping from WNID -> ImageNet index (0..999)."""
    if path and path.exists():
        mapping = json.loads(path.read_text())
    else:
        import urllib.request
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        mapping = json.load(urllib.request.urlopen(url))
    return {v[0]: int(k) for k, v in mapping.items()}


def main():
    ap = argparse.ArgumentParser(description="Evaluate FP32 DenseNet121 on a folder dataset")
    ap.add_argument("--images_dir", required=True, help="Dataset root with <WNID>/<image>.jpg structure")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Run on CPU or CUDA if available")
    ap.add_argument("--imagenet_index_json", type=Path, default=None,
                    help="Optional local copy of imagenet_class_index.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    dataset = dsets.ImageFolder(root=args.images_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=(device.type == "cuda"))

    # Map WNID -> true ImageNet index
    wnid_to_idx = load_imagenet_index(args.imagenet_index_json)
    class_to_imagenet_idx = {cls: wnid_to_idx[cls] for cls in dataset.classes}

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()

    top1 = top5 = total = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            # Convert ImageFolder targets (alphabetical WNID order) to ImageNet indices
            imagenet_targets = torch.tensor(
                [class_to_imagenet_idx[dataset.classes[t.item()]] for t in targets],
                device=device
            )
            logits = model(imgs)
            _, pred_top5 = logits.topk(5, dim=1)

            total += imagenet_targets.size(0)
            top1 += (pred_top5[:, 0] == imagenet_targets).sum().item()
            top5 += pred_top5.eq(imagenet_targets.unsqueeze(1)).sum().item()

    print(f"N={total}")
    print(f"Top-1: {top1 / total * 100:.2f}%")
    print(f"Top-5: {top5 / total * 100:.2f}%")


if __name__ == "__main__":
    main()