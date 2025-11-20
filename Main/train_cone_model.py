"""
Train a YOLO model on the combined cylinder and cone dataset.

Default usage (expects pillar_cone.yaml in dataset_pillar):

    python train_cone_model.py

You can override defaults, for example:

    python train_cone_model.py --model runs/detect/exp/weights/best.pt --epochs 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def import_yolo():
    """Import YOLO lazily so the script can report a friendly message if missing."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        print(
            "Missing dependency: ultralytics\n"
            "Install it with:\n"
            "    pip install ultralytics\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    return YOLO


def build_parser(default_data: Path) -> argparse.ArgumentParser:
    """Construct the CLI argument parser populated with sensible defaults."""
    parser = argparse.ArgumentParser(description="Train YOLO on cylinder + cone dataset")
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data,
        help=f"Dataset YAML file (default: {default_data})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model or checkpoint to start from (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Training device, e.g. '0' for GPU0 or 'cpu' for CPU only",
    )
    return parser


def main() -> int:
    """Parse arguments, validate inputs, and launch an Ultralytics training run."""
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir / "dataset_pillar" / "pillar_cone.yaml"

    parser = build_parser(default_data)
    args = parser.parse_args()

    # Resolve the dataset YAML so we fail fast if the user points to a bad path.
    data_path = args.data.resolve()
    if not data_path.exists():
        print(f"Dataset YAML not found: {data_path}", file=sys.stderr)
        return 1

    YOLO = import_yolo()

    # Echo the resolved configuration for transparency before training starts.
    print(
        "Starting YOLO training:\n"
        f"  data  : {data_path}\n"
        f"  model : {args.model}\n"
        f"  epochs: {args.epochs}\n"
        f"  imgsz : {args.imgsz}\n"
        f"  batch : {args.batch}\n"
        f"  device: {args.device or 'auto'}"
    )

    model = YOLO(args.model)
    # Kick off Ultralytics training with the parsed hyperparameters.
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
    )

    # Report where the best-performing checkpoint landed for convenience.
    best_weight = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best weights saved at: {best_weight}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
