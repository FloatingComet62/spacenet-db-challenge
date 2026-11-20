#!/usr/bin/env python3
"""Interactive SpaceNet V7 demo.

Space: original -> prediction -> TP/FP/FN -> next random tile
Backspace: undo the last action, including across images (10 actions)
S: screenshot; Q/Escape: quit

Run from the project root:
    python demo_v7.py
"""
from __future__ import annotations

import argparse
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Navigation:
    """Three stages per tile, with a bounded *action* undo history."""
    index: int = 0
    stage: int = 0
    history: deque[tuple[int, int]] = field(
        default_factory=lambda: deque(maxlen=10)
    )

    def forward(self, tile_count: int) -> None:
        if tile_count < 1:
            raise ValueError("No demo tiles available")
        self.history.append((self.index, self.stage))
        if self.stage < 2:
            self.stage += 1
        else:
            self.stage = 0
            self.index = (self.index + 1) % tile_count

    def backward(self) -> bool:
        if not self.history:
            return False
        self.index, self.stage = self.history.pop()
        return True


@dataclass
class Tile:
    rgb: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray | None = None
    counts: dict | None = None


def tinted_overlay(
    rgb: np.ndarray,
    regions: list[tuple[np.ndarray, tuple[int, int, int]]],
    opacity: float = 0.65,
) -> np.ndarray:
    """Alpha-overlay binary regions; untouched pixels remain original RGB."""
    result = np.array(rgb, copy=True)
    for mask, color in regions:
        if mask.any():
            result[mask] = (
                (1 - opacity) * rgb[mask].astype(np.float32)
                + opacity * np.asarray(color, dtype=np.float32)
            ).astype(np.uint8)
    return result


def render_stage(tile: Tile, stage: int) -> np.ndarray:
    """0: RGB, 1: prediction (yellow), 2: TP/FP/FN (green/red/blue)."""
    if stage == 0:
        return tile.rgb

    if tile.prediction is None:
        raise RuntimeError("Prediction has not been computed")
    pred = tile.prediction.astype(bool)

    if stage == 1:
        return tinted_overlay(tile.rgb, [(pred, (255, 225, 0))])

    if stage == 2:
        truth = tile.truth.astype(bool)
        return tinted_overlay(
            tile.rgb,
            [
                (pred & truth, (0, 255, 0)),   # True positive
                (pred & ~truth, (255, 0, 0)),  # False positive
                (~pred & truth, (0, 80, 255)), # False negative
            ],
        )
    raise ValueError(f"Unknown stage: {stage}")


def cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Keyboard-driven live V7 building-segmentation demo"
    )
    p.add_argument(
        "--checkpoint", type=Path,
        default=Path("experiment_v7/runs/rgb_clahe/seed_111/best.pt"),
        help="V7 .pt checkpoint (defaults to your tested CLAHE seed 111)",
    )
    p.add_argument(
        "--manifest", type=Path,
        default=Path("experiment_v7/splits.json"),
    )
    p.add_argument(
        "--data", type=Path, default=Path("data/building_dataset"),
    )
    p.add_argument(
        "--cache", type=Path, default=Path("data/experiment_v7_cache"),
    )
    p.add_argument(
        "--split", choices=("train", "val", "test"), default="test",
        help="Draw randomly from this fixed dataset split",
    )
    p.add_argument(
        "--include-empty", action="store_true",
        help="Also include tiles containing no ground-truth building pixels",
    )
    p.add_argument(
        "--random-seed", type=int, default=None,
        help="Set to reproduce a particular shuffled image order",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Overlapping inference patch batch size; reduce if GPU OOM",
    )
    p.add_argument(
        "--max-undo", type=int, default=10,
        help="Maximum reversible Space presses (default 10)",
    )
    p.add_argument(
        "--save-dir", type=Path, default=Path("experiment_v7/demo_screenshots"),
    )
    return p.parse_args()


class Demo:
    def __init__(self, args: argparse.Namespace) -> None:
        import matplotlib.pyplot as plt
        import torch

        from experiment_v7.cache import check_prepared
        from experiment_v7.dataset import load_tile
        from experiment_v7.models import make_model
        from experiment_v7.splits import read_split

        self.args = args
        self.plt = plt
        self.torch = torch
        self.load_tile = load_tile
        self.navigation = Navigation(
            history=deque(maxlen=args.max_undo)
        )
        self.tiles: OrderedDict[str, Tile] = OrderedDict()
        self.cache_max_tiles = max(6, args.max_undo // 3 + 3)

        manifest = read_split(args.manifest, args.data)
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
        for split_name in ("train", "val", "test"):
            if checkpoint.get(f"{split_name}_files") != manifest[split_name]:
                raise RuntimeError(
                    f"Checkpoint and split manifest disagree on {split_name} tiles"
                )

        self.mode = checkpoint["mode"]
        self.threshold = checkpoint["config"]["threshold"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = make_model(self.mode, pretrained=False).to(self.device)
        self.model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        self.model.eval()

        names = list(manifest[args.split])
        stale = [
            name for name in names
            if not check_prepared(args.data / name, args.cache)
        ]
        if stale:
            raise RuntimeError(
                f"{len(stale)} cached tiles are missing or stale "
                f"(e.g. {stale[0]}). Run: python -m experiment_v7.cache"
            )
        if not args.include_empty:
            names = [
                name for name in names
                if np.asarray(
                    load_tile(args.cache, name, False)[1]
                ).any()
            ]
        if not names:
            raise ValueError(
                "No images matched your filters. Try --include-empty."
            )

        random.Random(args.random_seed).shuffle(names)
        self.names = names

        # Prevent Matplotlib's toolbar shortcut from consuming Backspace.
        plt.rcParams["keymap.back"] = []
        plt.rcParams["keymap.forward"] = []
        plt.rcParams["keymap.save"] = []
        plt.rcParams["keymap.quit"] = []

        self.fig, self.ax = plt.subplots(
            figsize=(15, 9), facecolor="#101720"
        )
        self.ax.set_facecolor("#101720")
        self.ax.axis("off")
        self.ax.set_position([0.04, 0.14, 0.92, 0.76])
        self.header = self.fig.suptitle(
            "", color="white", fontsize=16, y=0.96
        )
        self.information = self.fig.text(
            0.5, 0.088, "", ha="center", va="center",
            color="#d6dee8", fontsize=12,
        )
        self.footer = self.fig.text(
            0.5, 0.035,
            "SPACE  next stage    BACKSPACE  undo (10 actions)    "
            "S  screenshot    Q / ESC  quit",
            ha="center", va="center", color="#aeb9c8", fontsize=11,
        )
        self.image_artist = None
        self.message = ""

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.refresh()
        print(
            f"Loaded {len(self.names)} randomly ordered "
            f"{args.split} tiles | mode={self.mode} | device={self.device}"
        )
        print("SPACE: RGB -> prediction -> TP/FP/FN -> next image")
        print("BACKSPACE: undo up to 10 actions | S: save | Q: quit")

    def tile(self, name: str) -> Tile:
        if name in self.tiles:
            self.tiles.move_to_end(name)
            return self.tiles[name]

        rgb, mask, _aux = self.load_tile(
            self.args.cache, name, False
        )
        tile = Tile(
            rgb=np.asarray(rgb).copy(),
            truth=np.asarray(mask, dtype=bool).copy(),
        )
        self.tiles[name] = tile
        while len(self.tiles) > self.cache_max_tiles:
            self.tiles.popitem(last=False)
        return tile

    def ensure_prediction(self, name: str, tile: Tile) -> None:
        if tile.prediction is not None:
            return

        # Reuse the EXACT inference/reconstruction code from V7 evaluation.
        from experiment_v7.evaluation import predict_tile
        from experiment_v7.metrics import summarize_counts

        self.header.set_text(f"Running {self.mode} inference on {name} ...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        with self.torch.inference_mode():
            prediction, counts, _loss, _patches = predict_tile(
                self.model, self.args.cache, name, self.mode,
                self.device,
                batch_size=self.args.batch_size,
                threshold=self.threshold,
            )
        tile.prediction = prediction
        tile.counts = summarize_counts(counts)

    def refresh(self) -> None:
        nav = self.navigation
        name = self.names[nav.index]
        tile = self.tile(name)

        if nav.stage > 0:
            self.ensure_prediction(name, tile)
        visual = render_stage(tile, nav.stage)

        if self.image_artist is None:
            self.image_artist = self.ax.imshow(
                visual, interpolation="nearest"
            )
        else:
            self.image_artist.set_data(visual)

        stage_name = (
            "ORIGINAL SATELLITE IMAGE",
            "MODEL PREDICTION   |   Yellow = predicted building",
            "SEGMENTATION ERRORS   |   Green = TP   Red = FP   Blue = FN",
        )[nav.stage]
        self.header.set_text(
            f"{stage_name}\n"
            f"Image {nav.index + 1}/{len(self.names)}: {name}  "
            f"|  Model: {self.mode}"
        )

        if nav.stage == 0:
            details = (
                "Original RGB image. Press SPACE to show the model's prediction."
            )
        else:
            stats = tile.counts
            details = (
                f"IoU {stats['iou']:.3f}    "
                f"Dice {stats['dice']:.3f}    "
                f"Precision {stats['precision']:.3f}    "
                f"Recall {stats['recall']:.3f}"
            )
        if self.message:
            details += f"    |    {self.message}"
            self.message = ""
        self.information.set_text(details)
        self.footer.set_text(
            f"SPACE  advance    BACKSPACE  undo "
            f"({len(nav.history)}/{nav.history.maxlen})"
            f"    S  save screenshot    Q / ESC  quit"
        )
        self.fig.canvas.draw_idle()

    def on_key(self, event) -> None:
        key = event.key
        if key in (" ", "space"):
            self.navigation.forward(len(self.names))
            self.refresh()
        elif key in ("backspace", "left"):
            if not self.navigation.backward():
                self.message = "No earlier action in undo history"
            self.refresh()
        elif key in ("q", "escape"):
            self.plt.close(self.fig)
        elif key == "s":
            self.args.save_dir.mkdir(parents=True, exist_ok=True)
            name = Path(self.names[self.navigation.index]).stem
            phase = ("original", "prediction", "errors")[
                self.navigation.stage
            ]
            target = (
                self.args.save_dir
                / f"{name}_{phase}.png"
            )
            self.fig.savefig(
                target, dpi=160, facecolor=self.fig.get_facecolor()
            )
            self.message = f"Saved: {target}"
            print(self.message)
            self.refresh()

    def show(self) -> None:
        self.plt.show()


def main() -> None:
    args = cli_args()
    if args.max_undo < 1:
        raise SystemExit("--max-undo must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    try:
        Demo(args).show()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"V7 demo error: {exc}") from exc


if __name__ == "__main__":
    main()
