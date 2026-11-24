"""Generate synthetic software-test data, not Pokemon images or benchmark data.

AI-assisted public repository preparation, 2026-11-24.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data/sample.npz")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 16:
        parser.error("--samples must be at least 16")
    if args.output.exists():
        parser.error("Output already exists; choose another path to avoid overwriting data")
    rng = np.random.default_rng(args.seed)
    images = rng.integers(0, 32, size=(args.samples, 3, 128, 128), dtype=np.uint8)
    labels = np.arange(args.samples, dtype=np.int64) % 8
    for index, label in enumerate(labels):
        top, left = rng.integers(8, 80, size=2)
        color = np.array([220 if label & (1 << bit) else 70 for bit in range(3)], dtype=np.uint8)
        images[index, :, top:top + 32, left:left + 32] = color[:, None, None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, images=images, labels=labels)
    print(f"Created {args.samples} synthetic images for software checks: {args.output}")
    print("Labels are arbitrary fixture categories; they have no Pokemon type meaning.")


if __name__ == "__main__":
    main()
