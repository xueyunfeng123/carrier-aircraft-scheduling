"""Remove optimizer state from a trained checkpoint for inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rl.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    source = Path(args.checkpoint)
    target = Path(args.output) if args.output else source
    payload = load_checkpoint(str(source), device="cpu")
    payload.pop("optimizer_state", None)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    print(f"inference_checkpoint_written: {target}")


if __name__ == "__main__":
    main()
