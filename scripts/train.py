"""
train.py
========

Parametrized training script for JRAA-DPS.

Reads experiment configuration from a YAML file (model, data, training params)
and runs a full training pipeline with checkpointing, logging, and resuming.

Usage:
    python train.py \
        --data configs/data.yml \
        --model configs/model.yml \
        --train configs/train.yml
"""

import os
import sys
import glob
import argparse
import warnings

import torch

warnings.filterwarnings("ignore", category=UserWarning, module="torchio")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.train_infer import set_seed, load_yaml, merge_configs, create_weights_logs_dirs, build_model, build_dataloader
from src.trainers.wavelet_diffusion_trainer import WaveletDiffusionTrainer


def main(args) -> None:
    # ---------------- LOAD CONFIGS ---------------- #
    data_cfg = load_yaml(args.data)
    model_cfg = load_yaml(args.model)
    train_cfg = load_yaml(args.train)
    cfg = merge_configs(data_cfg, model_cfg, train_cfg)

    set_seed(cfg.get("seed", 123))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    # ---------------- DATA ---------------- #
    train_dataloader = build_dataloader(cfg["data"], split="train")
    valid_dataloader = build_dataloader(cfg["data"], split="valid")

    # ---------------- MODEL ---------------- #
    dirs = create_weights_logs_dirs(cfg["train"]["save_root"], [args.data, args.model, args.train])
    model = build_model(cfg["model"], device)

    # Resume from last checkpoint
    start_epoch = 0
    last_ckpt = sorted(glob.glob(os.path.join(dirs["weights"], "*.pth")))
    if last_ckpt:
        last_path = last_ckpt[-1]
        start_epoch = int(os.path.splitext(os.path.basename(last_path))[0].split("_")[-1])
        print(f"Resuming from epoch {start_epoch}")
        model.load_state_dict(torch.load(last_path, map_location=device))

    # ---------------- TRAINING ---------------- #
    opt_cfg = cfg["optimizer"]
    optimizer = getattr(torch.optim, opt_cfg["type"])(model.parameters(), **opt_cfg["params"])
    scheduler = None
    if "scheduler" in cfg:
        sch_cfg = cfg["scheduler"]
        scheduler = getattr(torch.optim.lr_scheduler, sch_cfg["type"])(optimizer, **sch_cfg["params"])
    loss_fn = getattr(torch.nn, cfg["train"]["loss"])()

    trainer = WaveletDiffusionTrainer(
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader,
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg["train"]["epochs"],
        start_epoch=start_epoch,
        grad_clip=cfg["train"].get("grad_clip"),
        weights_dir=dirs["weights"],
        log_dir=dirs["logs"],
        ema_rate=cfg["train"].get("ema_rate", 0.9),
        display_progress=cfg["train"].get("display_progress", False),
        device=device,
    )

    print("\n" + "-" * 25 + " TRAINING " + "-" * 25 + "\n")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JRAA-DPS.")
    parser.add_argument("--data", 
                        default="configs/data.yaml",
                        help="Path to data YAML config.")
    parser.add_argument("--model", 
                        default="configs/model.yaml",
                        help="Path to model YAML config.")
    parser.add_argument("--train",
                        default="configs/train.yaml",
                        help="Path to training YAML config.")
    args = parser.parse_args()
    main(args)