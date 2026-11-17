"""SQNO v1.2 Moment-only training entry point."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from training_v12.dataset import M83Corpus
from training_v12.normalization import MomentNormalizer
from training_v12.trainer import SQNOTrainer
from training_v12.checkpoint import CheckpointManager

from spin_adapter.spin_quadrann_v12 import SpinQuadrANNV12


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-train-records", type=int, default=0)
    p.add_argument("--max-val-records", type=int, default=0)

    p.add_argument("--query-chunk", type=int, default=16)
    p.add_argument("--chunk-nodes", type=int, default=4096)
    p.add_argument("--grid-size", type=int, default=64)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=830251)
    p.add_argument("--resume", default="")
    p.add_argument("--dry-run", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    corpus = M83Corpus(args.manifest)

    summary = corpus.split_summary()

    model_config = {
        "preset": "large_ms",
        "hidden_dim": 32,
        "final_hidden_dim": 128,
        "num_layers": 3,
        "neighbors": 16,
        "temperature": 0.04,
        "quadrature_order": 2,
        "chunk_intervals": 512,
        "chunk_nodes": args.chunk_nodes,
        "query_chunk": args.query_chunk,
        "grid_size": args.grid_size,
    }

    if args.dry_run:
        print(json.dumps({
            "manifest_sha256": corpus.manifest_sha256,
            "split": summary,
            "model_config": model_config,
            "device": str(device),
            "cuda_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
            "status": "preflight_only",
        }, indent=2))
        return

    normalizer = MomentNormalizer.fit(corpus)

    model = SpinQuadrANNV12(
        preset=model_config["preset"],
        hidden_dim=model_config["hidden_dim"],
        final_hidden_dim=model_config["final_hidden_dim"],
        num_layers=model_config["num_layers"],
        neighbors=model_config["neighbors"],
        temperature=model_config["temperature"],
        quadrature_order=model_config["quadrature_order"],
        chunk_intervals=model_config["chunk_intervals"],
        chunk_nodes=model_config["chunk_nodes"],
        query_chunk=model_config["query_chunk"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def schedule(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)

        progress = (
            (epoch - args.warmup_epochs)
            / max(1, args.epochs - args.warmup_epochs)
        )

        return 0.5 * (
            1.0 + np.cos(np.pi * min(progress, 1.0))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, schedule
    )

    manager = CheckpointManager(
        output / "checkpoints",
        corpus.manifest_sha256,
        model_config,
    )

    start_epoch = 0
    global_step = 0

    if args.resume:
        restored = manager.load(
            args.resume,
            model,
            optimizer,
            scheduler,
        )

        start_epoch = restored["epoch"]
        global_step = restored["global_step"]

        normalizer = MomentNormalizer.from_state_dict(
            restored["normalizer"]
        )

        print(
            f"RESUMED epoch={start_epoch} "
            f"step={global_step}",
            flush=True,
        )

    trainer = SQNOTrainer(
        model=model,
        corpus=corpus,
        normalizer=normalizer,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        grid_size=args.grid_size,
        query_chunk=args.query_chunk,
        grad_clip=args.grad_clip,
        output_dir=output / "reports",
    )

    trainer.global_step = global_step

    config = {
        "arguments": vars(args),
        "model": model_config,
        "manifest_sha256": corpus.manifest_sha256,
        "normalizer": normalizer.state_dict(),
        "split": summary,
    }

    (output / "training_config.json").write_text(
        json.dumps(config, indent=2)
    )

    if args.max_train_records:
        corpus.splits["train"] = (
            corpus.splits["train"][:args.max_train_records]
        )

    if args.max_val_records:
        corpus.splits["val"] = (
            corpus.splits["val"][:args.max_val_records]
        )

    trainer.fit(
        epochs=args.epochs,
        checkpoint_manager=manager,
        seed=args.seed,
        start_epoch=start_epoch,
    )

    print("SQNO TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
