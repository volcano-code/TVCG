"""Atomic SQNO training checkpoints with strict resume validation."""

import os
import random
from pathlib import Path

import numpy as np
import torch


class CheckpointManager:
    def __init__(self, directory, manifest_sha256, model_config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_sha256 = manifest_sha256
        self.model_config = model_config

    @staticmethod
    def rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    @staticmethod
    def restore_rng(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

        if (
            state.get("cuda") is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(state["cuda"])

    def save(
        self,
        name,
        model,
        optimizer,
        scheduler,
        epoch,
        global_step,
        normalizer,
    ):
        destination = self.directory / f"{name}.pt"
        temporary = self.directory / f".{name}.tmp.pt"

        payload = {
            "schema": "sqno.training_v12.checkpoint.v1",
            "manifest_sha256": self.manifest_sha256,
            "model_config": self.model_config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None else None
            ),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "normalizer": normalizer.state_dict(),
            "rng": self.rng_state(),
        }

        torch.save(payload, temporary)
        os.replace(temporary, destination)

        return destination

    def load(
        self,
        path,
        model,
        optimizer=None,
        scheduler=None,
        restore_rng=True,
    ):
        path = Path(path)

        try:
            state = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            state = torch.load(path, map_location="cpu")

        if state.get("schema") != "sqno.training_v12.checkpoint.v1":
            raise ValueError("unsupported checkpoint schema")

        if state["manifest_sha256"] != self.manifest_sha256:
            raise ValueError("M83 manifest SHA256 mismatch")

        if state["model_config"] != self.model_config:
            raise ValueError("model configuration mismatch")

        model.load_state_dict(state["model"], strict=True)

        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])

        if scheduler is not None:
            if state["scheduler"] is None:
                raise ValueError("checkpoint lacks scheduler state")
            scheduler.load_state_dict(state["scheduler"])

        if restore_rng:
            self.restore_rng(state["rng"])

        return {
            "epoch": int(state["epoch"]),
            "global_step": int(state["global_step"]),
            "normalizer": state["normalizer"],
        }
