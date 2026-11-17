"""Geometry-level Moment-only training for SQNO v1.2."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from training_v12.losses import moment_mse
from training_v12.metrics import evaluate_moments
from training_v12.geometry_cache import GeometryCacheRunner


def model_moments(result):
    # Fixed moments are already restored by SQNO v1.2.
    if "moments_total" not in result:
        raise KeyError("SQNO output missing moments_total")
    return result["moments_total"]


class SQNOTrainer:
    def __init__(
        self,
        model,
        corpus,
        normalizer,
        optimizer,
        scheduler=None,
        device="cuda",
        grid_size=64,
        query_chunk=16,
        grad_clip=1.0,
        output_dir=".",
    ):
        self.model = model
        self.corpus = corpus
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.grid_size = int(grid_size)
        self.query_chunk = int(query_chunk)
        self.grad_clip = float(grad_clip)
        self.output_dir = Path(output_dir)

        self.cache_runner = GeometryCacheRunner(
            model, grid_size=self.grid_size
        )

        self.scales = normalizer.tensor(
            self.device,
            dtype=next(model.parameters()).dtype,
        )

        self.global_step = 0

    def _evaluate_geometry(
        self, raw_geometry, labels, training
    ):
        cache = self.cache_runner.encode(raw_geometry)

        queries = torch.as_tensor(
            np.array(labels["queries"], copy=True),
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )

        targets = torch.as_tensor(
            np.array(labels["moments"], copy=True),
            device=self.device,
            dtype=queries.dtype,
        )

        if len(queries) == 0:
            raise ValueError("geometry has no queries")

        losses = []
        predictions = []

        for start in range(0, len(queries), self.query_chunk):
            stop = min(start + self.query_chunk, len(queries))

            result = self.cache_runner.evaluate(
                cache, queries[start:stop]
            )

            prediction = model_moments(result)

            target = targets[start:stop]

            chunk_loss = moment_mse(
                prediction, target, self.scales
            )

            # Weighted aggregation gives the exact query mean.
            weighted_loss = (
                chunk_loss * (stop-start) / len(queries)
            )

            losses.append(weighted_loss)
            predictions.append(prediction)

        loss = torch.stack(losses).sum()

        prediction_all = torch.cat(predictions, dim=0)

        if prediction_all.shape != targets.shape:
            raise RuntimeError("prediction shape mismatch")

        if training:
            self.optimizer.zero_grad(set_to_none=True)

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip,
            )

            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("nonfinite gradient norm")

            self.optimizer.step()
        else:
            grad_norm = torch.zeros(
                (), device=self.device
            )

        return {
            "loss": float(loss.detach()),
            "prediction": prediction_all.detach(),
            "target": targets.detach(),
            "grad_norm": float(grad_norm),
            "query_count": len(queries),
        }

    def run_epoch(self, split, epoch=0, seed=830251):
        training = split == "train"

        self.model.train(training)

        rows = list(self.corpus.splits[split])

        if training:
            rng = np.random.default_rng(seed + epoch)
            order = rng.permutation(len(rows))
            rows = [rows[i] for i in order]

        total_loss = 0.0
        geometry_count = 0
        query_count = 0
        grad_norms = []

        predictions = []
        targets = []

        started = time.perf_counter()

        for row in rows:
            geometry, labels = self.corpus.load(row)

            if training:
                result = self._evaluate_geometry(
                    geometry, labels, training=True
                )
                self.global_step += 1
            else:
                with torch.no_grad():
                    result = self._evaluate_geometry(
                        geometry, labels, training=False
                    )

            total_loss += result["loss"]
            geometry_count += 1
            query_count += result["query_count"]

            grad_norms.append(result["grad_norm"])

            predictions.append(
                result["prediction"].cpu().numpy()
            )
            targets.append(
                result["target"].cpu().numpy()
            )

        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)

        metric = evaluate_moments(
            prediction,
            target,
            self.normalizer.scales,
        )

        metric.update({
            "split": split,
            "epoch": int(epoch),
            "geometries": geometry_count,
            "queries": query_count,
            "mean_geometry_loss": total_loss / geometry_count,
            "elapsed_seconds": time.perf_counter()-started,
            "mean_grad_norm": float(np.mean(grad_norms)),
            "geometry_encode_calls": (
                self.model.geometry_encode_calls
            ),
            "query_evaluation_calls": (
                self.model.query_evaluation_calls
            ),
        })

        return metric

    def fit(
        self,
        epochs,
        checkpoint_manager=None,
        seed=830251,
        start_epoch=0,
        best_val=float("inf"),
    ):
        history = []

        for epoch in range(start_epoch, epochs):
            train_metrics = self.run_epoch(
                "train", epoch, seed
            )

            val_metrics = self.run_epoch(
                "val", epoch, seed
            )

            if self.scheduler is not None:
                self.scheduler.step()

            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            history.append(record)

            report = self.output_dir / "epoch_metrics.jsonl"
            report.parent.mkdir(parents=True, exist_ok=True)

            with report.open("a") as f:
                f.write(json.dumps(record) + "\n")

            if checkpoint_manager is not None:
                checkpoint_manager.save(
                    name="latest",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    global_step=self.global_step,
                    normalizer=self.normalizer,
                )

                if val_metrics["normalized_rmse"] < best_val:
                    best_val = val_metrics["normalized_rmse"]

                    checkpoint_manager.save(
                        name="best_val",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch + 1,
                        global_step=self.global_step,
                        normalizer=self.normalizer,
                    )

            print(
                f"Epoch {epoch+1}/{epochs} "
                f"train={train_metrics['normalized_rmse']:.6f} "
                f"val={val_metrics['normalized_rmse']:.6f} "
                f"time={train_metrics['elapsed_seconds']:.1f}s",
                flush=True,
            )

        return history
