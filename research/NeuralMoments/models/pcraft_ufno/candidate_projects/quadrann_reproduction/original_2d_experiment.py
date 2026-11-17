import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from quadrann_model import QuadrANN


UPSTREAM_COMMIT = "111dd611aaeac1fb98c76385706b003ea8ef94be"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def gen_indexes(d, max_deg=5):
    return [
        list(comb)
        for comb in product(range(max_deg + 1), repeat=d)
        if sum(comb) <= max_deg
    ]


def cos_bf(x, kmax=10):
    q = np.random.randint(1, kmax + 1)
    phi = np.random.uniform(0, 2 * np.pi)
    val = (
        np.sin(q * np.pi + phi) - np.sin(phi)
    ) / (q * np.pi)

    return torch.cos(q * np.pi * x + phi) - val


def hermite(n, x):
    if n == 0:
        return torch.ones_like(x) + cos_bf(x)
    elif n == 1:
        return x / 0.5
    elif n == 2:
        return -(x**2 - 1.) * 1.5
    elif n == 3:
        return -(x**3 - 3. * x) / 1.25
    elif n == 4:
        return (x**4 - 6. * x**2 + 3) / 1.2
    elif n == 5:
        return (x**5 - 10. * x**3 + 15. * x) * 6 / 31
    raise ValueError(n)


def multi_hermite(indices, x):
    result = torch.ones(
        x.shape[0],
        x.shape[1],
        device=x.device,
        dtype=x.dtype,
    )

    for i, degree in enumerate(indices):
        result = result * hermite(degree, x[:, :, i])

    return result


def calc_base_fn(points):
    indices = gen_indexes(2)

    result = torch.zeros(
        len(indices),
        points.shape[0],
        points.shape[1],
        device=points.device,
    )

    for i, degree in enumerate(indices):
        result[i, :, :] = multi_hermite(degree, points)

    return result


def calc_test_fn(x):
    mu = torch.tensor(
        [0.5, 0.5],
        device=x.device,
        dtype=x.dtype,
    )

    variances = torch.tensor(
        [0.025**2, 0.025**2],
        device=x.device,
        dtype=x.dtype,
    )

    sigma = torch.diag(variances)
    sigma_inv = torch.inverse(sigma)
    sigma_det = torch.det(sigma)

    norm_const = 1.0 / torch.sqrt(
        (2 * np.pi)**2 * sigma_det
    )

    centered = x - mu

    exponent = -0.5 * torch.sum(
        (centered @ sigma_inv) * centered,
        dim=-1,
    )

    pdf_val = norm_const * torch.exp(exponent)

    return pdf_val.T


def transform_domain(data, a=0.5, w=0.05):
    p = 1. / (1. + (1. / a - 1.)**(1. / 3.))
    q = 1. - p

    return w * (
        (data - p)**3 / (p**3 + q**3)
        + p**3 / (p**3 + q**3)
    ) + (1. - w) * data


class SobolDataset(Dataset):

    def __init__(self, num_points=1024):
        self.num_points = num_points

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        engine = torch.quasirandom.SobolEngine(
            dimension=2,
            scramble=True,
        )

        return engine.draw(n=self.num_points)


def loss_fn(weights, base_fn):
    ann_integr = (base_fn * weights).sum(-1)
    diff2 = (ann_integr - 1.)**2
    return diff2.mean()


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="original_256")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)

    set_seed(args.seed)

    root = Path(__file__).resolve().parent
    upstream = root.parent / "upstream" / "QuadrANN"

    checkpoint_dir = root / "checkpoints"
    report_dir = root / "reports"

    checkpoint_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    model = QuadrANN(
        dim=2,
        num_gnn_layers=3,
        k_neighbors=16,
        pe_abs=3,
        pe_rel=5,
        hidden_dim=128,
        final_hidden_dim=256,
    ).to(device)

    loader = DataLoader(
        dataset=SobolDataset(
            num_points=1024
        ),
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
    )

    history = []

    start = time.perf_counter()

    for epoch in range(args.epochs):

        model.train()

        lr = 0.0002

        if epoch % 16 == 0 and epoch > 0:
            lr *= 0.75

        # Preserve original upstream optimizer behavior.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

        loss = torch.zeros(
            torch.Size([]),
            device=device,
        )

        p1_tmp = torch.zeros(
            torch.Size([]),
            device=device,
        )

        for points_batch in loader:

            points_batch = transform_domain(
                points_batch
            ).to(device)

            predicted_weights = model(points_batch)

            model.eval()

            weights = model(points_batch)

            model.train()

            base_fn = calc_base_fn(points_batch)

            test_fn = calc_test_fn(points_batch)

            p1_tmp += (
                weights @ test_fn
            )[0, 0]

            loss += loss_fn(
                predicted_weights,
                base_fn,
            )

        avg_loss = float(loss.detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        entry = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "p1": float(p1_tmp.detach().cpu()),
            "learning_rate": lr,
        }

        history.append(entry)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss={avg_loss:.8f} "
            f"p1={entry['p1']:.8f}",
            flush=True,
        )

    model.eval()

    quadrann_values = []
    sobol_values = []

    for i in range(args.eval_samples):

        sobol_data = next(iter(loader))

        x = transform_domain(
            sobol_data
        ).to(device)

        x0 = sobol_data.to(device)

        with torch.no_grad():

            weights = model(x)

            weights0 = torch.zeros_like(weights) + 1. / 1024

            test_fn = calc_test_fn(x)
            test_fn0 = calc_test_fn(x0)

            quadrann_values.append(
                ((weights @ test_fn)[0, 0]).item()
            )

            sobol_values.append(
                ((weights0 @ test_fn0)[0, 0]).item()
            )

        if (i + 1) % 64 == 0:
            print(
                f"Evaluation [{i+1}/{args.eval_samples}]",
                flush=True,
            )

    elapsed = time.perf_counter() - start

    report = {
        "schema": "quadrann.original-reproduction.v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": sha256(
            upstream / "quadrann-src.py"
        ),
        "seed": args.seed,
        "epochs": args.epochs,
        "num_points": 1024,
        "gnn_layers": 3,
        "knn": 16,
        "hidden_dim": 128,
        "final_hidden_dim": 256,
        "eval_samples": args.eval_samples,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "history": history,
        "quadrann": {
            "mean": float(np.mean(quadrann_values)),
            "std": float(np.std(quadrann_values)),
        },
        "sobol_qmc": {
            "mean": float(np.mean(sobol_values)),
            "std": float(np.std(sobol_values)),
        },
    }

    checkpoint = {
        "model": model.state_dict(),
        "config": vars(args),
        "history": history,
        "upstream_commit": UPSTREAM_COMMIT,
    }

    torch.save(
        checkpoint,
        checkpoint_dir / f"{args.tag}.pt",
    )

    report_path = report_dir / f"{args.tag}.json"

    report_path.write_text(
        json.dumps(report, indent=2)
    )

    print("\n===== FINAL RESULT =====")

    print(json.dumps({
        "elapsed_seconds": elapsed,
        "quadrann": report["quadrann"],
        "sobol_qmc": report["sobol_qmc"],
        "checkpoint": str(
            checkpoint_dir / f"{args.tag}.pt"
        ),
        "report": str(report_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
