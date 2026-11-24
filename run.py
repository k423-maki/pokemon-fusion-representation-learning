"""Portable CLI derived from the exploratory notebooks, 2026-11-23.

AI-assisted submission preparation. Saved historical results are not overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from models import ConvVAE

ROOT = Path(__file__).resolve().parent


class FusionDataset(Dataset):
    def __init__(self, path: Path, num_classes: int = 170):
        with np.load(path, allow_pickle=False) as data:
            self.images, self.labels = data["images"], data["labels"]
        if self.images.ndim != 4 or self.images.shape[1:] != (3, 128, 128):
            raise ValueError("images must have shape (N, 3, 128, 128)")
        if self.images.dtype != np.uint8:
            raise ValueError("images must be uint8 (0..255)")
        if self.labels.shape != (len(self.images),) or not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("labels must be an integer vector aligned with images")
        if len(self.images) < 2 or self.labels.min() < 0 or self.labels.max() >= num_classes:
            raise ValueError("Dataset is too small or labels are outside the configured classes")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.images[index]).float() / 255, torch.tensor(int(self.labels[index]))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def losses(output: tuple[torch.Tensor, ...], x: torch.Tensor, labels: torch.Tensor,
           kl_weight: float, ce_weight: float) -> tuple[torch.Tensor, ...]:
    recon, _, mean, logvar = output[:4]
    mse = F.mse_loss(recon, x)
    kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
    ce = F.cross_entropy(output[4], labels, label_smoothing=0.1) if len(output) == 5 else mse.new_zeros(())
    return mse + kl_weight * kl + ce_weight * ce, mse, kl, ce


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}. Choose a new directory.")
    path.mkdir(parents=True, exist_ok=True)


def make_loader(dataset: FusionDataset, indices: list[int], batch_size: int,
                shuffle: bool = False, seed: int = 42) -> DataLoader:
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(model: ConvVAE, loader: DataLoader, device: torch.device,
             max_batches: int | None = None) -> dict:
    was_training = model.training
    model.eval()
    squared_error, elements, images = 0.0, 0, 0
    for batch_index, (x, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = x.to(device)
        reconstruction = model(x)[0]
        squared_error += F.mse_loss(reconstruction, x, reduction="sum").item()
        elements += x.numel()
        images += len(x)
    model.train(was_training)
    return {"mse_per_pixel_channel": squared_error / elements,
            "squared_error_per_image": squared_error / images, "images_evaluated": images,
            "batch_limit": max_batches}


@torch.no_grad()
def save_reconstruction(model: ConvVAE, loader: DataLoader, device: torch.device, path: Path) -> None:
    model.eval()
    x, _ = next(iter(loader))
    x = x[:8].to(device)
    reconstruction = model(x)[0]
    top = np.concatenate([a.permute(1, 2, 0).cpu().numpy() for a in x], axis=1)
    bottom = np.concatenate([a.permute(1, 2, 0).cpu().numpy() for a in reconstruction], axis=1)
    grid = (np.clip(np.concatenate([top, bottom], axis=0), 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(grid).save(path)


def smoke(args: argparse.Namespace, device: torch.device) -> None:
    dataset = FusionDataset(args.data)
    x = torch.stack([dataset[i][0] for i in range(2)]).to(device)
    labels = torch.stack([dataset[i][1] for i in range(2)]).to(device)
    checks = []
    for architecture, channels, latent_shape in [("baseline", 8, [2, 8, 32, 32]), ("resnet", 32, [2, 32, 8, 8])]:
        model = ConvVAE(architecture, channels, pretrained=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        output = model(x)
        if output[0].shape != x.shape or list(output[1].shape) != latent_shape:
            raise RuntimeError("Architecture output shape mismatch")
        if architecture == "resnet" and output[4].shape != (2, 170):
            raise RuntimeError("Classifier output shape mismatch")
        total, *_ = losses(output, x, labels, 1e-4, 1.0)
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad()
        total.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("Missing or non-finite gradients")
        before = model.quant_conv.weight.detach().clone()
        optimizer.step()
        if torch.equal(before, model.quant_conv.weight):
            raise RuntimeError("Optimizer did not update the bottleneck")
        model.eval()
        with torch.no_grad():
            first, second = model(x)[0], model(x)[0]
        if not torch.equal(first, second) or not torch.isfinite(first).all():
            raise RuntimeError("Evaluation is not finite and deterministic")
        checks.append({"architecture": architecture, "latent_shape": latent_shape,
                       "output_shape": list(first.shape), "backward_and_optimizer_step": "passed",
                       "deterministic_eval": "passed", "pretrained": False})
    print(json.dumps({"purpose": "software_smoke_check_not_performance_evaluation",
                      "dataset_sample_size": len(dataset), "device": str(device),
                      "torch_version": torch.__version__, "checks": checks}, indent=2))


def train(args: argparse.Namespace, device: torch.device) -> None:
    dataset = FusionDataset(args.data)
    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    val_count = 0 if args.all_training_data else max(1, int(0.1 * len(dataset)))
    val_indices, train_indices = permutation[:val_count], permutation[val_count:]
    train_loader = make_loader(dataset, train_indices, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(dataset, val_indices, args.batch_size) if val_indices else None
    config = {"architecture": args.model, "latent_channels": args.latent_channels or (8 if args.model == "baseline" else 32),
              "num_classes": 170}
    model = ConvVAE(**config, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    new_output(args.output)
    history = []
    for epoch in range(args.epochs):
        model.train()
        warmup = args.model == "resnet" and epoch < 3
        for block in (model.decoder, model.post_quant_conv):
            for parameter in block.parameters():
                parameter.requires_grad = not warmup
            if warmup:
                block.eval()  # Freeze BatchNorm buffers as well as parameters.
        kl_weight = 1e-4 * epoch / args.epochs if args.model == "baseline" else 1e-4
        ce_weight = 2.0 if warmup else 1.0
        totals = np.zeros(4, dtype=np.float64)
        seen = 0
        for batch_index, (x, y) in enumerate(train_loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            values = losses(model(x), x, y, kl_weight, ce_weight)
            if not all(torch.isfinite(value) for value in values):
                raise RuntimeError("Non-finite loss; stopping without reporting a valid experiment")
            values[0].backward()
            optimizer.step()
            totals += np.array([v.item() for v in values]) * len(x)
            seen += len(x)
        record = {"epoch": epoch + 1, "training_images_processed": seen, "kl_weight": kl_weight,
                  **dict(zip(["loss", "reconstruction_mse", "kl", "cross_entropy"], (totals / seen).tolist()))}
        if val_loader is not None:
            record["validation"] = evaluate(model, val_loader, device, args.max_batches)
        history.append(record)
        print(json.dumps(record))
    torch.save({"config": config, "state_dict": model.state_dict(), "seed": args.seed,
                "train_indices": train_indices, "val_indices": val_indices,
                "dataset_size": len(dataset), "dataset_filename": args.data.name,
                "dataset_sha256": file_sha256(args.data),
                "epochs": args.epochs, "max_batches": args.max_batches,
                "initial_pretrained": args.pretrained}, args.output / "model.pt")
    save_json(args.output / "history.json", history)
    save_reconstruction(model, val_loader if val_loader is not None else train_loader, device,
                        args.output / "reconstruction.png")


@torch.no_grad()
def extract_latents(model: ConvVAE, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features, labels = [], []
    for x, y in loader:
        _, mean, _ = model.encode(x.to(device))
        features.append(mean.flatten(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)


def probe(args: argparse.Namespace, device: torch.device) -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not checkpoint["val_indices"]:
        raise ValueError("This checkpoint has no held-out split. Train without --all-training-data for a validation probe.")
    dataset = FusionDataset(args.data)
    if len(dataset) != checkpoint["dataset_size"]:
        raise ValueError("Dataset size differs from the checkpoint's recorded split")
    if file_sha256(args.data) != checkpoint["dataset_sha256"]:
        raise ValueError("Dataset bytes differ from the file used for training")
    model = ConvVAE(**checkpoint["config"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    train_x, train_y = extract_latents(model, make_loader(dataset, checkpoint["train_indices"], args.batch_size), device)
    val_x, val_y = extract_latents(model, make_loader(dataset, checkpoint["val_indices"], args.batch_size), device)
    if args.components > min(train_x.shape) or args.components < 1:
        raise ValueError(f"PCA components must be in 1..{min(train_x.shape)} for this dataset")
    if len(np.unique(train_y)) < 2:
        raise ValueError("Probe requires at least two training classes")
    new_output(args.output)
    pca = PCA(n_components=args.components, random_state=checkpoint["seed"])
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(pca.fit_transform(train_x), train_y)
    accuracy = accuracy_score(val_y, classifier.predict(pca.transform(val_x)))
    result = {"new_run_not_archived_score": True, "dataset_size": len(dataset),
              "train_size": len(train_y), "validation_size": len(val_y),
              "latent_dimension": train_x.shape[1], "pca_dimension": args.components,
              "validation_accuracy": float(accuracy), "seed": checkpoint["seed"]}
    save_json(args.output / "probe.json", result)
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "train", "probe"):
        p = sub.add_parser(command)
        p.add_argument("--data", type=Path, default=ROOT / "data/sample.npz")
        p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--threads", type=int, default=4)
        if command != "smoke":
            p.add_argument("--batch-size", type=int, default=16)
            p.add_argument("--output", type=Path, default=Path("runs") / command)
        if command == "train":
            p.add_argument("--model", choices=("baseline", "resnet"), default="baseline")
            p.add_argument("--latent-channels", type=int)
            p.add_argument("--pretrained", action="store_true")
            p.add_argument("--epochs", type=int, default=1)
            p.add_argument("--learning-rate", type=float, default=1e-3)
            p.add_argument("--max-batches", type=int)
            p.add_argument("--all-training-data", action="store_true")
        elif command == "probe":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--components", type=int, default=256)
    args = parser.parse_args()
    for name in ("threads", "batch_size", "epochs", "latent_channels", "max_batches"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "learning_rate", 1) <= 0:
        parser.error("--learning-rate must be positive")
    if getattr(args, "pretrained", False) and args.model != "resnet":
        parser.error("--pretrained applies only to --model resnet")
    return args


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    {"smoke": smoke, "train": train, "probe": probe}[args.command](args, device)


if __name__ == "__main__":
    main()
