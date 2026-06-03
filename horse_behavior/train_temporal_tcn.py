import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATASET_DIR = "dataset/timesequence"
DEFAULT_OUTPUT_DIR = "runs/timesequence/tcn_behavior"


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNBehaviorClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.20,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(feature_dim, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TemporalBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is [batch, time, features]; Conv1d expects [batch, features, time].
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.classifier(x)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_macro_f1: float


def resolve_project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.int64)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total = float(counts.sum())
    weights = np.zeros((num_classes,), dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = total / (float(num_classes) * counts[nonzero])
    return torch.from_numpy(weights)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == y).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_size = int(y.shape[0])
            predictions = logits.argmax(dim=1)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == y).sum().item())
            total_count += batch_size
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    predictions_np = np.concatenate(all_predictions) if all_predictions else np.empty((0,), dtype=np.int64)
    targets_np = np.concatenate(all_targets) if all_targets else np.empty((0,), dtype=np.int64)
    mean_loss = total_loss / max(1, total_count)
    mean_acc = total_correct / max(1, total_count)
    return mean_loss, mean_acc, predictions_np, targets_np


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    label_names: list[str],
    feature_names: list[str],
    epoch_metrics: EpochMetrics,
    input_shape: tuple[int, int],
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "label_names": label_names,
        "feature_names": feature_names,
        "input_shape": input_shape,
        "epoch_metrics": asdict(epoch_metrics),
        "model_config": {
            "hidden_channels": args.hidden_channels,
            "kernel_size": args.kernel_size,
            "dilations": [int(value) for value in args.dilations.split(",") if value.strip()],
            "dropout": args.dropout,
        },
    }
    torch.save(checkpoint, path)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TCN horse behavior classifier from temporal windows.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Directory produced by temporal_behavior_dataset.py.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for TCN model artifacts.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="TCN hidden channel count.")
    parser.add_argument("--kernel-size", type=int, default=3, help="Temporal convolution kernel size.")
    parser.add_argument("--dilations", default="1,2,4,8", help="Comma-separated dilation schedule.")
    parser.add_argument("--dropout", type=float, default=0.20, help="Dropout probability.")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience on validation macro F1.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def run(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    dataset_dir = resolve_project_path(args.dataset_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_npz(dataset_dir / "train.npz")
    X_val, y_val = load_npz(dataset_dir / "val.npz")
    label_names = list(load_json(dataset_dir / "label_names.json"))
    feature_names = list(load_json(dataset_dir / "feature_names.json"))

    if X_train.ndim != 3 or X_val.ndim != 3:
        raise RuntimeError("Expected X arrays shaped [windows, time, features].")
    if X_train.shape[-1] != len(feature_names):
        raise RuntimeError("Feature count in train.npz does not match feature_names.json.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dilations = tuple(int(value) for value in args.dilations.split(",") if value.strip())
    model = TCNBehaviorClassifier(
        feature_dim=X_train.shape[-1],
        num_classes=len(label_names),
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        dilations=dilations,
        dropout=args.dropout,
    ).to(device)

    weights = class_weights(y_train, len(label_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    history: list[EpochMetrics] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_predictions = np.empty((0,), dtype=np.int64)
    best_targets = np.empty((0,), dtype=np.int64)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc, val_pred, val_target = run_epoch(model, val_loader, criterion, device, optimizer=None)
        val_macro_f1 = f1_score(val_target, val_pred, labels=list(range(len(label_names))), average="macro", zero_division=0)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_macro_f1=float(val_macro_f1),
        )
        history.append(metrics)
        scheduler.step(val_macro_f1)

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_macro_f1:.3f}"
        )

        if val_macro_f1 > best_f1:
            best_f1 = float(val_macro_f1)
            best_epoch = epoch
            epochs_without_improvement = 0
            best_predictions = val_pred
            best_targets = val_target
            save_checkpoint(
                output_dir / "best.pt",
                model,
                args,
                label_names,
                feature_names,
                metrics,
                input_shape=(int(X_train.shape[1]), int(X_train.shape[2])),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    last_metrics = history[-1]
    save_checkpoint(
        output_dir / "last.pt",
        model,
        args,
        label_names,
        feature_names,
        last_metrics,
        input_shape=(int(X_train.shape[1]), int(X_train.shape[2])),
    )

    labels = list(range(len(label_names)))
    report = classification_report(
        best_targets,
        best_predictions,
        labels=labels,
        target_names=label_names,
        zero_division=0,
    )
    matrix = confusion_matrix(best_targets, best_predictions, labels=labels)
    pd.DataFrame(matrix, index=label_names, columns=label_names).to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8")
    pd.DataFrame([asdict(item) for item in history]).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8")
    write_text(output_dir / "classification_report.txt", report)

    metrics_summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_accuracy": float((best_predictions == best_targets).mean()) if best_targets.size else 0.0,
        "device": str(device),
        "train_windows": int(len(y_train)),
        "val_windows": int(len(y_val)),
        "train_class_counts": dict(Counter(label_names[int(value)] for value in y_train)),
        "val_class_counts": dict(Counter(label_names[int(value)] for value in y_val)),
        "label_names": label_names,
        "feature_count": len(feature_names),
        "time_steps": int(X_train.shape[1]),
    }
    write_text(output_dir / "metrics.json", json.dumps(metrics_summary, ensure_ascii=False, indent=2) + "\n")

    print("Best validation report:")
    print(report)
    print(f"Best model: {(output_dir / 'best.pt').resolve()}")
    print(f"Metrics: {(output_dir / 'metrics.json').resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"TCN training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
