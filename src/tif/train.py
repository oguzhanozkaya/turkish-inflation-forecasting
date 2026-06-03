"""Model training stage."""

from __future__ import annotations

import json
import os
import pickle
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

import tif.utils

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


class NumericMLP(nn.Module):
    """Tabular numeric deep learning baseline."""

    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, numeric_features: torch.Tensor) -> torch.Tensor:
        return self.network(numeric_features).squeeze(-1)


class NumericGRU(nn.Module):
    """GRU model over lag-structured numeric feature sequences."""

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.10) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, numeric_sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(numeric_sequence)
        return self.head(hidden[-1]).squeeze(-1)


class TextCNNEncoder(nn.Module):
    """TextCNN encoder with randomly initialized project-trained embeddings."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 64,
        channel_count: int = 48,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channel_count, kernel_size=kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output_size = channel_count * len(kernel_sizes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for convolution in self.convolutions:
            activation = torch.relu(convolution(embedded))
            pooled.append(torch.max(activation, dim=2).values)
        return self.dropout(torch.cat(pooled, dim=1))


class TextCNNRegressor(nn.Module):
    """Text-only inflation forecast model."""

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.encoder = TextCNNEncoder(vocabulary_size)
        self.head = nn.Linear(self.encoder.output_size, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(token_ids)).squeeze(-1)


class FusionRegressor(nn.Module):
    """Numeric plus text fusion forecast model."""

    def __init__(self, numeric_input_size: int, vocabulary_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.text_encoder = TextCNNEncoder(vocabulary_size)
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_input_size, hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + self.text_encoder.output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, numeric_features: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        numeric_representation = self.numeric_projection(numeric_features)
        text_representation = self.text_encoder(token_ids)
        return self.head(torch.cat([numeric_representation, text_representation], dim=1)).squeeze(-1)


LAG_FEATURE_PATTERN = re.compile(r"(?P<base>.+)_lag_(?P<lag>\d+)$")
SEQUENCE_STEPS = (12, 6, 3, 2, 1, 0)


@dataclass(frozen=True)
class TrainingConfig:
    """Runtime training configuration."""

    seed: int = 447
    epochs: int = 80
    patience: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    random_forest_trees: int = 200
    device: str = "auto"

    @classmethod
    def from_environment(cls) -> TrainingConfig:
        return cls(
            seed=int(os.environ.get("TIF_SEED", "447")),
            epochs=int(os.environ.get("TIF_EPOCHS", "80")),
            patience=int(os.environ.get("TIF_PATIENCE", "12")),
            batch_size=int(os.environ.get("TIF_BATCH_SIZE", "32")),
            learning_rate=float(os.environ.get("TIF_LEARNING_RATE", "0.001")),
            random_forest_trees=int(os.environ.get("TIF_RANDOM_FOREST_TREES", "200")),
            device=os.environ.get("TIF_DEVICE", "auto"),
        )


@dataclass(frozen=True)
class TrainingResult:
    """Summary of generated training artifacts."""

    predictions_csv_path: Path
    predictions_parquet_path: Path
    summary_path: Path
    history_csv_path: Path
    history_markdown_path: Path
    training_figure_paths: tuple[Path, ...]
    model_count: int
    prediction_rows: int


class TrainingError(RuntimeError):
    """Raised when processed features are missing or invalid."""


class ForecastTensorDataset(Dataset):
    """Tensor dataset shared by all PyTorch model variants."""

    def __init__(
        self,
        numeric_features: np.ndarray,
        numeric_sequence: np.ndarray,
        token_ids: np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.numeric_features = torch.tensor(numeric_features, dtype=torch.float32)
        self.numeric_sequence = torch.tensor(numeric_sequence, dtype=torch.float32)
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "numeric": self.numeric_features[index],
            "sequence": self.numeric_sequence[index],
            "tokens": self.token_ids[index],
            "target": self.target[index],
        }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(config: TrainingConfig) -> torch.device:
    if config.device == "cpu":
        return torch.device("cpu")
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError("TIF_DEVICE=cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pad_token_sequences(sequences: pd.Series, max_length: int = tif.utils.MAX_TOKENS) -> np.ndarray:
    token_matrix = np.zeros((len(sequences), max_length), dtype=np.int64)
    for row_index, sequence in enumerate(sequences):
        values = list(sequence) if isinstance(sequence, (list, tuple, np.ndarray)) else []
        bounded = values[:max_length]
        if bounded:
            token_matrix[row_index, : len(bounded)] = np.array(bounded, dtype=np.int64)
    return token_matrix


def _sequence_plan(numeric_feature_columns: list[str]) -> tuple[list[str], list[int]]:
    bases = set()
    available_lags: dict[str, set[int]] = {}
    for column in numeric_feature_columns:
        match = LAG_FEATURE_PATTERN.fullmatch(column)
        if not match:
            continue
        base = match.group("base")
        lag = int(match.group("lag"))
        bases.add(base)
        available_lags.setdefault(base, set()).add(lag)
    variables = sorted(base for base in bases if len(available_lags.get(base, set())) >= 2)
    return variables, list(SEQUENCE_STEPS)


def _build_numeric_sequence(
    scaled_numeric: pd.DataFrame, sequence_variables: list[str], sequence_steps: list[int]
) -> np.ndarray:
    sequence = np.zeros((len(scaled_numeric), len(sequence_steps), len(sequence_variables)), dtype=np.float32)
    for step_index, lag in enumerate(sequence_steps):
        for variable_index, variable in enumerate(sequence_variables):
            column = f"{variable}_lag_{lag}"
            if column in scaled_numeric.columns:
                sequence[:, step_index, variable_index] = scaled_numeric[column].to_numpy(dtype=np.float32)
    return sequence


def _split_indices(dataset: pd.DataFrame, split_name: str) -> list[int]:
    return dataset.index[dataset["split"] == split_name].tolist()


def _model_output(model: nn.Module, batch: dict[str, torch.Tensor], input_kind: str) -> torch.Tensor:
    if input_kind == "numeric":
        return model(batch["numeric"])
    if input_kind == "sequence":
        return model(batch["sequence"])
    if input_kind == "text":
        return model(batch["tokens"])
    if input_kind == "fusion":
        return model(batch["numeric"], batch["tokens"])
    raise ValueError(f"Unknown model input kind: {input_kind}")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _train_torch_model(
    model_name: str,
    model: nn.Module,
    tensor_dataset: ForecastTensorDataset,
    train_indices: list[int],
    validation_indices: list[int],
    input_kind: str,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    train_loader = DataLoader(Subset(tensor_dataset, train_indices), batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(
        Subset(tensor_dataset, validation_indices or train_indices), batch_size=config.batch_size, shuffle=False
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale_epochs = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_model_output(model, batch, input_kind), batch["target"])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                batch = _move_batch(batch, device)
                validation_losses.append(
                    float(criterion(_model_output(model, batch, input_kind), batch["target"]).cpu())
                )
        validation_loss = float(np.mean(validation_losses))
        train_loss = float(np.mean(train_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            "train: epoch "
            f"model={model_name} epoch={epoch}/{config.epochs} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} best_validation_loss={best_loss:.6f} stale_epochs={stale_epochs}"
        )
        if stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    return model.to("cpu"), {"best_validation_loss": best_loss, "epochs_ran": len(history), "history": history}


def _predict_torch_model(
    model: nn.Module,
    tensor_dataset: ForecastTensorDataset,
    input_kind: str,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            predictions.append(_model_output(model, batch, input_kind).cpu().numpy())
    return np.concatenate(predictions)


def _prediction_frame(dataset: pd.DataFrame, model_name: str, model_type: str, predictions: np.ndarray) -> pd.DataFrame:
    frame = dataset[
        [
            "forecast_origin_month_start",
            "forecast_origin_month",
            "target_month_start",
            "target_month",
            "split",
            "target_cpi_mom_percent",
            "cpi_mom_lag_1",
        ]
    ].copy()
    frame = frame.rename(
        columns={
            "target_cpi_mom_percent": "actual_cpi_mom_percent",
            "cpi_mom_lag_1": "previous_cpi_mom_percent",
        }
    )
    if "cpi_mom_trailing_std_12" in dataset.columns:
        frame["cpi_mom_trailing_std_12"] = dataset["cpi_mom_trailing_std_12"].to_numpy(dtype=float)
    frame["model_name"] = model_name
    frame["model_type"] = model_type
    frame["prediction_cpi_mom_percent"] = predictions.astype(float)
    return frame


def _write_training_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = ["# Training Summary", "", f"Device: `{summary['device']}`", ""]
    lines.append("| Model | Type | Status | Detail |")
    lines.append("| ----- | ---- | ------ | ------ |")
    for model_name, model_summary in summary["models"].items():
        detail = model_summary.get("detail", "")
        lines.append(f"| `{model_name}` | {model_summary['type']} | {model_summary['status']} | {detail} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _training_history_frame(summary: dict[str, object]) -> pd.DataFrame:
    rows = []
    for model_name, model_summary in summary["models"].items():
        history = model_summary.get("history", [])
        epoch_history = history.get("history", []) if isinstance(history, dict) else history
        for epoch_metrics in epoch_history:
            rows.append(
                {
                    "model_name": model_name,
                    "model_type": model_summary["type"],
                    "epoch": epoch_metrics["epoch"],
                    "train_loss": epoch_metrics["train_loss"],
                    "validation_loss": epoch_metrics["validation_loss"],
                }
            )
    return pd.DataFrame(rows)


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_training_history_artifacts(
    paths: tif.utils.ProjectPaths,
    summary: dict[str, object],
) -> tuple[Path, Path, tuple[Path, ...]]:
    history = _training_history_frame(summary)
    history_csv_path = paths.reports / "training_history.csv"
    history_markdown_path = paths.reports / "training_history.md"
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_csv_path, index=False)

    lines = ["# Training History", ""]
    if history.empty:
        lines.append("No neural training history was recorded.")
    else:
        lines.append("| Model | Epochs | Best Validation Loss | Final Train Loss | Final Validation Loss |")
        lines.append("| ----- | ------ | -------------------- | ---------------- | --------------------- |")
        for model_name, group in history.groupby("model_name", sort=True):
            final = group.sort_values("epoch").iloc[-1]
            lines.append(
                f"| `{model_name}` | {len(group)} | {group['validation_loss'].min():.6f} | "
                f"{final.train_loss:.6f} | {final.validation_loss:.6f} |"
            )
    history_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure_paths: list[Path] = []
    if not history.empty:
        for model_name, group in history.groupby("model_name", sort=True):
            group = group.sort_values("epoch")
            plt.figure(figsize=(8, 4.5))
            plt.plot(group["epoch"], group["train_loss"], label="Train loss")
            plt.plot(group["epoch"], group["validation_loss"], label="Validation loss")
            plt.title(f"Training Loss: {model_name}")
            plt.xlabel("Epoch")
            plt.ylabel("MSE loss")
            plt.legend()
            path = paths.figures / f"training_loss_{model_name}.png"
            _save_current_figure(path)
            figure_paths.append(path)

        plt.figure(figsize=(9, 5))
        for model_name, group in history.groupby("model_name", sort=True):
            group = group.sort_values("epoch")
            plt.plot(group["epoch"], group["validation_loss"], label=model_name)
        plt.title("Validation Loss by Neural Model")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.legend()
        path = paths.figures / "training_validation_loss_comparison.png"
        _save_current_figure(path)
        figure_paths.append(path)

    return history_csv_path, history_markdown_path, tuple(figure_paths)


def train_models(
    paths: tif.utils.ProjectPaths = tif.utils.DEFAULT_PATHS,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train baselines, classical models, and raw PyTorch models."""

    tif.utils.ensure_generated_directories(paths)
    config = TrainingConfig.from_environment() if config is None else config
    print(f"train: config={json.dumps(asdict(config), sort_keys=True)}")
    _set_seed(config.seed)
    dataset_path = paths.processed_data / "model_dataset.parquet"
    metadata_path = paths.processed_data / "feature_metadata.json"
    vocabulary_path = paths.processed_data / "text_vocabulary.json"
    missing = [path for path in (dataset_path, metadata_path, vocabulary_path) if not path.is_file()]
    if missing:
        missing_paths = ", ".join(path.relative_to(paths.root).as_posix() for path in missing)
        raise TrainingError(f"Missing processed artifacts: {missing_paths}. Run `just preprocess` first.")

    dataset = pd.read_parquet(dataset_path).sort_values("forecast_origin_month_start").reset_index(drop=True)
    metadata = _read_json(metadata_path)
    vocabulary = _read_json(vocabulary_path)
    numeric_feature_columns = list(metadata["numeric_feature_columns"])
    target = dataset["target_cpi_mom_percent"].to_numpy(dtype=np.float32)
    train_indices = _split_indices(dataset, "train")
    validation_indices = _split_indices(dataset, "validation")
    if not train_indices:
        raise TrainingError("Processed dataset does not contain training rows")
    split_counts = dataset["split"].value_counts().reindex(["train", "validation", "test"], fill_value=0).to_dict()
    print(
        "train: dataset "
        f"rows={len(dataset)} splits={split_counts} numeric_features={len(numeric_feature_columns)} "
        f"vocabulary={len(vocabulary)} target_mean={float(target.mean()):.4f} target_std={float(target.std()):.4f}"
    )

    scaler = StandardScaler()
    scaler.fit(dataset.loc[train_indices, numeric_feature_columns])
    numeric_scaled = scaler.transform(dataset[numeric_feature_columns]).astype(np.float32)
    scaled_numeric_frame = pd.DataFrame(numeric_scaled, columns=numeric_feature_columns)
    sequence_variables, sequence_steps = _sequence_plan(numeric_feature_columns)
    numeric_sequence = _build_numeric_sequence(scaled_numeric_frame, sequence_variables, sequence_steps)
    token_ids = _pad_token_sequences(dataset["text_token_ids"], int(metadata["max_text_tokens"]))
    tensor_dataset = ForecastTensorDataset(numeric_scaled, numeric_sequence, token_ids, target)
    print(
        "train: tensors "
        f"numeric_shape={numeric_scaled.shape} sequence_shape={numeric_sequence.shape} token_shape={token_ids.shape}"
    )

    predictions = [
        _prediction_frame(dataset, "last_value", "baseline", dataset["cpi_mom_lag_1"].to_numpy(dtype=float)),
        _prediction_frame(
            dataset,
            "rolling_mean_3",
            "baseline",
            dataset["cpi_mom_rolling_mean_3"].to_numpy(dtype=float),
        ),
    ]
    model_summary: dict[str, dict[str, object]] = {
        "last_value": {"type": "baseline", "status": "trained", "detail": "uses cpi_mom_lag_1"},
        "rolling_mean_3": {
            "type": "baseline",
            "status": "trained",
            "detail": "uses cpi_mom_rolling_mean_3",
        },
    }

    ridge = Ridge(alpha=1.0)
    ridge.fit(numeric_scaled[train_indices], target[train_indices])
    predictions.append(_prediction_frame(dataset, "ridge", "classical", ridge.predict(numeric_scaled)))
    model_summary["ridge"] = {"type": "classical", "status": "trained", "detail": "numeric features"}
    print("train: model=ridge status=trained input=numeric_features")

    forest = RandomForestRegressor(
        n_estimators=config.random_forest_trees,
        random_state=config.seed,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    forest.fit(numeric_scaled[train_indices], target[train_indices])
    predictions.append(_prediction_frame(dataset, "random_forest", "classical", forest.predict(numeric_scaled)))
    model_summary["random_forest"] = {
        "type": "classical",
        "status": "trained",
        "detail": f"{config.random_forest_trees} trees",
    }
    print(f"train: model=random_forest status=trained trees={config.random_forest_trees}")

    device = _resolve_device(config)
    print(f"train: device={device}")
    torch_models: list[tuple[str, str, nn.Module, str]] = [
        ("numeric_mlp", "deep_numeric", NumericMLP(len(numeric_feature_columns)), "numeric"),
        ("numeric_gru", "deep_numeric", NumericGRU(len(sequence_variables)), "sequence"),
        ("text_cnn", "deep_text", TextCNNRegressor(len(vocabulary)), "text"),
        ("fusion_mlp", "deep_fusion", FusionRegressor(len(numeric_feature_columns), len(vocabulary)), "fusion"),
    ]
    for model_name, model_type, model, input_kind in torch_models:
        trained_model, history = _train_torch_model(
            model_name, model, tensor_dataset, train_indices, validation_indices, input_kind, config, device
        )
        predictions.append(
            _prediction_frame(
                dataset,
                model_name,
                model_type,
                _predict_torch_model(trained_model, tensor_dataset, input_kind, config.batch_size),
            )
        )
        torch.save(
            {
                "state_dict": trained_model.state_dict(),
                "input_kind": input_kind,
                "metadata": {
                    "numeric_feature_columns": numeric_feature_columns,
                    "sequence_variables": sequence_variables,
                    "sequence_steps": sequence_steps,
                    "vocabulary_size": len(vocabulary),
                },
            },
            paths.models / f"{model_name}.pt",
        )
        model_summary[model_name] = {
            "type": model_type,
            "status": "trained",
            "detail": f"{history['epochs_ran']} epochs, best validation loss {history['best_validation_loss']:.4f}",
            "history": history,
        }

    with (paths.models / "classical_models.pkl").open("wb") as output:
        pickle.dump({"scaler": scaler, "ridge": ridge, "random_forest": forest}, output)

    predictions_frame = pd.concat(predictions, ignore_index=True)
    predictions_csv_path = paths.predictions / "predictions.csv"
    predictions_parquet_path = paths.predictions / "predictions.parquet"
    predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(predictions_csv_path, index=False)
    predictions_frame.to_parquet(predictions_parquet_path, index=False)

    summary = {
        "config": asdict(config),
        "device": str(device),
        "row_count": len(dataset),
        "numeric_feature_count": len(numeric_feature_columns),
        "vocabulary_size": len(vocabulary),
        "sequence_variables": sequence_variables,
        "sequence_steps": sequence_steps,
        "models": model_summary,
    }
    summary_path = paths.models / "training_summary.json"
    _write_json(summary_path, summary)
    _write_training_markdown(paths.reports / "training_summary.md", summary)
    history_csv_path, history_markdown_path, training_figure_paths = _write_training_history_artifacts(paths, summary)
    return TrainingResult(
        predictions_csv_path=predictions_csv_path,
        predictions_parquet_path=predictions_parquet_path,
        summary_path=summary_path,
        history_csv_path=history_csv_path,
        history_markdown_path=history_markdown_path,
        training_figure_paths=training_figure_paths,
        model_count=len(model_summary),
        prediction_rows=len(predictions_frame),
    )


def main() -> int:
    try:
        result = train_models(tif.utils.DEFAULT_PATHS)
    except (TrainingError, ValueError, FileNotFoundError) as exc:
        print(f"train: {exc}")
        return 1
    print(f"train: trained {result.model_count} models")
    print(
        "train: wrote "
        f"{result.prediction_rows} prediction rows to "
        f"{result.predictions_csv_path.relative_to(tif.utils.DEFAULT_PATHS.root)}"
    )
    print(f"train: wrote training history to {result.history_csv_path.relative_to(tif.utils.DEFAULT_PATHS.root)}")
    print(
        "train: wrote training history report to "
        f"{result.history_markdown_path.relative_to(tif.utils.DEFAULT_PATHS.root)}"
    )
    print(f"train: wrote {len(result.training_figure_paths)} training figures")
    for figure_path in result.training_figure_paths:
        print(f"train: wrote {figure_path.relative_to(tif.utils.DEFAULT_PATHS.root)}")
    return 0
