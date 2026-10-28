"""Game-winner model: an MLP embedding with EQUINE's Gaussian-process head.

``EquineGP`` turns the embedding into a Laplace-approximated GP over random
Fourier features. Its mean-field logits already shrink toward 50/50 where the
GP is unsure; the predictive variance behind that shrinkage is also reported
(``novelty``: the percentile of a game's GP variance among recent held-out games),
so the page can flag matchups unlike anything the model was trained on.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import equine as eq
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .features import FEATURES

LABELS = ["away", "home"]


@dataclass
class WinnerConfig:
    # Chosen by walk-forward backtest on 2023-24..2025-26. A 64-64 embedding
    # trained a fixed 40 epochs at lr 1e-3 overfit badly (Brier ~0.245, near a
    # coin flip's 0.25); this smaller, slower, early-stopped one scores ~0.211.
    hidden: tuple[int, ...] = (32,)
    emb_dim: int = 16
    random_features: int = 256
    max_epochs: int = 40
    patience: int = 6  # epochs without a held-out log-loss improvement
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    calib_frac: float = 0.08  # most recent games: early stopping + temperature
    # EquineGP hard-codes 25; SNGP's default is pi/8. Larger values shrink each
    # logit harder toward 50/50 in proportion to the GP's variance. Not stored
    # in EQUINE's checkpoint, so WinnerModel.save/load carry it.
    # pi/8 (SNGP's value) and no GP-feature LayerNorm: with pre-game
    # availability and impact features this took the features-only model
    # from Brier 0.2139 to 0.2070 on 2023-24..2025-26 (logistic: 0.2061).
    mean_field_factor: float = 0.39269908169872414
    # EquineGP always LayerNorms the embedding before its random features,
    # which discards the magnitude of a low-dimensional input (a stacked
    # log-odds). Off for the market stack; not stored in EQUINE's checkpoint.
    normalize_gp_features: bool = False
    seed: int = 0


@dataclass
class WinnerModel:
    model: eq.EquineGP
    mean: np.ndarray
    std: np.ndarray
    var_reference: np.ndarray  # sorted GP variances of held-out games
    features: list[str]
    epochs_trained: int = 0

    # ----------------------------------------------------------------------
    def _x(self, df: pd.DataFrame) -> torch.Tensor:
        x = df[self.features].to_numpy(dtype=np.float64)
        x = np.nan_to_num((x - self.mean) / self.std)
        return torch.tensor(x, dtype=torch.float32)

    def _variance(self, embeddings: torch.Tensor) -> np.ndarray:
        laplace = self.model.model
        cov = laplace.covariance
        var = ((embeddings @ cov) * embeddings).sum(dim=1) * laplace.ridge_penalty
        return var.cpu().numpy()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["p_home", "gp_var", "novelty"])
        out = self.model.predict(self._x(df))
        var = self._variance(out.embeddings)
        novelty = np.searchsorted(self.var_reference, var) / len(self.var_reference)
        return pd.DataFrame(
            {
                "p_home": out.classes[:, 1].cpu().numpy(),
                "gp_var": var,
                "novelty": novelty,
            },
            index=df.index,
        )

    # ----------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(str(directory / "equine_gp.pt"))
        meta = {
            "features": self.features,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "var_reference": self.var_reference.tolist(),
            "mean_field_factor": self.model.model.mean_field_factor,
            "normalize_gp_features": self.model.model.normalize_gp_features,
        }
        (directory / "winner_meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory: Path) -> WinnerModel:
        model = eq.load_equine_model(str(directory / "equine_gp.pt"), device="cpu")
        assert isinstance(model, eq.EquineGP)
        meta = json.loads((directory / "winner_meta.json").read_text())
        mff = meta.get("mean_field_factor", model.model.mean_field_factor)
        model.mean_field_factor = model.model.mean_field_factor = mff
        norm = meta.get("normalize_gp_features", True)
        model.normalize_gp_features = model.model.normalize_gp_features = norm
        return cls(
            model=model,
            mean=np.array(meta["mean"]),
            std=np.array(meta["std"]),
            var_reference=np.array(meta["var_reference"]),
            features=meta["features"],
        )


def fit(
    train: pd.DataFrame,
    config: WinnerConfig | None = None,
    features: list[str] | None = None,
) -> WinnerModel:
    """Fit on final games in ``train`` (chronological; the latest slice calibrates)."""
    config = config or WinnerConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train = train.dropna(subset=["home_win"]).sort_values("start")
    n_calib = max(1, int(len(train) * config.calib_frac))
    fit_df, calib_df = train.iloc[:-n_calib], train.iloc[-n_calib:]

    features = list(features or FEATURES)
    x_all = fit_df[features].to_numpy(dtype=np.float64)
    mean = np.nanmean(x_all, axis=0)
    std = np.nanstd(x_all, axis=0)
    std[std == 0] = 1.0

    embedding = eq.MLP(
        in_features=len(features),
        hidden_sizes=list(config.hidden),
        out_features=config.emb_dim,
    )
    model = eq.EquineGP(
        embedding,
        emb_out_dim=config.emb_dim,
        num_classes=2,
        num_random_features=config.random_features,
        feature_names=features,
        label_names=LABELS,
    )
    model.mean_field_factor = config.mean_field_factor
    model.model.mean_field_factor = config.mean_field_factor
    model.normalize_gp_features = config.normalize_gp_features
    model.model.normalize_gp_features = config.normalize_gp_features
    wm = WinnerModel(model, mean, std, np.array([0.0]), features)

    def dataset(df: pd.DataFrame) -> TensorDataset:
        y = torch.tensor(df["home_win"].to_numpy(), dtype=torch.long)
        return TensorDataset(wm._x(df), y)

    opt = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    # Early stopping on the held-out slice (the most recent games), one epoch
    # per train_model call; the GP's precision matrix is rebuilt every epoch,
    # so restoring the best state restores a consistent GP.
    fit_ds, calib_ds = dataset(fit_df), dataset(calib_df)
    x_calib, y_calib = calib_ds.tensors
    best_state, best_loss, stale = None, float("inf"), 0
    wm.epochs_trained = 0
    for epoch in range(config.max_epochs):
        model.train_model(
            fit_ds,
            loss_fn=torch.nn.CrossEntropyLoss(),
            opt=opt,
            num_epochs=1,
            batch_size=config.batch_size,
        )
        p = model.predict(x_calib).classes[:, 1].clamp(1e-6, 1 - 1e-6)
        loss = -torch.mean(
            y_calib * torch.log(p) + (1 - y_calib) * torch.log(1 - p)
        ).item()
        if loss < best_loss - 1e-4:
            best_state, best_loss, stale = copy.deepcopy(model.state_dict()), loss, 0
            wm.epochs_trained = epoch + 1
        else:
            stale += 1
            if stale >= config.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.model.recompute_covariance = True
    model.calibrate_model(calib_ds, num_calibration_epochs=20)

    # Held-out games, not training games: the GP's variance is lower on the
    # points that built its precision matrix.
    ref = model.predict(wm._x(calib_df))
    wm.var_reference = np.sort(wm._variance(ref.embeddings))
    return wm
