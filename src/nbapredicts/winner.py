"""Game-winner model: an embedding network with EQUINE's Gaussian-process head.

``EquineGP`` turns the embedding into a Laplace-approximated GP over random
Fourier features. Its mean-field logits already shrink toward 50/50 where the
GP is unsure; the predictive variance behind that shrinkage is also reported
(``novelty``: the percentile of a game's GP variance among recent held-out games),
so the page can flag matchups unlike anything the model was trained on.

The features-only model trains on every game twice, once from each side
(``mirrored``), and averages a prediction with its mirror's; its embedding is
a spectral-norm residual network (SNGP's recipe), so distances between games
survive into the GP and the novelty means what it says. Together they cut
the no-line Brier by 0.0006 on 2023-24..2025-26 and 0.0017 on 2021-22..2022-23
(docs/results/README.md, "Representations for EQUINE's GP head").
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

# How each feature reads from the away team's side: a home-minus-away
# difference (or a log-odds) flips sign, a home/away pair swaps, and the rest
# read the same from either side. A feature without a rule can't be mirrored.
MIRROR_PAIRS = (
    ("rest_home", "rest_away"),
    ("b2b_home", "b2b_away"),
    ("avail_delta_home", "avail_delta_away"),
    ("impact_delta_home", "impact_delta_away"),
)
MIRROR_SAME = frozenset({"neutral", "playoff", "pace_l_mean", "season_games_min"})
MIRROR_FLIP = frozenset({"mkt", "s1"})  # the stack's log-odds inputs


def mirror_rule(feature: str) -> str:
    """'flip', 'swap' or 'same'; ValueError for a feature with no rule."""
    if feature.endswith("_diff") or feature in MIRROR_FLIP:
        return "flip"
    if any(feature in pair for pair in MIRROR_PAIRS):
        return "swap"
    if feature in MIRROR_SAME:
        return "same"
    raise ValueError(f"no mirror rule for feature {feature!r} (winner.py)")


def home_side(df: pd.DataFrame) -> pd.Series:
    """+1 when the listed home team has home court, 0 at a neutral site, -1
    for a mirrored row (the listed home team is really the away team)."""
    if "home_side" in df:
        return df["home_side"].astype(float)
    return 1.0 - df["neutral"].astype(float)


def mirrored(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The same games seen from the away team's side, label flipped."""
    out = df.copy()
    partner = {a: b for pair in MIRROR_PAIRS for a, b in (pair, pair[::-1])}
    for f in features:
        rule = mirror_rule(f)
        if rule == "flip":
            out[f] = -df[f]
        elif rule == "swap":
            out[f] = df[partner[f]]
    out["home_side"] = -home_side(df)
    if "home_win" in df:
        out["home_win"] = 1 - df["home_win"]
    return out


class SNResMLP(torch.nn.Module):
    """SNGP's embedding: a free input projection, then a residual block whose
    branch is spectral-norm bounded (Lipschitz below 1, so the block neither
    collapses nor stretches distances), then a bounded projection."""

    def __init__(
        self, d_in: int, width: int, d_out: int, blocks: int = 1, c: float = 0.95
    ):
        super().__init__()
        sn = torch.nn.utils.parametrizations.spectral_norm
        self.inp = torch.nn.Linear(d_in, width)
        self.blocks = torch.nn.ModuleList(
            [sn(torch.nn.Linear(width, width)) for _ in range(blocks)]
        )
        self.out = sn(torch.nn.Linear(width, d_out))
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.inp(x))
        for b in self.blocks:
            h = h + self.c * torch.relu(b(h))
        return self.out(h)


# EQUINE saves an embedding as a registered architecture's recipe plus its
# weights, and rebuilds it from the registry on load.
eq.register_embedding_architecture("nbapredicts.sn_residual_mlp", SNResMLP)


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
    # Train on every game from both sides and average each prediction with
    # its mirror's, and embed with SNResMLP (scripts/representation.py chose
    # both; the market stack keeps a plain MLP without mirroring).
    mirror: bool = True
    encoder: str = "snres"  # "snres" or "mlp" (eq.MLP)
    seed: int = 0


@dataclass
class WinnerModel:
    model: eq.EquineGP
    mean: np.ndarray
    std: np.ndarray
    var_reference: np.ndarray  # sorted GP variances of held-out games
    features: list[str]
    epochs_trained: int = 0
    mirror: bool = False

    # ----------------------------------------------------------------------
    def _x(self, df: pd.DataFrame) -> torch.Tensor:
        x = _raw(df, self.features, self.mirror)
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
        p_home = out.classes[:, 1].cpu().numpy()
        if self.mirror:
            # The same game from the away side should give the other team's
            # chance; averaging the two makes the prediction symmetric.
            back = self.model.predict(self._x(mirrored(df, self.features)))
            p_home = (p_home + 1 - back.classes[:, 1].cpu().numpy()) / 2
        return pd.DataFrame(
            {
                "p_home": p_home,
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
            "mirror": self.mirror,
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
            mirror=meta.get("mirror", False),
        )


def _raw(df: pd.DataFrame, features: list[str], mirror: bool) -> np.ndarray:
    """The model's input columns, unscaled: the features, plus which side has
    home court for a mirrored model."""
    x = df[features].to_numpy(dtype=np.float64)
    if mirror:
        x = np.column_stack([x, home_side(df).to_numpy(dtype=np.float64)])
    return x


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
    if config.mirror:
        # Mirrored copies of the training games only: the held-out newest
        # games, which steer early stopping and calibration, stay as played.
        fit_df = fit_df.assign(home_side=home_side(fit_df))
        fit_df = pd.concat([fit_df, mirrored(fit_df, features)], ignore_index=True)
    x_all = _raw(fit_df, features, config.mirror)
    mean = np.nanmean(x_all, axis=0)
    std = np.nanstd(x_all, axis=0)
    std[std == 0] = 1.0

    names = [*features, "home_side"] if config.mirror else features
    if config.encoder == "snres":
        embedding = SNResMLP(len(names), config.hidden[0], config.emb_dim)
    else:
        embedding = eq.MLP(
            in_features=len(names),
            hidden_sizes=list(config.hidden),
            out_features=config.emb_dim,
        )
    model = eq.EquineGP(
        embedding,
        emb_out_dim=config.emb_dim,
        num_classes=2,
        num_random_features=config.random_features,
        feature_names=names,
        label_names=LABELS,
    )
    model.mean_field_factor = config.mean_field_factor
    model.model.mean_field_factor = config.mean_field_factor
    model.normalize_gp_features = config.normalize_gp_features
    model.model.normalize_gp_features = config.normalize_gp_features
    wm = WinnerModel(model, mean, std, np.array([0.0]), features, mirror=config.mirror)

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
