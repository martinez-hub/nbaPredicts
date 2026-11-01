"""Better representations for EQUINE's GP head (experiment).

EQUINE's features-only model is a small MLP over the 28 game features with
EquineGP's head on top, trained on one bit per game (did the home team win).
A logistic regression on the same features ties it, so the embedding finds
nothing nonlinear. Each arm here changes how the head's input is learned,
everything else as in ``winner.fit``: the newest 8% of training games held
out for early stopping, temperature and the novelty reference.

Arms are recipes of independent options, so combinations need no new code:

- input:    ``tab`` (the 28 features), ``lineup`` (the 22 non-roster
            features plus each team's expected rotation, player by player,
            instead of the six availability and impact summaries), ``seq``
            (each team's last 10 game lines, raw, instead of the 15 rolling
            team summaries) or ``seq_plus`` (the 28 features and the lines)
- encoder:  ``mlp`` (``eq.MLP``, as production) or ``snres`` (SNGP's
            recipe: a residual block under a spectral-norm bound, so
            distances in the embedding track distances in the input)
- pretrain: ``margin`` (the embedding first learns the final margin, a
            richer target than win/loss), ``outcome`` (it learns the whole
            outcome: both teams' points, pace, shooting, turnovers,
            rebounding, free throws, three-point rate and the margin),
            ``jepa`` (self-supervised: from a view with feature groups
            masked, predict a slow-moving target encoder's embedding of the
            whole game) or ``tjepa`` (temporal
            JEPA on the sequence input: from a team's older games, predict
            the target encoder's embedding of its three most recent)
- frozen:   keep a pretrained embedding fixed while the GP head trains
- mirror:   each training game twice, the second with home and away swapped
            (differences negated, the label flipped, a home-court column)
- sym:      at test time, average p(game) with 1 - p(mirrored game)

Seeds matter here (training is stochastic), so every arm runs on the same
seeds and the baseline's seed-to-seed spread is reported as the noise floor.

Steps:

    uv run python scripts/representation.py build            # features, rosters
    uv run python scripts/representation.py build_seq        # game lines
    uv run python scripts/representation.py check            # own loop == fit()
    uv run python scripts/representation.py run --arms base margin_ft ...
    uv run python scripts/representation.py report
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nbapredicts.winner import SNResMLP  # the deployed encoder

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "cache.nosync/repr"
PREDS = WORK / "preds"
SEASONS = (2024, 2025, 2026)  # test seasons 2023-24..2025-26
P = 15  # players per team in the lineup input
PLAYER_COLS = ["rating", "proj_min", "p_play", "post_value", "exp_min"]
K = 10  # games of each team's history in the sequence input
SEQ_COLS = [
    "off_rtg", "def_rtg", "net_rtg", "pace", "efg", "opp_efg", "tov_pct",
    "opp_tov_pct", "orb_pct", "ftr", "fg3a_rate", "pts", "opp_pts",
    "win", "is_home", "rest",
]  # fmt: skip
LABELS = ["away", "home"]
RNG = np.random.default_rng(0)

log = logging.getLogger("repr")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def build() -> None:
    """Game features as the daily job builds them, and each game's rosters."""
    from nbapredicts.collect import FIRST_SEASON, Store
    from nbapredicts.features import (
        apply_pregame_availability,
        build_features,
        player_values,
        pregame_availability,
        pregame_rosters,
        team_state,
    )
    from nbapredicts.impact import (
        apply_impact,
        monthly_ratings,
        pregame_impact,
        projected_minutes,
    )
    from nbapredicts.pipeline import MODEL_FEATURES

    store = Store(ROOT / "data")
    games, tb, pb = (
        store.load("games"),
        store.load("team_box"),
        store.load("player_box"),
    )
    teams = json.loads((ROOT / "data/teams.json").read_text())
    team_ids = {t["name"]: tid for tid, t in teams.items()}
    reports = store.load("injury_reports")
    pv = player_values(pb, games)
    ratings = monthly_ratings(pb, games, store.load("stints"))
    feats = build_features(games, team_state(games, tb, pv, []), [])
    feats = apply_pregame_availability(
        feats, pregame_availability(pv, games, reports, team_ids)
    )
    feats = apply_impact(
        feats, pregame_impact(pv, pb, games, reports, team_ids, ratings)
    )
    feats = feats.merge(
        store.load("odds")[["game_id", "p_close"]], on="game_id", how="left"
    )
    keep = ["game_id", "start", "season", "home_id", "away_id", "home_pts", "away_pts"]
    feats = feats[
        (feats["status"] == "final")
        & (feats["season"] > FIRST_SEASON)
        & feats["home_win"].notna()
    ][[*keep, "home_win", "p_close", *MODEL_FEATURES]].reset_index(drop=True)

    # The rotation each team could expect before tip-off, as pregame_impact
    # sees it, player by player.
    mins = projected_minutes(pb, games)
    app = pv.merge(
        mins[["game_id", "athlete_id", "proj_min"]], on=["game_id", "athlete_id"]
    )
    roster = pregame_rosters(
        app, games, reports, team_ids, 10, columns=("proj_min", "post_value")
    )
    roster = roster[roster["game_id"].isin(set(feats["game_id"]))]
    roster["month"] = (
        pd.to_datetime(roster["start"], utc=True)
        .dt.tz_localize(None)
        .dt.to_period("M")
        .dt.to_timestamp()
        .dt.tz_localize("UTC")
    )
    roster = roster.merge(ratings, on=["month", "athlete_id"], how="left")
    roster["rating"] = roster["rating"].fillna(0.0)
    roster["exp_min"] = roster["proj_min"] * roster["p_play"]
    roster = roster[["game_id", "team_id", "athlete_id", *PLAYER_COLS]]
    WORK.mkdir(parents=True, exist_ok=True)
    feats.to_pickle(WORK / "feats.pkl")
    roster.to_pickle(WORK / "rosters.pkl")
    print(
        json.dumps(
            {
                "games": len(feats),
                "with_line": int(feats["p_close"].notna().sum()),
                "roster_rows": len(roster),
                "players_per_team": round(
                    roster.groupby(["game_id", "team_id"]).size().mean(), 1
                ),
            }
        )
    )


def build_seq() -> None:
    """Each game's last K game lines for both teams, before tip-off:
    (games, 2, K, features + mask), oldest first, padded at the front."""
    from nbapredicts.collect import Store
    from nbapredicts.features import team_games

    store = Store(ROOT / "data")
    tg = team_games(store.load("games"), store.load("team_box"))
    days = pd.to_datetime(tg["date"]).groupby(tg["team_id"]).diff().dt.days
    tg["rest"] = days.clip(upper=10).fillna(10)
    tg["is_home"] = tg["is_home"].astype(float)
    feats = pd.read_pickle(WORK / "feats.pkl")
    f = len(SEQ_COLS)
    out = np.zeros((len(feats), 2, K, f + 1), dtype=np.float32)
    by_team = {
        tid: (g["start"].to_numpy(), g[SEQ_COLS].to_numpy(float))
        for tid, g in tg.groupby("team_id")
    }
    starts = feats["start"].to_numpy()
    for i, (start, h, a) in enumerate(
        zip(starts, feats["home_id"], feats["away_id"], strict=True)
    ):
        for side, tid in enumerate((h, a)):
            when, vals = by_team[tid]
            pos = int(np.searchsorted(when, start, "left"))
            window = vals[max(0, pos - K) : pos]
            out[i, side, K - len(window) :, :f] = window
            out[i, side, K - len(window) :, f] = 1.0
    np.save(WORK / "seq.npy", out)
    print(
        json.dumps(
            {
                "games": len(feats),
                "full_windows": float((out[..., -1].sum(-1) == K).mean()),
            }
        )
    )


def model_features() -> list[str]:
    from nbapredicts.pipeline import MODEL_FEATURES

    return list(MODEL_FEATURES)


# The outcome a game's embedding can be pretrained to predict (targets only,
# from games already played; never inputs).
OUTCOME_STATS = ["pts", "efg", "tov_pct", "orb_pct", "ftr", "fg3a_rate"]
OUTCOME_COLS = [
    *[f"out_{side}_{c}" for c in OUTCOME_STATS for side in ("home", "away")],
    "out_pace",
    "out_margin",
]


def outcomes() -> pd.DataFrame:
    """Each final game's outcome, home and away, from its box score."""
    from nbapredicts.collect import Store
    from nbapredicts.features import team_games

    store = Store(ROOT / "data")
    tg = team_games(store.load("games"), store.load("team_box"))
    home = tg[tg["is_home"]].set_index("game_id")
    away = tg[~tg["is_home"]].set_index("game_id")
    out = pd.DataFrame(index=home.index.intersection(away.index))
    for c in OUTCOME_STATS:
        out[f"out_home_{c}"], out[f"out_away_{c}"] = home[c], away[c]
    out["out_pace"] = home["pace"]
    out["out_margin"] = home["pts"] - home["opp_pts"]
    return out[OUTCOME_COLS].reset_index()


# Mirroring: a game seen from the away side. Every feature is one of these.
SWAP = [
    ("rest_home", "rest_away"),
    ("b2b_home", "b2b_away"),
    ("avail_delta_home", "avail_delta_away"),
    ("impact_delta_home", "impact_delta_away"),
]
SYMMETRIC = ["neutral", "playoff", "pace_l_mean", "season_games_min"]
ROSTER_FEATURES = [
    "avail_diff",
    "avail_delta_home",
    "avail_delta_away",
    "impact_diff",
    "impact_delta_home",
    "impact_delta_away",
]


# What the sequence input replaces: the rolling team summaries.
SUMMARY_FEATURES = [
    "elo_diff", "net_rtg_s_diff", "net_rtg_l_diff", "off_rtg_l_diff",
    "def_rtg_l_diff", "efg_l_diff", "opp_efg_l_diff", "tov_pct_l_diff",
    "opp_tov_pct_l_diff", "orb_pct_l_diff", "ftr_l_diff", "fg3a_rate_l_diff",
    "pace_l_mean", "win10_diff", "season_win_diff",
]  # fmt: skip


def mirrored(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """The same games from the away side: differences negated, home/away
    pairs swapped, the label and margin flipped."""
    out = df.copy()
    for c in cols:
        if c.endswith("_diff"):
            out[c] = -df[c]
    for a, b in SWAP:
        if a in cols:
            out[a], out[b] = df[b], df[a]
    out["home_side"] = -df["home_side"]
    out["home_win"] = 1 - df["home_win"]
    out["margin"] = -df["margin"]
    if "out_margin" in df:
        for c in OUTCOME_STATS:
            h, a = f"out_home_{c}", f"out_away_{c}"
            out[h], out[a] = df[a], df[h]
        out["out_margin"] = -df["out_margin"]
    if "mkt" in df:
        out["mkt"], out["s1"] = -df["mkt"], -df["s1"]
    out["_mirror"] = True
    return out


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------


class DeepSets(torch.nn.Module):
    """A team as the sum of its players, each encoded by the same small net."""

    def __init__(self, f: int, d: int = 16):
        super().__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(f, d),
            torch.nn.ReLU(),
            torch.nn.Linear(d, d),
            torch.nn.ReLU(),
        )

    def forward(self, vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (self.phi(vals) * mask.unsqueeze(-1)).sum(dim=1)


class GRUTeam(torch.nn.Module):
    """A team as a GRU's state after reading its game lines, oldest first
    (the mask is an input, so padding at the front reads as no game)."""

    def __init__(self, f: int, d: int = 16):
        super().__init__()
        self.gru = torch.nn.GRU(f + 1, d, batch_first=True)

    def forward(self, vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(torch.cat([vals * mask.unsqueeze(-1), mask.unsqueeze(-1)], 2))
        return out[:, -1]


class PairEncoder(torch.nn.Module):
    """Both teams through the same team net; the game is the two teams'
    difference and sum next to the other game features."""

    def __init__(self, n_other: int, slots: int, f: int, head, team_net):
        super().__init__()
        self.n_other, self.slots, self.f = n_other, slots, f
        self.head = head  # takes n_other + 2 x 16
        self.team_net = team_net

    def team(self, block: torch.Tensor) -> torch.Tensor:
        n = self.slots * self.f
        vals = block[:, :n].reshape(-1, self.slots, self.f)
        return self.team_net(vals, block[:, n:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = self.slots * (self.f + 1)
        other = x[:, : self.n_other]
        home = self.team(x[:, self.n_other : self.n_other + width])
        away = self.team(x[:, self.n_other + width :])
        return self.head(torch.cat([other, home - away, home + away], dim=1))


class JEPAInput(torch.nn.Module):
    """The encoder JEPA pretrains takes the features and a mask of which
    were hidden; downstream nothing is hidden."""

    def __init__(self, encoder: torch.nn.Module, d: int):
        super().__init__()
        self.encoder, self.d = encoder, d

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros_like(x[:, : self.d])
        return self.encoder(torch.cat([x * (1 - mask), mask], dim=1))


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    input: str = "tab"  # tab | lineup | seq | seq_plus
    encoder: str = "mlp"  # mlp | snres
    pretrain: str | None = None  # None | margin | outcome | jepa | tjepa
    frozen: bool = False
    mirror: bool = False
    sym: bool = False


ARMS = {
    "base": Arm(),
    "margin_ft": Arm(pretrain="margin"),
    "margin_frozen": Arm(pretrain="margin", frozen=True),
    "snres": Arm(encoder="snres"),
    "mirror": Arm(mirror=True),
    "mirror_sym": Arm(mirror=True, sym=True),
    "lineup": Arm(input="lineup"),
    "outcome_ft": Arm(pretrain="outcome"),
    "outcome_frozen": Arm(pretrain="outcome", frozen=True),
    "jepa_ft": Arm(pretrain="jepa"),
    "jepa_frozen": Arm(pretrain="jepa", frozen=True),
    "seq": Arm(input="seq"),
    "seq_plus": Arm(input="seq_plus"),
    "tjepa_ft": Arm(input="seq", pretrain="tjepa"),
    "tjepa_frozen": Arm(input="seq", pretrain="tjepa", frozen=True),
}


def arm_of(name: str) -> Arm:
    """A named arm, or a combination written as names joined by '+'
    (e.g. 'margin_ft+mirror_sym+snres')."""
    if name in ARMS:
        return ARMS[name]
    if name.startswith("stack_"):
        return arm_of(name.removeprefix("stack_"))
    fields: dict = {}
    for part in name.split("+"):
        a = ARMS[part]
        for k, v in a.__dict__.items():
            if v != Arm.__dataclass_fields__[k].default:
                fields[k] = v
    return Arm(**fields)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


class Inputs:
    """Turns game frames into the model's input matrix; statistics come from
    the fit portion of the training games only. The lineup and sequence
    inputs add a block per team (slots x features, then a mask)."""

    def __init__(self, arm: Arm, cols: list[str], fit: pd.DataFrame, side: dict):
        self.arm, self.cols, self.side = arm, list(cols), side
        drop = {"lineup": ROSTER_FEATURES, "seq": SUMMARY_FEATURES}.get(arm.input, [])
        self.tab_cols = [c for c in self.cols if c not in drop]
        if arm.mirror and "mkt" not in self.cols:
            self.tab_cols = [*self.tab_cols, "home_side"]
        x = fit[self.tab_cols].to_numpy(float)
        self.mean, self.std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        self.std[self.std == 0] = 1.0
        self.slots, self.f = {
            "lineup": (P, len(PLAYER_COLS)),
            "seq": (K, len(SEQ_COLS)),
            "seq_plus": (K, len(SEQ_COLS)),
        }.get(arm.input, (0, 0))
        if self.slots:
            bl = self.raw_blocks(fit)
            vals, mask = bl[..., :-1], bl[..., -1:]
            n = mask.sum()
            self.b_mean = (vals * mask).sum(axis=(0, 1, 2)) / n
            sq = (((vals - self.b_mean) * mask) ** 2).sum(axis=(0, 1, 2)) / n
            self.b_std = np.sqrt(sq)
            self.b_std[self.b_std == 0] = 1.0

    @property
    def n_tab(self) -> int:
        return len(self.tab_cols)

    def raw_blocks(self, df: pd.DataFrame) -> np.ndarray:
        """(games, 2 teams, slots, features + mask), home first; a mirrored
        row has its teams swapped."""
        flipped = (
            df["_mirror"].fillna(False).astype(bool).to_numpy()
            if "_mirror" in df
            else np.zeros(len(df), bool)
        )
        if self.arm.input in ("seq", "seq_plus"):
            idx, seq = self.side["seq"]
            out = seq[[idx[g] for g in df["game_id"]]].astype(float)
            out[flipped] = out[flipped][:, ::-1]
            return out
        out = np.zeros((len(df), 2, self.slots, self.f + 1))
        by = self.side["rosters"]
        for i, (gid, h, a, m) in enumerate(
            zip(df["game_id"], df["home_id"], df["away_id"], flipped, strict=True)
        ):
            for s, tid in enumerate((a, h) if m else (h, a)):
                rows = by.get((gid, tid))
                if rows is None:
                    continue
                k = min(self.slots, len(rows))
                out[i, s, :k, : self.f] = rows[:k]
                out[i, s, :k, self.f] = 1.0
        return out

    def blocks(self, df: pd.DataFrame) -> np.ndarray:
        """raw_blocks standardized, padding left at zero."""
        bl = self.raw_blocks(df)
        mask = bl[..., -1:]
        vals = np.nan_to_num((bl[..., :-1] - self.b_mean) / self.b_std) * mask
        return np.concatenate([vals, mask], axis=-1)

    def __call__(self, df: pd.DataFrame) -> torch.Tensor:
        x = np.nan_to_num((df[self.tab_cols].to_numpy(float) - self.mean) / self.std)
        if self.slots:
            bl = self.blocks(df)
            parts = [x]
            for s in range(2):
                parts += [bl[:, s, :, :-1].reshape(len(df), -1), bl[:, s, :, -1]]
            x = np.concatenate(parts, axis=1)
        return torch.tensor(x, dtype=torch.float32)

    def shifted(self, df: pd.DataFrame, k: float = 4.0) -> torch.Tensor:
        """The same games pushed far from the data (feature values x k),
        for the distance-awareness check; masks stay masks."""
        x = self(df).clone()
        if not self.slots:
            return x * k
        width = self.slots * (self.f + 1)
        x[:, : self.n_tab] *= k
        for s in range(2):
            start = self.n_tab + s * width
            x[:, start : start + self.slots * self.f] *= k
        return x


def roster_lookup(rosters: pd.DataFrame) -> dict:
    r = rosters.sort_values("exp_min", ascending=False)
    return {
        k: g[PLAYER_COLS].to_numpy(float) for k, g in r.groupby(["game_id", "team_id"])
    }


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def make_encoder(arm: Arm, inputs: Inputs, cfg) -> torch.nn.Module:
    import equine as eq

    d_in = inputs.n_tab
    if inputs.slots:
        d_in = inputs.n_tab + 32  # PairEncoder's head input: other + 2 x 16

    def core(d: int) -> torch.nn.Module:
        if arm.encoder == "snres":
            return SNResMLP(d, cfg.hidden[0], cfg.emb_dim)
        return eq.MLP(
            in_features=d, hidden_sizes=list(cfg.hidden), out_features=cfg.emb_dim
        )

    if arm.pretrain == "jepa":
        enc = JEPAInput(core(2 * d_in), d_in)
    else:
        enc = core(d_in)
    if inputs.slots:
        team = DeepSets(inputs.f) if arm.input == "lineup" else GRUTeam(inputs.f)
        enc = PairEncoder(inputs.n_tab, inputs.slots, inputs.f, enc, team)
    return enc


def _early_stopped(step, evaluate, max_epochs: int, patience: int, module) -> float:
    best, best_loss, stale = None, float("inf"), 0
    for _ in range(max_epochs):
        step()
        loss = evaluate()
        if loss < best_loss - 1e-4:
            best, best_loss, stale = copy.deepcopy(module.state_dict()), loss, 0
        else:
            stale += 1
            if stale >= patience:
                break
    module.load_state_dict(best)
    return best_loss


def pretrain_targets(enc, x_fit, t_fit, x_cal, t_cal, cfg) -> None:
    """The embedding plus a linear head learn the targets first (the margin,
    or the whole outcome), each standardized; missing targets don't count."""
    t_fit, t_cal = np.atleast_2d(t_fit.T).T, np.atleast_2d(t_cal.T).T
    head = torch.nn.Linear(cfg.emb_dim, t_fit.shape[1])
    net = torch.nn.Sequential(enc, head)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=cfg.weight_decay)
    mu, sd = np.nanmean(t_fit, axis=0), np.nanstd(t_fit, axis=0)
    sd[sd == 0] = 1.0

    def tensors(t):
        z = (t - mu) / sd
        return torch.tensor(np.nan_to_num(z), dtype=torch.float32), torch.tensor(
            ~np.isnan(z), dtype=torch.float32
        )

    (y_fit, w_fit), (y_cal, w_cal) = tensors(t_fit), tensors(t_cal)

    def loss(x, y, w):
        return (((net(x) - y) ** 2) * w).sum() / w.sum().clamp(min=1)

    def step():
        net.train()
        for idx in torch.randperm(len(x_fit)).split(cfg.batch_size):
            opt.zero_grad()
            loss(x_fit[idx], y_fit[idx], w_fit[idx]).backward()
            opt.step()

    def evaluate():
        net.eval()
        with torch.no_grad():
            return float(loss(x_cal, y_cal, w_cal))

    _early_stopped(step, evaluate, 100, 8, net)


GROUPS = {  # feature groups JEPA hides together, as on the slides
    "strength": [
        "elo_diff",
        "net_rtg_s_diff",
        "net_rtg_l_diff",
        "off_rtg_l_diff",
        "def_rtg_l_diff",
    ],
    "factors": [
        "efg_l_diff",
        "opp_efg_l_diff",
        "tov_pct_l_diff",
        "opp_tov_pct_l_diff",
        "orb_pct_l_diff",
        "ftr_l_diff",
        "fg3a_rate_l_diff",
        "pace_l_mean",
    ],
    "form": ["win10_diff", "season_win_diff", "season_games_min"],
    "schedule": [
        "rest_home",
        "rest_away",
        "b2b_home",
        "b2b_away",
        "neutral",
        "playoff",
        "home_side",
    ],
    "roster": ROSTER_FEATURES,
}


def pretrain_jepa(
    enc: JEPAInput, inputs: Inputs, x_fit, x_cal, cfg, momentum: float = 0.996
) -> None:
    """JEPA: from a view with one to three feature groups hidden, predict
    the target encoder's embedding of the whole game; the target encoder is
    a moving average of the context encoder (no labels used)."""
    d = enc.d
    group_idx = [
        [i for i, c in enumerate(inputs.tab_cols) if c in cols]
        for cols in GROUPS.values()
    ]
    group_idx = [g for g in group_idx if g]
    target = copy.deepcopy(enc)
    for p in target.parameters():
        p.requires_grad_(False)
    pred = torch.nn.Sequential(
        torch.nn.Linear(cfg.emb_dim + d, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, cfg.emb_dim),
    )
    opt = torch.optim.Adam(
        [*enc.parameters(), *pred.parameters()], lr=1e-3, weight_decay=cfg.weight_decay
    )
    member = torch.zeros(len(group_idx), d)
    for g, idx in enumerate(group_idx):
        member[g, idx] = 1.0

    def masks(n: int) -> torch.Tensor:
        # One to three whole groups per game, chosen at random.
        rank = torch.rand(n, len(group_idx)).argsort(dim=1).argsort(dim=1)
        k = torch.randint(1, 4, (n, 1))
        return ((rank < k).float() @ member).clamp(max=1.0)

    cal_mask = masks(len(x_cal))

    def loss_on(x, m):
        z = enc(x, m)
        with torch.no_grad():
            t = torch.nn.functional.layer_norm(target(x), (cfg.emb_dim,))
        return torch.nn.functional.smooth_l1_loss(pred(torch.cat([z, m], dim=1)), t)

    def step():
        enc.train()
        for idx in torch.randperm(len(x_fit)).split(cfg.batch_size):
            opt.zero_grad()
            loss = loss_on(x_fit[idx], masks(len(idx)))
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pt, pc in zip(target.parameters(), enc.parameters(), strict=True):
                    pt.mul_(momentum).add_(pc.detach(), alpha=1 - momentum)

    def evaluate():
        enc.eval()
        with torch.no_grad():
            return float(loss_on(x_cal, cal_mask))

    _early_stopped(step, evaluate, 60, 6, torch.nn.ModuleList([enc, pred]))


def pretrain_tjepa(team_net, b_fit, b_cal, cfg, k: int = 3, momentum: float = 0.996):
    """Temporal JEPA: from a team's older games, predict the target team
    net's embedding of its k most recent; the target net is a moving average
    of the context net. Everything in a window is before the game."""
    f = b_fit.shape[-1] - 1

    def windows(b: np.ndarray) -> torch.Tensor:
        seqs = torch.tensor(b.reshape(-1, K, f + 1), dtype=torch.float32)
        keep = (seqs[:, : K - k, -1].sum(1) > 0) & (seqs[:, K - k :, -1].sum(1) > 0)
        return seqs[keep]

    w_fit, w_cal = windows(b_fit), windows(b_cal)
    target = copy.deepcopy(team_net)
    for p in target.parameters():
        p.requires_grad_(False)
    d = 16
    pred = torch.nn.Sequential(
        torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, d)
    )
    opt = torch.optim.Adam(
        [*team_net.parameters(), *pred.parameters()],
        lr=1e-3,
        weight_decay=cfg.weight_decay,
    )

    def loss_on(w):
        ctx, tgt = w[:, : K - k], w[:, K - k :]
        z = team_net(ctx[..., :-1], ctx[..., -1])
        with torch.no_grad():
            t = torch.nn.functional.layer_norm(
                target(tgt[..., :-1], tgt[..., -1]), (d,)
            )
        return torch.nn.functional.smooth_l1_loss(pred(z), t)

    def step():
        team_net.train()
        for idx in torch.randperm(len(w_fit)).split(cfg.batch_size):
            opt.zero_grad()
            loss_on(w_fit[idx]).backward()
            opt.step()
            with torch.no_grad():
                for pt, pc in zip(
                    target.parameters(), team_net.parameters(), strict=True
                ):
                    pt.mul_(momentum).add_(pc.detach(), alpha=1 - momentum)

    def evaluate():
        team_net.eval()
        with torch.no_grad():
            return float(loss_on(w_cal))

    _early_stopped(step, evaluate, 60, 6, torch.nn.ModuleList([team_net, pred]))


def gp_variance(model, embeddings: torch.Tensor) -> np.ndarray:
    laplace = model.model
    var = ((embeddings @ laplace.covariance) * embeddings).sum(
        dim=1
    ) * laplace.ridge_penalty
    return var.detach().cpu().numpy()


def train(arm: Arm, cols: list[str], train_df: pd.DataFrame, cfg, seed: int, side):
    """As winner.fit: newest 8% of games held out (original orientation
    only), early stopping on their log loss, temperature, novelty
    reference. Returns (predict(df) -> frame, inputs)."""
    import equine as eq

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_df = train_df.sort_values("start")
    n_cal = max(1, int(len(train_df) * cfg.calib_frac))
    fit_df, cal_df = train_df.iloc[:-n_cal], train_df.iloc[-n_cal:]
    if arm.mirror:
        fit_df = pd.concat([fit_df, mirrored(fit_df, cols)], ignore_index=True)
    inputs = Inputs(arm, cols, fit_df, side)
    x_fit, x_cal = inputs(fit_df), inputs(cal_df)
    y_fit = torch.tensor(fit_df["home_win"].to_numpy(), dtype=torch.long)
    y_cal = torch.tensor(cal_df["home_win"].to_numpy(), dtype=torch.long)

    enc = make_encoder(arm, inputs, cfg)
    if arm.pretrain in ("margin", "outcome"):
        targets = ["margin"] if arm.pretrain == "margin" else OUTCOME_COLS
        pretrain_targets(
            enc,
            x_fit,
            fit_df[targets].to_numpy(float),
            x_cal,
            cal_df[targets].to_numpy(float),
            cfg,
        )
    elif arm.pretrain == "jepa":
        jepa = enc.head if arm.input == "lineup" else enc
        if arm.input == "lineup":
            raise ValueError("jepa over the lineup input is not implemented")
        pretrain_jepa(jepa, inputs, x_fit, x_cal, cfg)
    elif arm.pretrain == "tjepa":
        pretrain_tjepa(enc.team_net, inputs.blocks(fit_df), inputs.blocks(cal_df), cfg)
    if arm.frozen:
        # What was pretrained stays as it is: the team net for tjepa.
        for p in (enc.team_net if arm.pretrain == "tjepa" else enc).parameters():
            p.requires_grad_(False)

    model = eq.EquineGP(
        enc,
        emb_out_dim=cfg.emb_dim,
        num_classes=2,
        num_random_features=cfg.random_features,
        label_names=LABELS,
    )
    model.mean_field_factor = model.model.mean_field_factor = cfg.mean_field_factor
    model.normalize_gp_features = model.model.normalize_gp_features = (
        cfg.normalize_gp_features
    )
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    fit_ds = torch.utils.data.TensorDataset(x_fit, y_fit)
    cal_ds = torch.utils.data.TensorDataset(x_cal, y_cal)

    def step():
        model.train_model(
            fit_ds,
            loss_fn=torch.nn.CrossEntropyLoss(),
            opt=opt,
            num_epochs=1,
            batch_size=cfg.batch_size,
        )

    def evaluate():
        p = model.predict(x_cal).classes[:, 1].clamp(1e-6, 1 - 1e-6)
        return -torch.mean(y_cal * torch.log(p) + (1 - y_cal) * torch.log(1 - p)).item()

    _early_stopped(step, evaluate, cfg.max_epochs, cfg.patience, model)
    model.model.recompute_covariance = True
    model.calibrate_model(cal_ds, num_calibration_epochs=20)
    var_ref = np.sort(gp_variance(model, model.predict(x_cal).embeddings))

    def predict(df: pd.DataFrame) -> pd.DataFrame:
        out = model.predict(inputs(df))
        var = gp_variance(model, out.embeddings)
        res = pd.DataFrame(
            {
                "game_id": df["game_id"].to_numpy(),
                "p": out.classes[:, 1].detach().cpu().numpy(),
                "var": var,
                "novelty": np.searchsorted(var_ref, var) / len(var_ref),
            }
        )
        if arm.sym:
            back = (
                model.predict(inputs(mirrored(df, cols)))
                .classes[:, 1]
                .detach()
                .cpu()
                .numpy()
            )
            res["p"] = (res["p"] + 1 - back) / 2
        shifted = model.predict(inputs.shifted(df))
        res["var_shifted"] = gp_variance(model, shifted.embeddings)
        return res

    return predict


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def load(arms: list[Arm] | None = None) -> tuple[pd.DataFrame, dict]:
    """The games, and what the lineup and sequence inputs read (loaded only
    when an arm needs them)."""
    feats = pd.read_pickle(WORK / "feats.pkl")
    feats["margin"] = feats["home_pts"] - feats["away_pts"]
    feats["home_side"] = 1.0 - feats["neutral"].astype(float)
    inputs = {a.input for a in arms} if arms is not None else {"lineup", "seq"}
    if arms is None or any(a.pretrain == "outcome" for a in arms):
        path = WORK / "outcomes.pkl"
        if not path.exists():
            outcomes().to_pickle(path)
        feats = feats.merge(pd.read_pickle(path), on="game_id", how="left")
    side: dict = {}
    if "lineup" in inputs:
        side["rosters"] = roster_lookup(pd.read_pickle(WORK / "rosters.pkl"))
    if inputs & {"seq", "seq_plus"}:
        idx = {g: i for i, g in enumerate(feats["game_id"])}
        side["seq"] = (idx, np.load(WORK / "seq.npy"))
    return feats, side


def stack_frames(feats: pd.DataFrame, season: int):
    """Training and test games with a closing line, with the stack's two
    inputs: the line's log-odds and stage 1's (out of fold for training)."""
    from nbapredicts.stack import Stage1, market_logit, out_of_fold_stage1

    cols = model_features()
    tr = feats[(feats["season"] < season) & feats["p_close"].notna()].copy()
    te = feats[(feats["season"] == season) & feats["p_close"].notna()].copy()
    tr["mkt"], te["mkt"] = market_logit(tr["p_close"]), market_logit(te["p_close"])
    tr["s1"] = out_of_fold_stage1(tr, cols)
    te["s1"] = Stage1.fit(tr, cols).logit(te)
    return tr, te


def run(names: list[str], seeds: list[int], seasons: tuple[int, ...] = SEASONS) -> None:
    from nbapredicts.stack import STACK_CONFIG
    from nbapredicts.winner import WinnerConfig

    feats, side = load([arm_of(n) for n in names])
    PREDS.mkdir(parents=True, exist_ok=True)
    stack_cache: dict = {}
    for name in names:
        arm = arm_of(name)
        stack = name.startswith("stack_")
        for season in seasons:
            for seed in seeds:
                path = PREDS / f"{name}_{season}_{seed}.pkl"
                if path.exists():
                    continue
                t0 = time.time()
                if stack:
                    if season not in stack_cache:
                        stack_cache[season] = stack_frames(feats, season)
                    tr, te = stack_cache[season]
                    cols, cfg = ["mkt", "s1"], replace(STACK_CONFIG, seed=seed)
                else:
                    tr, te = (
                        feats[feats["season"] < season],
                        feats[feats["season"] == season],
                    )
                    cols, cfg = model_features(), replace(WinnerConfig(), seed=seed)
                predict = train(arm, cols, tr, cfg, seed, side)
                predict(te).to_pickle(path)
                log.info("%s %d seed %d: %.0fs", name, season, seed, time.time() - t0)


def check() -> None:
    """The experiment's loop reproduces winner.fit for the baseline."""
    from nbapredicts.winner import WinnerConfig, fit

    feats, side = load([ARMS["base"]])
    season = SEASONS[0]
    tr, te = feats[feats["season"] < season], feats[feats["season"] == season]
    t0 = time.time()
    ours = train(ARMS["base"], model_features(), tr, WinnerConfig(), 0, side)(te)
    t1 = time.time()
    prod = fit(tr, WinnerConfig(), features=model_features()).predict(te)
    print(
        json.dumps(
            {
                "max_abs_diff": float(
                    np.abs(ours["p"].to_numpy() - prod["p_home"].to_numpy()).max()
                ),
                "seconds_one_fit": round(t1 - t0, 1),
            }
        )
    )


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def paired(y, a, b, n=2000):
    """Mean Brier(a) - Brier(b) and its bootstrap 95% interval."""
    d = (a - y) ** 2 - (b - y) ** 2
    boots = d[RNG.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return [
        round(float(d.mean()), 5),
        *np.round(np.percentile(boots, [2.5, 97.5]), 5).tolist(),
    ]


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(
        roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])
    )


def nn_distance(feats: pd.DataFrame, season: int) -> pd.Series:
    """Each test game's distance to its nearest training game, in the
    standardized 28 features (the same yardstick for every arm)."""
    from sklearn.neighbors import NearestNeighbors

    cols = model_features()
    tr, te = feats[feats["season"] < season], feats[feats["season"] == season]
    x = tr[cols].to_numpy(float)
    mu, sd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    sd[sd == 0] = 1
    z = lambda d: np.nan_to_num((d[cols].to_numpy(float) - mu) / sd)  # noqa: E731
    dist = NearestNeighbors(n_neighbors=1).fit(z(tr)).kneighbors(z(te))[0][:, 0]
    return pd.Series(dist, index=te["game_id"].to_numpy())


MIX = ("mirror_sym", "snres", "lineup", "seq")  # an ensemble of diverse arms


def logistic(feats: pd.DataFrame, seasons) -> pd.Series:
    """The L2 logistic on the 28 features (stack stage 1), season by season:
    the linear yardstick EQUINE's features-only model has tied so far."""
    from nbapredicts.stack import Stage1

    out = []
    for s in seasons:
        tr, te = feats[feats["season"] < s], feats[feats["season"] == s]
        z = Stage1.fit(tr, model_features()).logit(te)
        out.append(pd.Series(1 / (1 + np.exp(-z)), index=te["game_id"].to_numpy()))
    return pd.concat(out)


def report(
    seasons: tuple[int, ...] = SEASONS, out_name: str = "representation"
) -> None:
    from scipy.stats import spearmanr

    from nbapredicts.backtest import classification_metrics as cm

    feats, _ = load([])
    y_of = feats.set_index("game_id")["home_win"]
    runs: dict = {}
    for f in sorted(PREDS.glob("*.pkl")):
        name, season, seed = f.stem.rsplit("_", 2)
        if int(season) not in seasons:
            continue
        runs.setdefault(name, {}).setdefault(int(seed), []).append(pd.read_pickle(f))
    frames = {
        name: {s: pd.concat(v).set_index("game_id") for s, v in seeds.items()}
        for name, seeds in runs.items()
        if all(len(v) == len(seasons) for v in seeds.values())
    }
    nn = pd.concat([nn_distance(feats, s) for s in seasons])
    lin = logistic(feats, seasons)
    if all(m in frames for m in MIX):
        seeds = sorted(set.intersection(*(set(frames[m]) for m in MIX)))
        frames["mix:" + "+".join(MIX)] = {
            s: pd.concat([frames[m][s][["p"]] for m in MIX], axis=1)
            .mean(axis=1)
            .to_frame("p")
            .join(frames[MIX[0]][s][["var", "novelty", "var_shifted"]])
            for s in seeds
        }
    out: dict = {"seeds": {}, "arms": {}}
    for group, base in (("features_only", "base"), ("stack", "stack_base")):
        if base not in frames:
            continue
        b = frames[base]
        idx = b[min(b)].index
        y = y_of.loc[idx].to_numpy()
        # Noise floor: the same model, different seeds.
        seeds = sorted(b)
        floor = [
            paired(y, b[s]["p"].loc[idx].to_numpy(), b[t]["p"].loc[idx].to_numpy())[0]
            for i, s in enumerate(seeds)
            for t in seeds[i + 1 :]
        ]
        out["seeds"][group] = {
            "pairs": floor,
            "floor": round(max(map(abs, floor), default=0), 5),
        }
        b_avg = np.mean([b[s]["p"].loc[idx].to_numpy() for s in seeds], axis=0)
        for name, fr in frames.items():
            if (group == "stack") != name.startswith("stack_"):
                continue
            common = sorted(set(fr) & set(b))
            p_avg = np.mean([fr[s]["p"].loc[idx].to_numpy() for s in common], axis=0)
            b_c = np.mean([b[s]["p"].loc[idx].to_numpy() for s in common], axis=0)
            per_seed = [
                paired(
                    y, fr[s]["p"].loc[idx].to_numpy(), b[s]["p"].loc[idx].to_numpy()
                )[0]
                for s in common
            ]
            season_of = feats.set_index("game_id")["season"].loc[idx].to_numpy()
            row = {
                "seeds": common,
                "brier": round(cm(y, p_avg)["brier"], 5),
                "log_loss": round(cm(y, p_avg)["log_loss"], 5),
                "accuracy": round(cm(y, p_avg)["accuracy"], 4),
                "ece": round(cm(y, p_avg)["ece"], 4),
                "minus_base_by_seed": per_seed,
                "minus_base_seed_avg": paired(y, p_avg, b_c),
                "minus_logistic_seed_avg": paired(y, p_avg, lin.loc[idx].to_numpy()),
                "minus_base_by_season": {
                    int(s): paired(
                        y[season_of == s], p_avg[season_of == s], b_c[season_of == s]
                    )[0]
                    for s in seasons
                },
            }
            # Uncertainty: calibration by novelty tercile, and distance awareness.
            s0 = fr[common[0]].loc[idx]
            terc = pd.qcut(
                s0["novelty"].rank(method="first"), 3, labels=False
            ).to_numpy()
            row["by_novelty"] = [
                {
                    k: round(v, 4)
                    for k, v in cm(y[terc == t], s0["p"].to_numpy()[terc == t]).items()
                    if k in ("accuracy", "brier", "ece")
                }
                for t in range(3)
            ]
            row["var_vs_nn_distance_spearman"] = round(
                float(spearmanr(s0["var"], nn.loc[idx]).statistic), 3
            )
            row["shifted_auroc"] = round(
                auroc(s0["var"].to_numpy(), s0["var_shifted"].to_numpy()), 3
            )
            out["arms"][name] = row
            print(
                json.dumps(
                    {
                        name: {
                            k: row[k]
                            for k in (
                                "brier",
                                "minus_base_seed_avg",
                                "minus_base_by_seed",
                                "var_vs_nn_distance_spearman",
                                "shifted_auroc",
                            )
                        }
                    }
                ),
                flush=True,
            )
        print(json.dumps({group: out["seeds"][group]}), flush=True)
        out.setdefault("base_brier", {})[group] = round(cm(y, b_avg)["brier"], 5)
        out.setdefault("logistic_brier", {})[group] = round(
            cm(y, lin.loc[idx].to_numpy())["brier"], 5
        )
    (ROOT / f"docs/results/{out_name}.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step", choices=["build", "build_seq", "check", "run", "report"]
    )
    parser.add_argument("--arms", nargs="+", default=["base"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seasons", type=int, nargs="+", default=list(SEASONS))
    parser.add_argument("--out", default="representation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.step == "build":
        build()
    elif args.step == "build_seq":
        build_seq()
    elif args.step == "check":
        check()
    elif args.step == "run":
        run(args.arms, args.seeds, tuple(args.seasons))
    elif args.step == "report":
        report(tuple(args.seasons), args.out)


if __name__ == "__main__":
    main()
