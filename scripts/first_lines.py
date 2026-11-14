"""Each game's first prediction made with a betting line posted, rebuilt from
the prediction log's git history.

The log keeps a game's last pick before tip-off and overwrites the rest; the
daily job now also keeps the first pick made with a line (``first_line_*``,
pipeline.with_first_line). For games logged before that, every version of the
log that the job committed is still in git, so the first one with a line
gives the same fields.

    uv run python scripts/first_lines.py --season 2027
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import pandas as pd

from nbapredicts.collect import ID_COLS, Store
from nbapredicts.pipeline import FIRST_LINE

ROOT = Path(__file__).resolve().parents[1]


def versions(path: str) -> list[pd.DataFrame]:
    """Every committed version of ``path``, oldest first."""
    shas = subprocess.run(
        ["git", "log", "--format=%H", "--reverse", "--", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    out = []
    for sha in shas:
        text = subprocess.run(
            ["git", "show", f"{sha}:{path}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout
        if text.strip():
            out.append(pd.read_csv(io.StringIO(text), dtype={c: str for c in ID_COLS}))
    return out


def first_lines(season: int) -> pd.DataFrame:
    seen: dict[str, dict] = {}
    for v in versions(f"data/predictions/{season}.csv"):
        if "p_market" not in v:
            continue
        v = v[v["p_market"].notna()]
        if "backfilled" in v:
            v = v[v["backfilled"].astype(str) != "True"]
        before = pd.to_datetime(v["predicted_at"], utc=True) < pd.to_datetime(
            v["start"], utc=True
        )
        for r in v[before].itertuples(index=False):
            seen.setdefault(
                r.game_id,
                {
                    "game_id": r.game_id,
                    "first_line_at": r.predicted_at,
                    "first_line_p_home": r.p_home,
                    "first_line_p_market": r.p_market,
                },
            )
    return pd.DataFrame(list(seen.values()), columns=["game_id", *FIRST_LINE])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    store = Store(ROOT / "data")
    preds = store.load("predictions")
    found = first_lines(args.season).set_index("game_id")
    for c in FIRST_LINE:
        if c not in preds:
            preds[c] = pd.Series(pd.NA, index=preds.index, dtype=object)
    todo = preds["first_line_at"].isna() & preds["game_id"].isin(found.index)
    for c in FIRST_LINE:
        preds.loc[todo, c] = preds.loc[todo, "game_id"].map(found[c])
    store.save("predictions", preds)
    print(json.dumps({"games_found": len(found), "filled": int(todo.sum())}))


if __name__ == "__main__":
    main()
