"""One table of ids for players, teams and games across the local sources.

    uv run python scripts/build_crosswalk.py

Writes data_local/crosswalk/{players,teams,team_history,games}.parquet.

- Players: NBA.com person id <-> ESPN athlete id <-> Wikidata <->
  Basketball-Reference. Joined on ids (sportsdataverse's season crosswalks,
  Wikidata's id properties); a name match (normalized name, unique on both
  sides) is used only for ESPN athletes no id source covers, and flagged.
- Teams: ESPN <-> NBA.com team ids, abbreviations, Polymarket slug codes,
  FiveThirtyEight franchise names; plus each NBA.com team id's past names.
- Games: NBA.com game id <-> ESPN event id (sportsdataverse, else date and
  teams), Polymarket event slug (date and teams), FiveThirtyEight game id
  (date and both final scores).

Each id must map one to one; the run prints unmatched and duplicate counts.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "data_local"
OUT = DL / "crosswalk"
KAG = DL / "kaggle/eoinamoore__historical-nba-data-and-player-box-scores/files"
SDV = DL / "sportsdataverse/nba_crosswalk"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nbapredicts.collect import Store  # noqa: E402
from nbapredicts.injuries import normalize_name  # noqa: E402


def sdv(kind: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(SDV / f"nba_{kind}_crosswalk_*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def as_id(s: pd.Series) -> pd.Series:
    """Ids as strings without a trailing '.0' (floats from parquet / csv)."""
    out = s.astype("string").str.replace(r"\.0$", "", regex=True)
    return out.where(out.notna() & (out != "") & (out != "nan"))


def one_to_one(df: pd.DataFrame, a: str, b: str, label: str) -> pd.DataFrame:
    pairs = df.dropna(subset=[a, b])
    dup_a = pairs[a].duplicated(keep=False).sum()
    dup_b = pairs[b].duplicated(keep=False).sum()
    print(
        f"  {label}: {len(pairs)} pairs; {dup_a} rows share an {a}, {dup_b} share a {b}"
    )
    return pairs


# --------------------------------------------------------------------------
# Players
# --------------------------------------------------------------------------


def players() -> pd.DataFrame:
    kag = pd.read_csv(KAG / "Players.csv", low_memory=False)
    kag["nba_id"] = as_id(kag["personId"])
    kag["name"] = (
        kag["firstName"].fillna("") + " " + kag["lastName"].fillna("")
    ).str.strip()

    wd = pd.DataFrame(json.loads((DL / "history/wikidata/players.json").read_text()))
    wd["wikidata"] = wd["p"].str.rsplit("/", n=1).str[1]
    wd = wd.groupby("wikidata").first().reset_index()
    for c in ("nba_id", "espn_id"):
        wd[c] = as_id(wd[c])

    cw = sdv("player")
    cw["nba_id"], cw["espn_id"] = (
        as_id(cw["nba_player_id"]),
        as_id(cw["espn_athlete_id"]),
    )
    # The pairing seen in most seasons wins.
    pairs = (
        cw.dropna(subset=["nba_id", "espn_id"])
        .groupby(["nba_id", "espn_id"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    pairs = pairs.drop_duplicates("nba_id").drop_duplicates("espn_id")

    ids = pd.DataFrame(
        {
            "nba_id": pd.concat([kag["nba_id"], wd["nba_id"], pairs["nba_id"]])
            .dropna()
            .unique()
        }
    )
    out = ids.merge(pairs[["nba_id", "espn_id"]], on="nba_id", how="left")
    out["espn_source"] = (
        out["espn_id"].notna().map({True: "sportsdataverse", False: None})
    )
    by_nba = wd.dropna(subset=["nba_id"]).drop_duplicates("nba_id").set_index("nba_id")
    fill = out["espn_id"].isna() & out["nba_id"].isin(by_nba.index)
    out.loc[fill, "espn_id"] = out.loc[fill, "nba_id"].map(by_nba["espn_id"])
    out.loc[fill & out["espn_id"].notna(), "espn_source"] = "wikidata"
    out["wikidata"] = out["nba_id"].map(by_nba["wikidata"])
    out["bref_id"] = out["nba_id"].map(by_nba["bref_id"])
    k = kag.drop_duplicates("nba_id").set_index("nba_id")
    out["name"] = (
        out["nba_id"].map(k["name"]).fillna(out["nba_id"].map(by_nba["pLabel"]))
    )
    out["birth_date"] = out["nba_id"].map(k["birthDate"])
    out["from_year"], out["to_year"] = (
        out["nba_id"].map(k["fromYear"]),
        out["nba_id"].map(k["toYear"]),
    )

    # ESPN athletes in our box scores that no id source covers: unique name match.
    pb = (
        Store(ROOT / "data")
        .load("player_box")[["athlete_id", "name"]]
        .drop_duplicates("athlete_id")
    )
    pb["espn_id"] = as_id(pb["athlete_id"])
    missing = pb[~pb["espn_id"].isin(out["espn_id"].dropna())].copy()
    missing["key"] = missing["name"].map(normalize_name)
    free = out[out["espn_id"].isna()].copy()
    free["key"] = free["name"].fillna("").map(normalize_name)
    uniq_free = free[~free["key"].duplicated(keep=False)].set_index("key")["nba_id"]
    uniq_missing = missing[~missing["key"].duplicated(keep=False)]
    hit = uniq_missing[uniq_missing["key"].isin(uniq_free.index)]
    for _, r in hit.iterrows():
        row = out["nba_id"] == uniq_free[r["key"]]
        out.loc[row, "espn_id"] = r["espn_id"]
        out.loc[row, "espn_source"] = "name match"
    left = len(missing) - len(hit)
    print(
        f"players: {len(out)} NBA.com ids; ESPN id for {out['espn_id'].notna().sum()}"
    )
    print(f"  by source: {out['espn_source'].value_counts().to_dict()}")
    print(
        f"  our ESPN box-score athletes: {len(pb)}; {left} still without an NBA.com id"
    )
    print(
        f"  Wikidata for {out['wikidata'].notna().sum()}, "
        f"Basketball-Reference for {out['bref_id'].notna().sum()}"
    )
    one_to_one(out, "nba_id", "espn_id", "nba_id <-> espn_id")
    return out


# --------------------------------------------------------------------------
# Teams
# --------------------------------------------------------------------------


def teams() -> tuple[pd.DataFrame, pd.DataFrame]:
    cw = sdv("team")
    cw = cw[cw["season"] == cw["season"].max()].copy()
    out = pd.DataFrame(
        {
            "espn_team_id": as_id(cw["espn_team_id"]),
            "nba_team_id": as_id(cw["nba_team_id"]),
            "espn_abbr": cw["espn_abbreviation"],
            "nba_abbr": cw["nba_team_abbreviation"],
            "name": cw["espn_display_name"],
            "short": cw["espn_short_name"],
        }
    )
    import polymarket_archive as pa

    events = [pa.read_json(p) for p in sorted((pa.OUT / "events").glob("*.json.gz"))]
    ours = json.loads((ROOT / "data/teams.json").read_text())
    codes = pa.team_codes(events, ours)
    out["polymarket_code"] = out["espn_team_id"].map(codes)
    out["fte_fran_id"] = out["short"].replace(
        {"76ers": "Sixers", "Trail Blazers": "Trailblazers"}
    )
    fte = pd.read_csv(
        DL / "history/fivethirtyeight/nba-elo/nbaallelo.csv", usecols=["fran_id"]
    )
    print(
        f"teams: {len(out)}; "
        f"Polymarket code for {out['polymarket_code'].notna().sum()}, "
        f"FiveThirtyEight franchise for {out['fte_fran_id'].isin(fte['fran_id']).sum()}"
    )
    one_to_one(out, "espn_team_id", "nba_team_id", "espn <-> nba team")
    hist = pd.read_csv(KAG / "TeamHistories.csv")
    hist["nba_team_id"] = as_id(hist["teamId"])
    return out, hist


# --------------------------------------------------------------------------
# Games
# --------------------------------------------------------------------------


def games(team_map: pd.DataFrame) -> pd.DataFrame:
    kg = pd.read_csv(KAG / "Games.csv", low_memory=False)
    when = pd.to_datetime(kg["gameDateTimeEst"], errors="coerce", format="mixed")
    out = pd.DataFrame(
        {
            "nba_game_id": as_id(kg["gameId"]).str.zfill(10),
            "date": when.dt.strftime("%Y-%m-%d"),
            "home_nba_team_id": as_id(kg["hometeamId"]),
            "away_nba_team_id": as_id(kg["awayteamId"]),
            "home_pts": kg["homeScore"],
            "away_pts": kg["awayScore"],
        }
    ).drop_duplicates("nba_game_id")

    # ESPN: sportsdataverse's schedule crosswalk, then date and teams.
    sc = sdv("schedule")
    sc = (
        pd.DataFrame(
            {
                "nba_game_id": as_id(sc["nba_game_id"]).str.zfill(10),
                "espn_game_id": as_id(sc["espn_game_id"]),
            }
        )
        .dropna()
        .drop_duplicates()
    )
    out = out.merge(sc.drop_duplicates("nba_game_id"), on="nba_game_id", how="left")
    g = Store(ROOT / "data").load("games")
    e2n = team_map.set_index("espn_team_id")["nba_team_id"]
    ours = pd.DataFrame(
        {
            "espn_game_id": as_id(g["game_id"]),
            "date": pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d"),
            "home_nba_team_id": as_id(g["home_id"]).map(e2n),
            "away_nba_team_id": as_id(g["away_id"]).map(e2n),
        }
    )
    have = set(out["espn_game_id"].dropna())
    todo = ours[~ours["espn_game_id"].isin(have)]
    keyed = out[out["espn_game_id"].isna()].merge(
        todo,
        on=["date", "home_nba_team_id", "away_nba_team_id"],
        how="inner",
        suffixes=("", "_m"),
    )
    fill = (
        keyed.drop_duplicates("nba_game_id")
        .drop_duplicates("espn_game_id_m")
        .set_index("nba_game_id")["espn_game_id_m"]
    )
    out.loc[out["nba_game_id"].isin(fill.index), "espn_game_id"] = out[
        "nba_game_id"
    ].map(fill)
    print(
        f"games: {len(out)} NBA.com games, {out['date'].min()} to {out['date'].max()}"
    )
    print(
        f"  ESPN id for {out['espn_game_id'].notna().sum()}; "
        f"our {len(ours)} ESPN games, "
        f"{(~ours['espn_game_id'].isin(out['espn_game_id'])).sum()} unmatched"
    )

    # Polymarket: event slugs -> date and teams.
    import polymarket_archive as pa

    code2nba = team_map.dropna(subset=["polymarket_code"]).set_index("polymarket_code")[
        "nba_team_id"
    ]
    rows = []
    for p in sorted((pa.OUT / "events").glob("*.json.gz")):
        m = pa.GAME_SLUG.match(p.name.removesuffix(".json.gz"))
        if m:
            rows.append(
                {
                    "polymarket_slug": m.group(0),
                    "a": code2nba.get(m.group(1)),
                    "b": code2nba.get(m.group(2)),
                    "day": m.group(3),
                }
            )
    pm = pd.DataFrame(rows)
    out["polymarket_slug"] = pd.NA
    for a, b in (
        ("away_nba_team_id", "home_nba_team_id"),
        ("home_nba_team_id", "away_nba_team_id"),
    ):
        for shift in (0, 1, -1):  # the slug's date can be the UTC date
            day = (pd.to_datetime(out["date"]) + pd.Timedelta(days=shift)).dt.strftime(
                "%Y-%m-%d"
            )
            key = out.assign(day=day)[["nba_game_id", a, b, "day"]].rename(
                columns={a: "a", b: "b"}
            )
            hit = key.merge(pm, on=["a", "b", "day"]).drop_duplicates("nba_game_id")
            hit = hit[~hit["polymarket_slug"].isin(out["polymarket_slug"].dropna())]
            todo = out["polymarket_slug"].isna() & out["nba_game_id"].isin(
                hit["nba_game_id"]
            )
            out.loc[todo, "polymarket_slug"] = out.loc[todo, "nba_game_id"].map(
                hit.set_index("nba_game_id")["polymarket_slug"]
            )
    print(
        f"  Polymarket event for {out['polymarket_slug'].notna().sum()} "
        f"of {len(pm)} events"
    )

    # FiveThirtyEight: date and both final scores (home rows only).
    fte = pd.read_csv(
        DL / "history/fivethirtyeight/nba-elo/nbaallelo.csv",
        usecols=["game_id", "date_game", "game_location", "pts", "opp_pts", "lg_id"],
    )
    fte = fte[(fte["game_location"] == "H") & (fte["lg_id"] == "NBA")]
    fte["date"] = pd.to_datetime(fte["date_game"]).dt.strftime("%Y-%m-%d")
    key = fte.rename(
        columns={"pts": "home_pts", "opp_pts": "away_pts", "game_id": "fte_game_id"}
    )[["fte_game_id", "date", "home_pts", "away_pts"]]
    hit = out.merge(key, on=["date", "home_pts", "away_pts"], how="inner")
    hit = hit[
        ~hit["nba_game_id"].duplicated(keep=False)
        & ~hit["fte_game_id"].duplicated(keep=False)
    ]
    out["fte_game_id"] = out["nba_game_id"].map(
        hit.set_index("nba_game_id")["fte_game_id"]
    )
    print(
        f"  FiveThirtyEight id for {out['fte_game_id'].notna().sum()} "
        f"of {len(key)} home-team rows"
    )
    for a in ("espn_game_id", "polymarket_slug", "fte_game_id"):
        one_to_one(out, "nba_game_id", a, f"nba_game_id <-> {a}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = players()
    t, hist = teams()
    g = games(t)
    p.to_parquet(OUT / "players.parquet", index=False)
    t.to_parquet(OUT / "teams.parquet", index=False)
    hist.to_parquet(OUT / "team_history.parquet", index=False)
    g.to_parquet(OUT / "games.parquet", index=False)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
