"""NBA history beyond the game data: every game since 1946, every player,
team, arena, award and season, and the league's late-game officiating
reviews. Kept in data_local/history/.

    uv run python scripts/history_archive.py fivethirtyeight  # CC BY 4.0
    uv run python scripts/history_archive.py wikidata         # CC0
    uv run python scripts/history_archive.py wikidata --refresh  # weekly
    uv run python scripts/history_archive.py l2m              # NBA, local only

- FiveThirtyEight's NBA folders (github.com/fivethirtyeight/data): Elo for
  every game 1946-2015 with scores, RAPTOR, win probabilities, draft model.
- Wikidata: players (bio, size, position, draft, IDs that join to NBA.com,
  ESPN and Basketball-Reference), their teams by year, awards, franchises,
  arenas (location, capacity, opening) and seasons (champions).
- Last Two Minute reports, 2015-2019: the PDFs the officiating archive page
  links on the league's file server (later reports sit behind a page that
  refuses plain clients, so they are not fetched).

Requests go one at a time with a plain client name; files already on disk
are skipped, except that ``--refresh`` queries Wikidata again and replaces
its tables (new players, awards and seasons; the weekly refresh does this).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_local" / "history"
USER_AGENT = "research-client/1.0"

log = logging.getLogger("history_archive")


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


# --------------------------------------------------------------------------
# FiveThirtyEight
# --------------------------------------------------------------------------

FTE_TREE = "https://api.github.com/repos/fivethirtyeight/data/git/trees/master"
FTE_RAW = "https://raw.githubusercontent.com/fivethirtyeight/data/master/{path}"


def fivethirtyeight(s: requests.Session) -> None:
    tree = s.get(FTE_TREE, params={"recursive": 1}, timeout=120).json()["tree"]
    paths = [
        t["path"]
        for t in tree
        if t["type"] == "blob" and re.match(r"^nba-[^/]+/", t["path"])
    ]
    for path in paths:
        dest = OUT / "fivethirtyeight" / path
        if dest.exists():
            continue
        resp = s.get(FTE_RAW.format(path=path), timeout=300)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        time.sleep(0.3)
    log.info("fivethirtyeight: %d files", len(paths))


# --------------------------------------------------------------------------
# Wikidata
# --------------------------------------------------------------------------

SPARQL = "https://query.wikidata.org/sparql"
NBA = "wd:Q155223"
PLAYER = "wd:Q3665646"
COACH = "wd:Q5137571"
# NBA players: the league on their item, or an NBA.com player id (many
# current players carry only the id). The other player queries time out
# with the list inline, so they join a copy the server builds once (a named
# subquery); the players query keeps it inline, since with the named copy
# its answer gets cut off.
PLAYER_LIST = (
    "SELECT DISTINCT ?p WHERE { "
    f"{{ ?p wdt:P118 {NBA}; wdt:P106 {PLAYER}. }} UNION {{ ?p wdt:P3647 []. }}"
    " }"
)
NBA_PLAYERS = f"WITH {{ {PLAYER_LIST} }} AS %players"

QUERIES = {
    # One row per player and value; the server times out if these are
    # grouped in one query, so positions and countries come separately.
    "players": f"""
SELECT ?p ?pLabel ?birth ?death ?birthplaceLabel ?height ?weight ?nba_id
  ?bref_id ?espn_id ?article WHERE {{
  {{ {PLAYER_LIST} }}
  OPTIONAL {{ ?p wdt:P569 ?birth. }}
  OPTIONAL {{ ?p wdt:P570 ?death. }}
  OPTIONAL {{ ?p wdt:P19 ?birthplace. }}
  OPTIONAL {{ ?p wdt:P2048 ?height. }}
  OPTIONAL {{ ?p wdt:P2067 ?weight. }}
  OPTIONAL {{ ?p wdt:P3647 ?nba_id. }}
  OPTIONAL {{ ?p wdt:P2685 ?bref_id. }}
  OPTIONAL {{ ?p wdt:P3685 ?espn_id. }}
  OPTIONAL {{ ?article schema:about ?p;
    schema:isPartOf <https://en.wikipedia.org/>. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}""",
    "player_positions": f"""
SELECT ?p ?pos ?posLabel
{NBA_PLAYERS}
WHERE {{
  INCLUDE %players
  ?p wdt:P413 ?pos.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}""",
    "player_countries": f"""
SELECT ?p ?country ?countryLabel
{NBA_PLAYERS}
WHERE {{
  INCLUDE %players
  ?p wdt:P27 ?country.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}""",
    # Each spell with a team, with its start and end where recorded.
    "player_teams": f"""
SELECT ?p ?team ?teamLabel ?start ?end
{NBA_PLAYERS}
WHERE {{
  INCLUDE %players
  ?p p:P54 ?st.
  ?st ps:P54 ?team.
  OPTIONAL {{ ?st pq:P580 ?start. }}
  OPTIONAL {{ ?st pq:P582 ?end. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}""",
    "player_draft": f"""
SELECT ?p ?team ?teamLabel ?when ?pick ?round
{NBA_PLAYERS}
WHERE {{
  INCLUDE %players
  ?p p:P647 ?st.
  ?st ps:P647 ?team.
  OPTIONAL {{ ?st pq:P585 ?when. }}
  OPTIONAL {{ ?st pq:P1545 ?pick. }}
  OPTIONAL {{ ?st pq:P5021 ?round. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}""",
    "player_awards": f"""
SELECT ?p ?award ?awardLabel ?when
{NBA_PLAYERS}
WHERE {{
  INCLUDE %players
  ?p p:P166 ?st.
  ?st ps:P166 ?award.
  OPTIONAL {{ ?st pq:P585 ?when. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}""",
    "coaches": f"""
SELECT DISTINCT ?c ?cLabel ?birth ?team ?teamLabel ?start ?end WHERE {{
  ?c wdt:P106 {COACH}; p:P6087 ?st.
  ?st ps:P6087 ?team. ?team wdt:P118 {NBA}.
  OPTIONAL {{ ?c wdt:P569 ?birth. }}
  OPTIONAL {{ ?st pq:P580 ?start. }}
  OPTIONAL {{ ?st pq:P582 ?end. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}""",
    # Franchises and former names, with home city and arena.
    "teams": f"""
SELECT ?t ?tLabel ?inception ?dissolved ?cityLabel ?venue ?venueLabel
  ?replaced_by ?replaced_byLabel WHERE {{
  ?t wdt:P118 {NBA}; wdt:P31/wdt:P279* wd:Q13393265.
  OPTIONAL {{ ?t wdt:P571 ?inception. }}
  OPTIONAL {{ ?t wdt:P576 ?dissolved. }}
  OPTIONAL {{ ?t wdt:P159 ?city. }}
  OPTIONAL {{ ?t wdt:P115 ?venue. }}
  OPTIONAL {{ ?t wdt:P1366 ?replaced_by. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}""",
    "arenas": f"""
SELECT DISTINCT ?v ?vLabel ?coord ?capacity ?opened ?closed ?cityLabel WHERE {{
  ?t wdt:P118 {NBA}; wdt:P115 ?v.
  OPTIONAL {{ ?v wdt:P625 ?coord. }}
  OPTIONAL {{ ?v wdt:P1083 ?capacity. }}
  OPTIONAL {{ ?v wdt:P1619 ?opened. }}
  OPTIONAL {{ ?v wdt:P3999 ?closed. }}
  OPTIONAL {{ ?v wdt:P131 ?city. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}""",
    # Seasons, with the champion and the regular season's dates.
    "seasons": """
SELECT ?s ?sLabel ?start ?end ?winner ?winnerLabel WHERE {
  ?s wdt:P3450 wd:Q155223.
  OPTIONAL { ?s wdt:P580 ?start. }
  OPTIONAL { ?s wdt:P582 ?end. }
  OPTIONAL { ?s wdt:P1346 ?winner. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
}


def wikidata_rows(s: requests.Session, query: str) -> list[dict] | None:
    """One query's rows, or None if no whole answer came back in 4 tries."""
    for attempt in range(4):
        wait = 30 * (attempt + 1)
        try:
            resp = s.get(
                SPARQL,
                params={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=120,
            )
        except requests.RequestException as err:
            log.warning("wikidata: %s (try %d)", err, attempt + 1)
            time.sleep(wait)
            continue
        if resp.status_code in (401, 403):
            log.error("wikidata refused (%d): stopping", resp.status_code)
            sys.exit(2)
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(float(resp.headers.get("Retry-After") or wait))
            continue
        resp.raise_for_status()
        try:
            bindings = resp.json()["results"]["bindings"]
        except ValueError:
            # The service can cut a large answer off mid-stream and still
            # answer 200; the JSON then does not parse.
            log.warning("wikidata: answer cut off (try %d)", attempt + 1)
            time.sleep(wait)
            continue
        return [{k: v["value"] for k, v in b.items()} for b in bindings]
    return None


def wikidata(s: requests.Session, refresh: bool = False) -> None:
    """Save each table; a table that does not come back whole (or comes
    back empty) keeps its last copy, and the run exits 1 so the weekly
    refresh tries again the next morning."""
    failed = []
    for name, query in QUERIES.items():
        dest = OUT / "wikidata" / f"{name}.json"
        if dest.exists() and not refresh:
            continue
        rows = wikidata_rows(s, query)
        if not rows:
            log.warning("wikidata %s: no whole answer; kept what was there", name)
            failed.append(name)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows))
            tmp.rename(dest)
            log.info("wikidata %s: %d rows", name, len(rows))
        time.sleep(5)
    if failed:
        log.error("wikidata: not refreshed: %s", ", ".join(failed))
        sys.exit(1)


# --------------------------------------------------------------------------
# Last Two Minute reports
# --------------------------------------------------------------------------

L2M_PAGE = "https://official.nba.com/nba-officiating-last-two-minute-reports-archive"
L2M_PDF = re.compile(
    r'href="(https://ak-static\.cms\.nba\.com/wp-content/uploads/[^"]+\.pdf)"'
)


def l2m(s: requests.Session) -> None:
    page = s.get(L2M_PAGE, timeout=60)
    page.raise_for_status()
    urls = sorted({u for u in L2M_PDF.findall(page.text) if "L2M" in u})
    folder = OUT / "l2m"
    folder.mkdir(parents=True, exist_ok=True)
    saved = failed = 0
    for url in urls:
        dest = folder / url.rsplit("/", 1)[1]
        if dest.exists():
            continue
        time.sleep(1.0)
        resp = s.get(url, timeout=60)
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            dest.write_bytes(resp.content)
            saved += 1
        else:
            failed += 1
            if saved == 0 and failed >= 25:
                log.error(
                    "first %d requests failed (%d): stopping", failed, resp.status_code
                )
                sys.exit(2)
    log.info("l2m: %d linked PDFs, %d saved, %d failed", len(urls), saved, failed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "steps", nargs="+", choices=["fivethirtyeight", "wikidata", "l2m"]
    )
    parser.add_argument(
        "--refresh", action="store_true", help="query Wikidata again (weekly)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = session()
    for step in args.steps:
        if step == "wikidata":
            wikidata(s, refresh=args.refresh)
        else:
            {"fivethirtyeight": fivethirtyeight, "l2m": l2m}[step](s)


if __name__ == "__main__":
    main()
