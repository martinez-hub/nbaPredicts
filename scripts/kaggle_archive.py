"""NBA datasets from Kaggle, kept local.

    uv run python scripts/kaggle_archive.py

- eoinamoore/historical-nba-data-and-player-box-scores (CC0, updated
  nightly): every game and player box score since 1946-47, play-by-play and
  advanced stats from about 1996, player bios.
- wyattowalsh/basketball (CC BY-SA 4.0): a SQLite database of NBA.com data
  back to 1946-47.
- ghopkins/nba-injuries-2010-2018 (CC0): injuries 2010-11 to 2019-20.

Kaggle downloads need the owner's account: the API token in
~/.kaggle/access_token (or KAGGLE_API_TOKEN) is sent to kaggle.com only, as
a bearer token; the older ~/.kaggle/kaggle.json (username and key) also
works. Each dataset's licence is checked before downloading, and a dataset
is fetched again only when Kaggle shows a newer version.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_local" / "kaggle"
API = "https://www.kaggle.com/api/v1"
DATASETS = {
    "eoinamoore/historical-nba-data-and-player-box-scores": "CC0",
    "wyattowalsh/basketball": "CC BY-SA",
    "ghopkins/nba-injuries-2010-2018": "CC0",
}

log = logging.getLogger("kaggle_archive")


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "research-client/1.0"
    token = os.environ.get("KAGGLE_API_TOKEN")
    path = Path.home() / ".kaggle" / "access_token"
    if not token and path.exists():
        token = path.read_text().strip()
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
        return s
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    if legacy.exists():
        creds = json.loads(legacy.read_text())
        s.auth = (creds["username"], creds["key"])
        return s
    sys.exit("No Kaggle token: ~/.kaggle/access_token or ~/.kaggle/kaggle.json")


def archive(s: requests.Session, ref: str, licence: str) -> None:
    info = s.get(f"{API}/datasets/view/{ref}", timeout=60)
    info.raise_for_status()
    meta = info.json()
    name = meta.get("licenseName") or ""
    if not name.startswith(licence):
        log.warning("%s: licence is %r, not %s: skipped", ref, name, licence)
        return
    folder = OUT / ref.replace("/", "__")
    stamp_path = folder / "kaggle_version.json"
    stamp = {"lastUpdated": meta.get("lastUpdated"), "licence": name}
    if stamp_path.exists() and json.loads(stamp_path.read_text()) == stamp:
        log.info("%s: up to date (%s)", ref, stamp["lastUpdated"])
        return
    folder.mkdir(parents=True, exist_ok=True)
    zip_path = folder / "download.zip"
    with s.get(f"{API}/datasets/download/{ref}", stream=True, timeout=600) as r:
        if r.status_code in (401, 403):
            sys.exit(f"{ref}: Kaggle refused the token ({r.status_code})")
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(folder / "files")
        n = len(z.namelist())
    zip_path.unlink()
    stamp_path.write_text(json.dumps(stamp))
    log.info("%s: %d files (%s, updated %s)", ref, n, name, stamp["lastUpdated"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = session()
    for ref, licence in DATASETS.items():
        archive(s, ref, licence)


if __name__ == "__main__":
    main()
