import os
import requests
import numpy as np
import pandas as pd
from io import StringIO

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
FIXTURE_DOWNLOAD_URL = "https://fixturedownload.com/download/epl-2025-GMTStandardTime.csv"
PATCH_PATH = "Data/PL2025_patch.csv"

PATCH_COLS = [
    "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG",
    "HS", "AS", "HST", "AST",
    "HC", "AC",
    "HF", "AF",
    "HY", "AY",
    "HR", "AR",
    "PSH", "PSD", "PSA",
    "Avg<2.5", "Avg>2.5",
    "P<2.5", "P>2.5",
]

# fixturedownload.com name -> football-data.co.uk name
TEAM_NAME_MAP = {
    "Man Utd":  "Man United",
    "Spurs":    "Tottenham",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_team_name_map(series):
    return series.replace(TEAM_NAME_MAP)


def _load_patch():
    """Return patch DataFrame, or an empty one with correct columns if absent."""
    if os.path.exists(PATCH_PATH) and os.path.getsize(PATCH_PATH) > 0:
        return pd.read_csv(PATCH_PATH)
    return pd.DataFrame(columns=PATCH_COLS)


def _save_patch(df):
    df.to_csv(PATCH_PATH, index=False)


def _normalize_dates(series):
    return pd.to_datetime(series, dayfirst=True, errors="coerce").dt.normalize()


# ---------------------------------------------------------------------------
# Public data-loading helper (imported by AssistingFunctions.py)
# ---------------------------------------------------------------------------

def load_season_data():
    """
    Load Data/PL2025.csv merged with Data/PL2025_patch.csv.

    Real CSV rows always take precedence: any patch entry whose
    (Date, HomeTeam, AwayTeam) already appears in the real CSV is excluded.

    Returns
    -------
    pd.DataFrame  (same schema as PL2025.csv)
    """
    real = pd.read_csv("Data/PL2025.csv")
    patch = _load_patch()

    if patch.empty:
        return real

    real_keys = set(zip(
        _normalize_dates(real["Date"]),
        real["HomeTeam"],
        real["AwayTeam"],
    ))

    patch_dates = _normalize_dates(patch["Date"])
    active_mask = pd.Series([
        (d, h, a) not in real_keys
        for d, h, a in zip(patch_dates, patch["HomeTeam"], patch["AwayTeam"])
    ])

    active_patch = patch[active_mask.values]
    if active_patch.empty:
        return real

    return pd.concat([real, active_patch], ignore_index=True)


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def download_season_csv():
    """Download the latest season CSV from football-data.co.uk."""
    print("  Downloading season data...            ", end="", flush=True)
    resp = requests.get(FOOTBALL_DATA_URL, timeout=20)
    resp.raise_for_status()

    content = resp.content.decode("utf-8", errors="replace").strip()
    if not content or len(content.splitlines()) < 2:
        raise ValueError("Downloaded season CSV appears to be empty.")

    with open("Data/PL2025.csv", "w", encoding="utf-8") as f:
        f.write(content)

    rows = len(content.splitlines()) - 1
    print(f"✓  ({rows} matches)")
    return rows


def download_schedule():
    """
    Download the latest schedule from fixturedownload.com, apply the
    team-name mapping, and overwrite Data/schedule.csv.
    """
    print("  Downloading schedule...               ", end="", flush=True)
    resp = requests.get(FIXTURE_DOWNLOAD_URL, timeout=20)
    resp.raise_for_status()

    schedule = pd.read_csv(StringIO(resp.content.decode("utf-8", errors="replace")))
    schedule["Home Team"] = _apply_team_name_map(schedule["Home Team"])
    schedule["Away Team"] = _apply_team_name_map(schedule["Away Team"])

    schedule.to_csv("Data/schedule.csv", index=False)
    print(f"✓  ({len(schedule)} fixtures)")
    return schedule


# ---------------------------------------------------------------------------
# Patch management
# ---------------------------------------------------------------------------

def reconcile_patch():
    """
    Remove patch entries whose fixture now appears in the real CSV.

    Returns
    -------
    int  number of entries removed
    """
    patch = _load_patch()
    if patch.empty:
        print("  Reconciling patch...                  ✓  (patch is empty)")
        return 0

    real = pd.read_csv("Data/PL2025.csv")
    real_keys = set(zip(
        _normalize_dates(real["Date"]),
        real["HomeTeam"],
        real["AwayTeam"],
    ))

    patch_dates = _normalize_dates(patch["Date"])
    keep = pd.Series([
        (d, h, a) not in real_keys
        for d, h, a in zip(patch_dates, patch["HomeTeam"], patch["AwayTeam"])
    ])

    cleaned = patch[keep.values]
    removed = len(patch) - len(cleaned)
    _save_patch(cleaned)

    label = "entry" if removed == 1 else "entries"
    remaining = len(cleaned)
    if removed > 0:
        print(f"  Reconciling patch...                  ✓  ({removed} {label} removed, {remaining} remaining)")
    else:
        print(f"  Reconciling patch...                  ✓  ({remaining} remaining, nothing new in real data)")
    return removed


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def find_missing_games():
    """
    Identify fixtures that should have been played (Date < today) but are
    absent from both the real CSV and the patch.

    Returns
    -------
    pd.DataFrame  rows from schedule.csv that are missing, reset-indexed
    """
    schedule = pd.read_csv("Data/schedule.csv")
    schedule["_date"] = _normalize_dates(schedule["Date"])

    today = pd.Timestamp.today().normalize()
    past = schedule[schedule["_date"] < today].copy()

    combined = load_season_data()
    existing_keys = set(zip(
        _normalize_dates(combined["Date"]),
        combined["HomeTeam"],
        combined["AwayTeam"],
    ))

    missing = past[
        ~past.apply(
            lambda r: (r["_date"], r["Home Team"], r["Away Team"]) in existing_keys,
            axis=1,
        )
    ].copy()

    return missing.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def update_pipeline():
    """
    Single entry point for the UpdateData notebook.

    Steps
    -----
    1. Download fresh schedule  -> overwrite Data/schedule.csv
    2. Download fresh season data -> overwrite Data/PL2025.csv
    3. Reconcile patch (prune entries now covered by real CSV)
    4. Detect missing games (played but absent from data)
    5. Display any missing fixtures
    """
    print("\n" + "=" * 52)
    print("   PREMIER LEAGUE DATA UPDATE")
    print("=" * 52 + "\n")

    # --- Downloads ---
    download_schedule()
    download_season_csv()
    print()

    # --- Reconcile ---
    reconcile_patch()
    print()

    # --- Gap check ---
    missing = find_missing_games()

    if missing.empty:
        print("  All fixtures up to date. Ready to run the model.\n")
        return

    print(f"  {len(missing)} missing {'game' if len(missing) == 1 else 'games'} found:\n")
    for i, row in missing.iterrows():
        dt = pd.to_datetime(str(row["Date"]), dayfirst=True)
        print(f"    [{i + 1}]  {row['Home Team']:<18}  vs  {row['Away Team']:<18}  ({dt.strftime('%a %d %b, %H:%M')})")
    print()
