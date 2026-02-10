#!/usr/bin/env python3
"""
S&P 500 Analyzer — Incremental Data Update Script

Fetches new S&P 500 price data, total return data, and CPI data,
appends to existing JSON files, and rebuilds the combined HTML.

Data sources:
  - S&P 500 prices (^GSPC) — Yahoo Finance via yfinance
  - S&P 500 total return (^SP500TR) — Yahoo Finance via yfinance
  - CPI (CPIAUCSL) — FRED API

Usage:
  python scripts/update_data.py                  # uses FRED_API_KEY env var
  python scripts/update_data.py --fred-key XXXX  # explicit key
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PARTS_DIR = REPO_ROOT / "parts"
OUTPUT_FILE = REPO_ROOT / "sp500-analyzer.html"

PRICE_FILE = DATA_DIR / "sp500_price.json"
TR_FILE = DATA_DIR / "sp500_tr.json"
CPI_FILE = DATA_DIR / "cpi.json"

PART1_FILE = PARTS_DIR / "part1.html"
PART2_FILE = PARTS_DIR / "part2.html"

# The "mid" fragments that go between data sections
MID1 = ';\n// === EMBEDDED TOTAL RETURN DATA: [epochDays, trValue] ===\nvar RAW_TR_DATA ='
MID2 = ';\n// === EMBEDDED CPI DATA: {"YYYY-MM": cpiValue, ...} ===\nvar RAW_CPI_DATA ='

# Epoch day 0 = 1970-01-01
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def date_to_epoch_days(dt):
    """Convert a datetime to epoch days (integer)."""
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        delta = dt - EPOCH
    else:
        delta = dt - EPOCH.replace(tzinfo=None)
    return int(delta.days)


def epoch_days_to_date(ed):
    """Convert epoch days to a datetime."""
    return EPOCH + timedelta(days=ed)


# ---------------------------------------------------------------------------
# Load existing data
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    """Save JSON compactly (no spaces, matching the original format)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Fetch S&P 500 price data from Yahoo Finance
# ---------------------------------------------------------------------------
def fetch_new_prices(last_epoch_day):
    """Fetch S&P 500 daily closes after the last stored date."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    # Start one day after the last stored date
    start_date = epoch_days_to_date(last_epoch_day + 1)
    end_date = datetime.now(timezone.utc) + timedelta(days=1)

    if start_date >= end_date:
        print("Price data is already up to date.")
        return []

    print(f"Fetching ^GSPC prices from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    ticker = yf.Ticker("^GSPC")
    hist = ticker.history(start=start_date.strftime("%Y-%m-%d"),
                          end=end_date.strftime("%Y-%m-%d"),
                          auto_adjust=True)

    if hist.empty:
        print("No new price data available.")
        return []

    new_entries = []
    for idx, row in hist.iterrows():
        dt = idx.to_pydatetime()
        ed = date_to_epoch_days(dt)
        close = round(row["Close"], 2)
        if ed > last_epoch_day and close > 0 and not math.isnan(close):
            new_entries.append([ed, close])

    print(f"  Got {len(new_entries)} new price entries.")
    return new_entries


# ---------------------------------------------------------------------------
# Fetch S&P 500 Total Return data from Yahoo Finance
# ---------------------------------------------------------------------------
def fetch_new_tr(existing_tr, last_epoch_day):
    """
    Fetch ^SP500TR data after the last stored date.
    Scale new values to match our existing combined TR index.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    # We need some overlap to compute the scaling ratio.
    # Fetch from 5 trading days before the last stored date.
    overlap_date = epoch_days_to_date(last_epoch_day - 7)
    end_date = datetime.now(timezone.utc) + timedelta(days=1)

    print(f"Fetching ^SP500TR from {overlap_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    ticker = yf.Ticker("^SP500TR")
    hist = ticker.history(start=overlap_date.strftime("%Y-%m-%d"),
                          end=end_date.strftime("%Y-%m-%d"),
                          auto_adjust=True)

    if hist.empty:
        print("No new TR data available.")
        return []

    # Build a lookup of existing TR data by epoch day (last few entries)
    existing_lookup = {}
    for entry in existing_tr[-20:]:  # last 20 entries for overlap matching
        existing_lookup[entry[0]] = entry[1]

    # Find the scaling ratio from overlapping dates
    ratios = []
    yahoo_data = {}
    for idx, row in hist.iterrows():
        dt = idx.to_pydatetime()
        ed = date_to_epoch_days(dt)
        close = row["Close"]
        if math.isnan(close) or close <= 0:
            continue
        yahoo_data[ed] = close
        if ed in existing_lookup:
            ratios.append(existing_lookup[ed] / close)

    if not ratios:
        print("WARNING: No overlapping TR dates found for scaling. Skipping TR update.")
        return []

    # Use the average ratio for scaling
    scale = sum(ratios) / len(ratios)
    print(f"  TR scaling ratio: {scale:.6f} (from {len(ratios)} overlapping dates)")

    # Append only new entries (after the last stored date)
    new_entries = []
    for ed in sorted(yahoo_data.keys()):
        if ed > last_epoch_day:
            scaled_val = round(yahoo_data[ed] * scale, 2)
            new_entries.append([ed, scaled_val])

    print(f"  Got {len(new_entries)} new TR entries.")
    return new_entries


# ---------------------------------------------------------------------------
# Fetch CPI data from FRED
# ---------------------------------------------------------------------------
def fetch_cpi(fred_api_key, existing_cpi):
    """
    Fetch CPI data (CPIAUCSL) from FRED.
    CPI is small enough to fetch the full series and merge.
    """
    if not fred_api_key:
        print("WARNING: No FRED API key provided. Skipping CPI update.")
        return existing_cpi, False

    try:
        import urllib.request
        import urllib.error
    except ImportError:
        pass

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=CPIAUCSL&api_key={fred_api_key}"
        f"&file_type=json&observation_start=1928-01-01"
    )

    print("Fetching CPI data from FRED...")
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "SP500Analyzer/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"WARNING: Failed to fetch CPI from FRED: {e}")
        return existing_cpi, False

    observations = data.get("observations", [])
    if not observations:
        print("No CPI observations returned.")
        return existing_cpi, False

    new_cpi = dict(existing_cpi)  # copy existing
    added = 0
    for obs in observations:
        date_str = obs["date"]  # "YYYY-MM-DD"
        value = obs["value"]
        if value == ".":
            continue
        key = date_str[:7]  # "YYYY-MM"
        val = round(float(value), 3)
        if key not in new_cpi or new_cpi[key] != val:
            new_cpi[key] = val
            added += 1

    print(f"  CPI: {len(new_cpi)} total months ({added} new/updated entries).")
    changed = added > 0
    return new_cpi, changed


# ---------------------------------------------------------------------------
# Assemble the HTML
# ---------------------------------------------------------------------------
def assemble_html(price_data, tr_data, cpi_data):
    """Combine parts + data into the final HTML file."""
    part1 = PART1_FILE.read_text(encoding="utf-8")
    part2 = PART2_FILE.read_text(encoding="utf-8")

    price_json = json.dumps(price_data, separators=(",", ":"))
    tr_json = json.dumps(tr_data, separators=(",", ":"))
    cpi_json = json.dumps(cpi_data, separators=(",", ":"))

    combined = part1 + price_json + MID1 + tr_json + MID2 + cpi_json + part2

    OUTPUT_FILE.write_text(combined, encoding="utf-8")
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"Assembled {OUTPUT_FILE.name}: {size_kb:.0f} KB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Update S&P 500 Analyzer data")
    parser.add_argument("--fred-key", default=os.environ.get("FRED_API_KEY", ""),
                        help="FRED API key (defaults to FRED_API_KEY env var)")
    parser.add_argument("--assemble-only", action="store_true",
                        help="Just rebuild HTML from existing data files (no fetching)")
    args = parser.parse_args()

    # Load existing data
    print("Loading existing data...")
    price_data = load_json(PRICE_FILE)
    tr_data = load_json(TR_FILE)
    cpi_data = load_json(CPI_FILE)

    print(f"  Prices: {len(price_data)} entries, last epoch day: {price_data[-1][0]}")
    print(f"  TR: {len(tr_data)} entries, last epoch day: {tr_data[-1][0]}")
    print(f"  CPI: {len(cpi_data)} months")

    data_changed = False

    if not args.assemble_only:
        # --- Fetch new price data ---
        last_price_day = price_data[-1][0]
        new_prices = fetch_new_prices(last_price_day)
        if new_prices:
            price_data.extend(new_prices)
            save_json(PRICE_FILE, price_data)
            print(f"  Updated {PRICE_FILE.name}: {len(price_data)} total entries")
            data_changed = True

        # --- Fetch new TR data ---
        last_tr_day = tr_data[-1][0]
        new_tr = fetch_new_tr(tr_data, last_tr_day)
        if new_tr:
            tr_data.extend(new_tr)
            save_json(TR_FILE, tr_data)
            print(f"  Updated {TR_FILE.name}: {len(tr_data)} total entries")
            data_changed = True

        # --- Fetch CPI data ---
        cpi_data, cpi_changed = fetch_cpi(args.fred_key, cpi_data)
        if cpi_changed:
            # Sort CPI by key for consistent output
            sorted_cpi = dict(sorted(cpi_data.items()))
            save_json(CPI_FILE, sorted_cpi)
            cpi_data = sorted_cpi
            print(f"  Updated {CPI_FILE.name}: {len(cpi_data)} months")
            data_changed = True

    # --- Assemble HTML ---
    print("\nAssembling HTML...")
    assemble_html(price_data, tr_data, cpi_data)

    if data_changed:
        print("\nData was updated. New data has been saved.")
    else:
        if not args.assemble_only:
            print("\nNo new data found. Everything is up to date.")
        else:
            print("\nHTML reassembled from existing data.")

    # Return exit code based on whether data changed (useful for CI)
    return 0 if data_changed else 2


if __name__ == "__main__":
    sys.exit(main())
