# Stooq Data Download & Setup

**Goal:** Download 10+ years hourly OHLCV data for US + UK markets.

**Expected directory structure after download:**
```
data/stooq_raw/
  data/
    hourly/
      us/
        aapl.txt
        msft.txt
        spy.txt
        ... (all S&P500 + Nasdaq tickers)
      gb/
        gsk.txt (GSK.L in live_sim maps to gsk.txt in gb/)
        ... (all FTSE100 tickers)
```

**File format:** Date,Open,High,Low,Close,Volume (CSV, no header row)
- Date format: YYYY-MM-DD (daily files) or YYYY-MM-DD HH:MM (hourly)
- Volume: integer or decimal

---

## Option 1: Manual Download (Recommended for Initial Setup)

1. Visit https://stooq.com/db/h_d.php
2. Scroll to "Historical Data Downloads"
3. Download:
   - `daily_us.zip` — extract to data/stooq_raw/
   - `hourly_us.zip` — extract to data/stooq_raw/
   - `daily_gb.zip` — extract to data/stooq_raw/
   - `hourly_gb.zip` — extract to data/stooq_raw/ (or daily_gb if hourly unavailable)

4. Unzip all to data/stooq_raw/ (should auto-create subdirs)
5. Run validation below

---

## Option 2: Automated Download (Python Script)

If Stooq provides direct URLs, this script can fetch them:

```python
import os
import zipfile
import requests

stooq_base = "data/stooq_raw"
os.makedirs(stooq_base, exist_ok=True)

# URLs — update if Stooq changes their structure
urls = [
    "https://stooq.com/db/h_d/full/hourly_us.zip",
    "https://stooq.com/db/h_d/full/hourly_gb.zip",
]

for url in urls:
    print(f"Downloading {url}...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        zip_path = os.path.join(stooq_base, os.path.basename(url))
        with open(zip_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(stooq_base)
        os.remove(zip_path)
        print(f"  ✓ Extracted")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
```

---

## Validation

Run once files are in place:

```bash
# List hourly/us structure
ls -la data/stooq_raw/data/hourly/us/ | head -20

# Check file format (should be Date,Open,High,Low,Close,Volume)
head -5 data/stooq_raw/data/hourly/us/aapl.txt

# Test fetch_hourly_stooq() in Python
python -c "
from Strategy_Auto_Trader.quant_hmm.data_cache import fetch_hourly_stooq
df = fetch_hourly_stooq('AAPL')
print(f'AAPL: {len(df)} bars, {df.index[0]} to {df.index[-1]}')

df = fetch_hourly_stooq('GSK.L')
print(f'GSK.L: {len(df)} bars, {df.index[0]} to {df.index[-1]}')
"
```

---

## Hardcoded Constraints

- Hard-fail if ticker not found (no fallback to yfinance)
- Market routing: .L suffix → gb/, else us/
- No re-sampling (hourly files must be hourly already)
- Date format must be parseable by pd.read_csv

---

## Troubleshooting

**File not found error:**
```
FileNotFoundError: Stooq file not found: .../data/stooq_raw/data/hourly/us/aapl.txt
```
→ Ensure aapl.txt exists in the correct subdirectory. Case-sensitive on Linux.

**Missing columns error:**
```
ValueError: Missing columns in .../aapl.txt. Expected [...], got [...]
```
→ CSV must have exactly: Date, Open, High, Low, Close, Volume. No extra columns.

**Empty file error:**
```
ValueError: Stooq file empty: .../aapl.txt
```
→ File exists but has no data rows (only header or blank).

---

## Next Steps

1. Download and unzip Stooq files to data/stooq_raw/
2. Run validation above
3. Wire fetcher into live_sim.py (Phase 3 of plan)
4. Run top-K backtest with --data-source stooq flag

