"""
Module 01 — Data Download
=========================
Downloads OHLCV data for VN100 constituent stocks.

Source: vnstock (TCBS / SSI)

Output: data/raw/{TICKER}.csv
Columns: date, open, high, low, close, volume
"""

import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download")

# ── VN100 Universe ────────────────────────────────────────────────────────────
VN100_TICKERS = [
# Banking
    "VCB", "BID", "CTG", "MBB", "TCB", "ACB", "STB", "HDB", "VPB", "LPB",
    "OCB", "EIB", "MSB", "SHB", "TPB", "SSB", "VIB", "NAB", "EVF",
    
    # Insurance / Securities
    "BVH", "SSI", "VCI", "HCM", "VND", "FTS", "VIX", "CTS", "BSI", "DSE",
    
    # Real Estate & Industrial Parks
    "VHM", "VIC", "VRE", "DXG", "KDH", "PDR", "DIG", "HDG", "TCH", "SJS",
    "BCM", "CII", "DXS", "HDC", "KBC", "KOS", "NLG", "SIP", "SZC", "VPI",
    
    # Consumer / Retail / Agriculture
    "VNM", "MSN", "MWG", "PNJ", "SAB", "KDC", "VHC", "ANV", "DBC", "DGW",
    "FRT", "HAG", "PAN", "SBT", "TLG",
    
    # Industry / Steel / Construction / Materials
    "HPG", "NKG", "HSG", "VGC", "CTD", "PC1", "VCG", "HHV", "HT1", "PTB",
    
    # Oil & Gas / Utilities
    "GAS", "PLX", "POW", "NT2", "REE", "PPC", "BWE", "GEE", "GEX", "PVD",
    "PVT",
    
    # Technology & Telecommunications
    "FPT", "CMG", "CTR", "VTP",
    
    # Logistics / Aviation
    "VJC", "GMD", "SCS", "VSC",
    
    # Chemicals / Healthcare / Other
    "DPM", "DCM", "BMP", "DGC", "GVR", "PHR", "IMP"
]

VN100_DEFAULT = [
#    "HPG", "VCB", "VHM", "VIC", "VNM", "MSN", "TCB", "BID", "CTG", "MBB",
#    "ACB", "FPT", "MWG", "SSI", "GAS", "PLX", "VRE", "HDB", "VPB", "STB",
    # Banking
    "VCB", "BID", "CTG", "MBB", "TCB", "ACB", "STB", "HDB", "VPB", "LPB",
    "OCB", "EIB", "MSB", "SHB", "TPB", "SSB", "VIB", "NAB", "EVF",
    
    # Insurance / Securities
    "BVH", "SSI", "VCI", "HCM", "VND", "FTS", "VIX", "CTS", "BSI", "DSE",
    
    # Real Estate & Industrial Parks
    "VHM", "VIC", "VRE", "DXG", "KDH", "PDR", "DIG", "HDG", "TCH", "SJS",
    "BCM", "CII", "DXS", "HDC", "KBC", "KOS", "NLG", "SIP", "SZC", "VPI",
    
    # Consumer / Retail / Agriculture
    "VNM", "MSN", "MWG", "PNJ", "SAB", "KDC", "VHC", "ANV", "DBC", "DGW",
    "FRT", "HAG", "PAN", "SBT", "TLG",
    
    # Industry / Steel / Construction / Materials
    "HPG", "NKG", "HSG", "VGC", "CTD", "PC1", "VCG", "HHV", "HT1", "PTB",
    
    # Oil & Gas / Utilities
    "GAS", "PLX", "POW", "NT2", "REE", "PPC", "BWE", "GEE", "GEX", "PVD",
    "PVT",
    
    # Technology & Telecommunications
    "FPT", "CMG", "CTR", "VTP",
    
    # Logistics / Aviation
    "VJC", "GMD", "SCS", "VSC",
    
    # Chemicals / Healthcare / Other
    "DPM", "DCM", "BMP", "DGC", "GVR", "PHR", "IMP"
]


# ── Download via vnstock ──────────────────────────────────────────────────────
def _download_vnstock(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        from vnstock import Vnstock

        df = (
            Vnstock()
            .stock(symbol=ticker, source="KBS")
            .quote.history(start=start, end=end)
        )

        if df is None or df.empty:
            log.warning(f"  {ticker:6s}  vnstock returned empty data")
            return None

        # v3 returns: time, open, high, low, close, volume (already normalised)
        df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
        df = df.sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        log.warning(f"  {ticker:6s}  vnstock error: {e}")
        return None


# ── Main download logic ───────────────────────────────────────────────────────
def download_ticker(
    ticker: str,
    start: str,
    end: str,
    force: bool = False,
) -> pd.DataFrame | None:
    out_path = RAW / f"{ticker}.csv"

    if not force and out_path.exists():
        log.info(f"  {ticker:6s}  cached → {out_path.name}")
        return pd.read_csv(out_path, dtype={"date": str})

    log.info(f"  {ticker:6s}  downloading {start} → {end} …")
    df = _download_vnstock(ticker, start, end)

    if df is None or len(df) < 60:
        log.error(f"  {ticker:6s}  insufficient data — skipping")
        return None

    df.to_csv(out_path, index=False)
    log.info(f"  {ticker:6s}  saved {len(df)} rows → {out_path.name}")
    return df


def download_all(
    tickers: list[str],
    start: str,
    end: str,
    force: bool = False,
    delay: float = 3.2,
) -> dict[str, pd.DataFrame]:
    results = {}
    failed  = []

    for i, t in enumerate(tickers, 1):
        log.info(f"[{i:02d}/{len(tickers)}] {t}")
        df = download_ticker(t, start, end, force=force)
        if df is not None:
            results[t] = df
        else:
            failed.append(t)
        time.sleep(delay)

    log.info(f"\n✓ Downloaded : {len(results)}/{len(tickers)} tickers")
    if failed:
        log.warning(f"✗ Failed     : {', '.join(failed)}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VN100 Data Downloader")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Tickers to download (default: VN100_DEFAULT)")
    parser.add_argument("--all", action="store_true",
                        help="Download full VN100 universe")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (default: 5 years ago)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cache exists")
    parser.add_argument("--delay", type=float, default=3.2,
                        help="Seconds between requests (default: 3.2)")
    args = parser.parse_args()

    end_date   = args.end   or datetime.today().strftime("%Y-%m-%d")
    start_date = args.start or "2000-01-01"

    tickers = VN100_TICKERS if args.all else (args.tickers or VN100_DEFAULT)

    log.info(f"=== VN100 Download: {len(tickers)} tickers  {start_date} → {end_date} ===\n")
    download_all(tickers, start_date, end_date, force=args.force, delay=args.delay)