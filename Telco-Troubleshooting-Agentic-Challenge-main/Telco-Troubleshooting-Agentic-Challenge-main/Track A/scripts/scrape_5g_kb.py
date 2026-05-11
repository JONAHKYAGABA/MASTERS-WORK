"""
scripts/scrape_5g_kb.py

Tier-1 RAG scraper for 5G RAN troubleshooting knowledge. Downloads the
ShareTechNote 5G pages that cover the exact KPIs, events, and parameters
that appear in this challenge's scenario data:

    SS-RSRP / SS-SINR semantics
    A2 / A3 / A5 handover events
    PDCCH / CCE / symbol allocation
    Beamforming geometry, mainlobe, tilt
    Neighbour cell list, PCI conflict
    Drive test interpretation
    RACH / RRC reestablishment

Saves one .txt per page in knowledge/raw/ with a SOURCE: header.

Run on the Linux box (after `unset HF_HUB_OFFLINE` if it was set
earlier — though scraping itself doesn't go through huggingface).

Usage:
    pip install requests trafilatura
    unset HF_HUB_OFFLINE
    python scripts/scrape_5g_kb.py
    # outputs: knowledge/raw/doc_*.txt   (one per URL that succeeded)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    import requests
    import trafilatura
except ImportError:
    sys.stderr.write(
        "Missing deps. Install with:\n"
        "  pip install requests trafilatura\n"
    )
    sys.exit(1)


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
RAW_DIR = PROJECT_DIR / "knowledge" / "raw"


# Tier 1 — ShareTechNote pages that map directly to KPIs/events in scenario data.
TIER_1_URLS = [
    # KPI semantics — what RSRP/SINR/etc actually measure
    "https://www.sharetechnote.com/html/5G/5G_SS_RSRP.html",
    "https://www.sharetechnote.com/html/5G/5G_SS_SINR.html",
    "https://www.sharetechnote.com/html/5G/5G_SS_RSRQ.html",
    "https://www.sharetechnote.com/html/5G/5G_CSIRS.html",
    "https://www.sharetechnote.com/html/5G/5G_CQI.html",
    "https://www.sharetechnote.com/html/5G/5G_MCS_TBS.html",
    "https://www.sharetechnote.com/html/5G/5G_BLER.html",
    # Mobility events — A2/A3/A5 trigger conditions, ping-pong
    "https://www.sharetechnote.com/html/5G/5G_HandOver.html",
    "https://www.sharetechnote.com/html/5G/5G_MeasurementEvent.html",
    "https://www.sharetechnote.com/html/5G/5G_RRCConnectionReestablishment.html",
    "https://www.sharetechnote.com/html/5G/5G_RandomAccess.html",
    # PHY scheduling — PDCCH symbols, CCE, RB allocation
    "https://www.sharetechnote.com/html/5G/5G_PDCCH.html",
    "https://www.sharetechnote.com/html/5G/5G_PDSCH.html",
    "https://www.sharetechnote.com/html/5G/5G_ResourceAllocationType.html",
    # Antenna / beamforming — mainlobe, tilt, azimuth, gain
    "https://www.sharetechnote.com/html/5G/5G_BeamForming.html",
    "https://www.sharetechnote.com/html/5G/5G_MIMO.html",
    "https://www.sharetechnote.com/html/5G/5G_BeamManagement.html",
    # Neighbour relations / PCI / coverage planning
    "https://www.sharetechnote.com/html/5G/5G_NeighbourCellList.html",
    "https://www.sharetechnote.com/html/5G/5G_PCI.html",
    # Drive test methodology
    "https://www.sharetechnote.com/html/5G/5G_DriveTest.html",
    # Power control
    "https://www.sharetechnote.com/html/5G/5G_PowerControl.html",
]

# Tier 2 — Huawei parameter docs. Mostly behind-login but try a few public URLs.
# Safe to fail; don't block the run.
TIER_2_URLS = [
    "https://support.huawei.com/enterprise/en/doc/EDOC1100141408",
    "https://support.huawei.com/enterprise/en/doc/EDOC1100141413",
]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_one(url: str, idx: int) -> bool:
    try:
        r = requests.get(url, timeout=30, headers=HEADERS)
        if r.status_code != 200:
            print(f"  FAIL {idx:02d} status={r.status_code}: {url}")
            return False
        text = trafilatura.extract(r.text, include_comments=False, include_tables=True)
        if not text or len(text) < 300:
            print(f"  THIN {idx:02d} ({len(text or '')} chars): {url}")
            return False
        path = RAW_DIR / f"doc_{idx:03d}.txt"
        path.write_text(f"SOURCE: {url}\n\n{text}", encoding="utf-8")
        print(f"  OK   {idx:02d} ({len(text):>6d} chars): {url}")
        return True
    except Exception as e:
        print(f"  ERR  {idx:02d}: {url}  -> {e}")
        return False


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urls = TIER_1_URLS + TIER_2_URLS
    print(f"[scrape] {len(urls)} URLs -> {RAW_DIR}")
    n_ok = 0
    for i, url in enumerate(urls):
        ok = scrape_one(url, i)
        n_ok += int(ok)
        time.sleep(2)
    print()
    print(f"[scrape] DONE: {n_ok}/{len(urls)} pages saved to {RAW_DIR}")
    if n_ok < 5:
        print("[scrape] WARNING: very few pages succeeded. Check network / proxy.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
