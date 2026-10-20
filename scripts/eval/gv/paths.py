"""Where the guitar-sound test data lives. All scripts in this folder run from the repo root.

Recordings and sample sets whose license allows redistribution are versioned under eval_data/; everything else
(YouTube audio, classtab, non-redistributable sample libraries) is fetched into data/raw/gv/ (git-ignored) by
scripts/data/fetch_gv_data.py, so the repo never re-publishes third-party material it may not share.
"""
from pathlib import Path

EVAL_DATA = Path("eval_data")
COMMONS = EVAL_DATA / "commons"
ICHIYANAGI = EVAL_DATA / "ichiyanagi" / "Classical-Guitar_Ichiyanagi-v1_03.sf3"
VIEWER_FONT = Path("scripts/app/soundfont/nylon_guitar.sf3.js")

RAW = Path("data/raw/gv")
CLASSTAB = RAW / "classtab"
YOUTUBE = RAW / "youtube"
LIBRARIES = RAW / "libraries"

GAPS_TAB = Path("data/cache/gaps_tab_rec")
GAPS_TEST = Path("data/cache/gaps_test")
RESULTS = Path("docs/results/gv")
