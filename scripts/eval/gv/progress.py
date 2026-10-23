"""Progress bars for long background chains (training, comparison, calibration).

A chain is described by data/raw/gv/progress_plan.json (git-ignored), a list of stages:
  {"label": "...", "glob": "checkpoints/x_ep*.pt", "total": 15}    count = files matching the glob
  {"label": "...", "lines": "data/raw/gv/progress_compare.txt", "total": 30}    count = lines in the file
  {"label": "...", "file": "checkpoints/x.cal.json", "minutes": 66}    done when the file exists; until then the bar
      shows elapsed / expected time since the previous stage finished (capped at 99 %)
      + "log": path, "chains": [w1, w2, ...]: read the real percentage from the log of eval_pitch's own progress bar
      ("[kalibrasyon <chain>] ... % NN", one chain after another, weights = share of the total time); immune to
      sleep / pauses, unlike elapsed time
The remaining time of a stage is estimated from the modification times of what was counted so far.
Run from the repo root:
  python -m scripts.eval.gv.progress            # live view, refreshes every 5 s (Ctrl+C to stop)
  python -m scripts.eval.gv.progress --once     # print once
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from scripts.eval.gv.paths import RAW

PLAN = RAW / "progress_plan.json"
WIDTH = 24


CHAIN = re.compile(r"\[kalibrasyon ([^\]]+)\] \[[# ]*\] %\s*(\d+)")


def log_percent(path: Path, weights: list[float]) -> float:
    # chains run one after another; a new chain starts whenever the label changes
    if not path.exists():
        return 0.0
    hits = CHAIN.findall(path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n"))
    if not hits:
        return 0.0
    k = sum(a[0] != b[0] for a, b in zip(hits, hits[1:]))
    k = min(k, len(weights) - 1)
    return 100 * (sum(weights[:k]) + weights[k] * int(hits[-1][1]) / 100) / sum(weights)


def counted(stage: dict, started: float | None) -> tuple[int, list[float]]:
    # -> (units done, modification times of the units / of the line file)
    if "file" in stage:
        p = Path(stage["file"])
        if p.exists():
            # creation time: the marker may be a log that keeps growing (its mtime would move the next stage's start)
            return stage["minutes"], [p.stat().st_ctime if os.name == "nt" else p.stat().st_mtime]
        if "log" in stage:
            pct = log_percent(Path(stage["log"]), stage["chains"])
            return min(int(pct * stage["minutes"] / 100), stage["minutes"] - 1), []
        return (min(int((time.time() - started) / 60), stage["minutes"] - 1), []) if started else (0, [])
    if "glob" in stage:
        files = sorted(Path(".").glob(stage["glob"]))
        return len(files), sorted(f.stat().st_mtime for f in files)
    p = Path(stage["lines"])
    if not p.exists():
        return 0, []
    return len(p.read_text(encoding="utf-8").splitlines()), [p.stat().st_mtime]


def eta(done: int, total: int, times: list[float], started: float) -> str:
    if done == 0 or done >= total:
        return ""
    per = (times[-1] - started) / done            # seconds per unit since the stage started
    left = per * (total - done) - (time.time() - times[-1])
    return f"  ~{max(left, 0) / 60:.0f} dk kaldı"


RATES = RAW / "progress_rates.json"
WINDOW_S, MIN_SPAN_S, MAX_GAP_S = 15 * 60, 3 * 60, 5 * 60


def observed_eta(label: str, pct: float) -> str:
    # remaining time from the speed seen over the last 15 min (a gap > 5 min = sleep: the window restarts there)
    try:
        rates = json.loads(RATES.read_text(encoding="utf-8")) if RATES.exists() else {}
    except (OSError, ValueError):
        rates = {}
    now = time.time()
    s = [x for x in rates.get(label, []) if now - x[0] <= WINDOW_S and x[1] <= pct] + [[now, pct]]
    gaps = [i for i in range(1, len(s)) if s[i][0] - s[i - 1][0] > MAX_GAP_S]
    s = s[gaps[-1]:] if gaps else s
    rates[label] = s[-200:]
    try:
        RATES.write_text(json.dumps(rates), encoding="utf-8")
    except OSError:
        pass                                       # two viewers writing at once: the next refresh catches up
    (t0, p0), (t1, p1) = s[0], s[-1]
    if t1 - t0 < MIN_SPAN_S or p1 <= p0:
        return "  (hız ölçülüyor)"
    return f"  ~{(100 - p1) / ((p1 - p0) / (t1 - t0)) / 60:.0f} dk kaldı"


def render(plan: list[dict], chain_start: float) -> list[str]:
    # the first stage starts when the plan was written (= when the chain was launched)
    lines, prev_end = [], chain_start
    for st in plan:
        done, times = counted(st, prev_end)
        total = st.get("total", st.get("minutes"))
        frac = min(done / total, 1.0)
        bar = "█" * round(WIDTH * frac) + "░" * (WIDTH - round(WIDTH * frac))
        state = "bitti" if done >= total else ("sırada" if prev_end is None else "")
        started = prev_end or (times[0] if times else time.time())
        unit = " dk" if "file" in st else ""
        left = ""
        if "file" in st and not state:
            left = (observed_eta(st["label"], log_percent(Path(st["log"]), st["chains"])) if "log" in st
                    else f"  ~{total - done} dk kaldı (tahmini)")
        lines.append(f"{st['label']:<30} [{bar}] {100 * frac:3.0f}%  {done}/{total}{unit}"
                     + (f"  {state}" if state else left or eta(done, total, times, started)))
        prev_end = times[-1] if done >= total and times else None
    return lines


def main(once: bool) -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console default (cp1254) has no block characters
    while True:
        plan = json.loads(PLAN.read_text(encoding="utf-8")) if PLAN.exists() else []
        out = render(plan, PLAN.stat().st_mtime if PLAN.exists() else time.time()) or ["(aktif plan yok)"]
        if once:
            print("\n".join(out))
            return
        os.system("cls" if os.name == "nt" else "clear")    # old Windows consoles ignore ANSI clear codes
        print(time.strftime("%H:%M:%S") + "  (Ctrl+C ile kapat)\n" + "\n".join(out), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    main(ap.parse_args().once)
