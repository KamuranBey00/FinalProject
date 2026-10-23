"""README22 Stage 1 check: compare acoustic checkpoints under the SAME decoding (the active gaps_val selection), so a
difference comes from the model, not from re-calibration. Per checkpoint and split: note P / R / F1, frame F1,
fast-repeat miss (as calibrate_reattack) and the ghost composition (as ghost_audit).
gaps_val = selection, gaps_test = report only. Full re-calibration follows only for the chosen checkpoint.
Appends one line per finished checkpoint to data/raw/gv/progress_compare.txt (git-ignored) for scripts.eval.gv.progress.
--own-cal: each checkpoint with its own calibration (checkpoints/<stem>.cal.json) = the product-level comparison.
Run from the repo root: python -m scripts.eval.gv.compare_ckpts tabcrnn_deep2.pt tabcrnn_g1_try.pt ... [--test] [--own-cal]
"""
import argparse
from collections import Counter

from gtab.evaluation.pitch_eval import load_selection, load_split
from gtab.models.inference import load_model
from gtab.models.stack import active
from gtab.paths import ckpt_path
from gtab.utils import get_device
from scripts.eval.gv.calibrate_reattack import row, run
from scripts.eval.gv.ghost_audit import piece
from scripts.eval.gv.paths import RAW

GROUPS = ("overtone", "fragment", "open string", "displaced")
PROGRESS = RAW / "progress_compare.txt"


def ghost_groups(data: list, sel: tuple) -> Counter:
    c = Counter()
    for item in data:
        for g in piece(item, sel)[0]:
            cat = g["cat"]
            c[next((k for k in GROUPS if k in cat), "other")] += 1
    return c


def main(ckpts: list[str], splits: tuple[str, ...], own_cal: bool) -> None:
    device = get_device()
    shared = load_selection(ckpt_path(active()["cal"]))["gaps_val"]
    for split in splits:
        print(f"== {split} (decoding: {'own calibration' if own_cal else 'active gaps_val selection'})")
        for name in ckpts:
            sel = load_selection(ckpt_path(name.replace(".pt", ".cal.json")))["gaps_val"] if own_cal else shared
            _, t0, o0, dec = sel
            model, _, _ = load_model("crnn", device, name)
            data = load_split(split, model, None, "crnn", device)
            print(row(name, run(data, t0, o0, dec)))
            c = ghost_groups(data, sel)
            print(f"    ghosts {sum(c.values())}: " + ", ".join(f"{k} {c[k]}" for k in GROUPS + ("other",)), flush=True)
            with open(PROGRESS, "a", encoding="utf-8") as f:
                f.write(f"{split} {name}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--test", action="store_true", help="also report gaps_test")
    ap.add_argument("--own-cal", action="store_true")
    a = ap.parse_args()
    main(a.ckpts, ("gaps_val", "gaps_test") if a.test else ("gaps_val",), a.own_cal)
