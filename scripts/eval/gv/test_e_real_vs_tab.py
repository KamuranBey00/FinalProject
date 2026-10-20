"""Test E/F: real performances (Wikimedia Commons, YouTube) vs the classtab tablature of the same piece.
Both TAB modes are scored: "ses" (heard position) and "edisyon" (edition suggestion), see classtab.compare.
Run from the repo root: python -m scripts.eval.gv.test_e_real_vs_tab docs/results/gv/pairs_commons2.tsv <out_csv>
(pairs.tsv: audio_path <TAB> tab_txt_path, repo-relative)
"""
import csv
import sys
from pathlib import Path

from gtab.data.features import cqt_from_file
from gtab.pipeline import analyze, load_models
from gtab.utils import get_device
from scripts.eval.gv import classtab


def main(pairs_file: Path, out: Path) -> None:
    m = load_models(get_device())
    pairs = [l.split("\t") for l in pairs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = None
        for audio, tab_path in pairs:
            tab = classtab.events(Path(tab_path))
            cqt = cqt_from_file(audio)
            for mode in ("ses", "edisyon"):
                an = analyze(cqt, m, mode)
                row = {"audio": Path(audio).name, "tab": Path(tab_path).stem, "mode": mode,
                       **classtab.compare(tab, an["notes"], an["string"])}
                w = w or csv.DictWriter(f, fieldnames=list(row))
                if f.tell() == 0:
                    w.writeheader()
                w.writerow(row)
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
