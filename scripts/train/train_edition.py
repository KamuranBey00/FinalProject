"""
Katman 3.13 Adım 2 — "Klasik edisyon önerisi" modelini GAPS partisyonlarından öğren (sembolik, ses yok, dakikalar).

Veri: build_gaps_tab etiketleri (data/cache/gaps_tab), YALNIZ gaps_fit kayıtları (gaps_val/gaps_test görülmez).
Çıktı: checkpoints/edition.npz (konum + parmak ağırlıkları; transitions.npz'ye dokunulmaz).

Çalıştırma:
    python -m scripts.train.train_edition
"""

import argparse
import os

import numpy as np

from gtab.data.gaps import GAPS_TAB_DIR, gaps_files
from gtab.decoding.edition import ChordPositionModel, FingerModel, as_tuples, label_sequence, save_edition
from gtab.paths import ckpt_path


def labeled(files):
    out = [os.path.join(GAPS_TAB_DIR, os.path.basename(f)) for f in files]
    return [f for f in out if os.path.exists(f)]


def uniform_nll(mask):
    return float(np.log(mask.sum(1)).mean())


def main(out):
    fit_f, val_f = (labeled(x) for x in gaps_files("gaps_train"))
    fit = [as_tuples(label_sequence(f)) for f in fit_f]
    val = [as_tuples(label_sequence(f)) for f in val_f]
    print(f"gaps_fit {len(fit)} kayit ({sum(len(s) for s in fit)} nota) | gaps_val {len(val)} kayit")
    pos, fing = ChordPositionModel(), FingerModel()
    for name, m in (("konum", pos), ("parmak", fing)):
        _, mk, _ = m._build(fit); _, mv, _ = m._build(val)
        print(f"\n[{name}] ornek: fit {len(mk)} / val {len(mv)} | ozellik {m.dim}")
        m.fit(fit)
        print(f"[{name}] NLL fit {m.nll(fit):.3f} (uniform {uniform_nll(mk):.3f}) | "
              f"val {m.nll(val):.3f} (uniform {uniform_nll(mv):.3f})")
    save_edition(ckpt_path(out), pos, fing)

    # açıklanabilirlik: öğrenilen maliyetin okunur kısmı
    S = pos.instrument.num_strings
    print("\nakor ici d-tel agirligi (perde artan sirada; + = tercih):",
          " ".join(f"{d:+d}:{w:+.1f}" for d, w in zip(range(-(S - 1), S), pos.w[pos._cds])))
    dh = list(range(-5, 6)) + ["uzak"]
    for nm, sl in (("sirali", fing._dh), ("akor ici", fing._cdh)):
        print(f"el pozisyonu degisimi dh ({nm}):", " ".join(f"{d}:{w:+.1f}" for d, w in zip(dh, fing.w[sl])))
    print("parmak 1-4 taban egilimi:", " ".join(f"{g}:{fing.w[g]:+.2f}" for g in range(1, 5)),
          f"| ayni parmakla kayma {fing.w[fing._same_shift]:+.2f} | barre (akor) {fing.w[fing._csame_barre]:+.2f}")
    print(f"-> {ckpt_path(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="edition.npz")
    main(ap.parse_args().out)
