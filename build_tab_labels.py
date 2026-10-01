"""
Katman 3 — Önbelleği tel etiketleriyle zenginleştir.

ÖNEMLİ: CQT'yi YENİDEN HESAPLAMIYORUZ. Katman 1'de önbelleğe aldığımız cqt'yi
okuyup, sadece yeni 'tab' etiketini ekleyip .npz'yi tekrar yazıyoruz. CQT yavaş;
tel etiketi hızlı. Böylece dakikalar içinde biter.

Her .npz dosya adı zaten track_id; mirdata'dan o track'in tel-bazlı notalarını
çekip, önbellekteki kare sayısına (T) hizalı (T, 6) etiket üretiyoruz.

Çalıştırma:
    python build_tab_labels.py
"""

import glob
import os
import numpy as np

from gtab.tab_labels import build_tab_targets
from gtab.instrument import STANDARD_6
from gtab.config import GUITARSET_DATA_HOME

CACHE_DIR = "data/cache"


def main():
    import mirdata
    gset = mirdata.initialize("guitarset", data_home=GUITARSET_DATA_HOME)
    tracks = gset.load_tracks()

    paths = sorted(glob.glob(os.path.join(CACHE_DIR, "*", "*.npz")))
    print(f"{len(paths)} önbellek dosyası güncellenecek.")

    for i, path in enumerate(paths, 1):
        tid = os.path.basename(path)[:-4]
        # Diziyi TAMAMEN belleğe al ve dosya tutucuyu KAPAT, sonra yaz.
        # Aksi halde Windows'ta acik npz handle'i varken ayni yola yazmak
        # PermissionError verir ya da sessizce dosyayi bozar.
        with np.load(path) as d:
            cqt = d["cqt"].copy(); frame = d["frame"].copy(); onset = d["onset"].copy()
        n_frames = cqt.shape[0]

        notes_dict = tracks[tid].notes                       # {tel: NoteData}
        tab = build_tab_targets(notes_dict, n_frames, STANDARD_6)   # (T, 6)

        np.savez_compressed(path, cqt=cqt, frame=frame, onset=onset, tab=tab)
        if i % 20 == 0 or i == len(paths):
            active = int((tab > 0).sum())
            print(f"[{i}/{len(paths)}] {tid}  tab={tab.shape}  aktif_tel_kare={active}")

    print("Bitti. Artik her .npz 'tab' etiketini iceriyor.")


if __name__ == "__main__":
    main()
