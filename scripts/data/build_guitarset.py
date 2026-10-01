"""
Katman 1 — Veri hattı: GuitarSet -> (CQT, etiketler) önbelleği.

Yaptığı işler ve NEDEN önemli oldukları:

1) SADECE 'solo' kayıtları alır (monofonik). 'comp' kayıtları akor içerir = çok
   sesli; onları Katman 6'ya bırakıyoruz. Senin "temiz solo ile başla" tercihin.

2) OYUNCUYA GÖRE böler (train/val). GuitarSet'te 6 gitarist var. Aynı gitaristin
   çalışını hem eğitimde hem testte görürsek model o kişinin tarzını "ezberler" ve
   sahte yüksek skor verir (data leakage). Bir gitaristi tamamen test'e ayırırsak
   modelin GERÇEKTEN genelleyip genellemediğini ölçeriz. Bu, ileride kandırmayan
   sonuçlar için kritik bir future-proofing.

3) Her kaydı bir kez işleyip .npz olarak ÖNBELLEĞE alır. CQT hesabı yavaştır;
   Katman 2'de modeli defalarca eğitirken aynı sesi tekrar tekrar işlemek istemeyiz.

Çalıştırma:
    python -m scripts.data.build_guitarset     # tüm solo kayıtları işler, data/cache/ altına yazar
"""

import os
import glob
import numpy as np

from gtab.data.features import cqt_from_file
from gtab.data.labels import notes_to_rolls
from gtab.data.guitarset import is_solo, player_of, track_to_transcription
from gtab.config import GUITARSET_DATA_HOME
from gtab.paths import CACHE_DIR


def build(val_player: str = "05", solo_only: bool = True):
    import mirdata
    gset = mirdata.initialize("guitarset", data_home=GUITARSET_DATA_HOME)
    tracks = gset.load_tracks()

    selected = [t for t in tracks if (is_solo(t) or not solo_only)]
    print(f"{len(selected)} kayit islenecek (solo_only={solo_only}).")

    for i, tid in enumerate(selected, 1):
        track = tracks[tid]
        # NOT: alan adi mirdata surumune gore degisebilir; hata alirsan dir(track) ile bak.
        cqt = cqt_from_file(track.audio_mic_path)          # (n_frames, n_bins)
        n_frames = cqt.shape[0]

        tr = track_to_transcription(track)
        frame_roll, onset_roll = notes_to_rolls(tr, n_frames)

        split = "val" if player_of(tid) == val_player else "train"
        out_dir = os.path.join(CACHE_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, tid + ".npz"),
            cqt=cqt, frame=frame_roll, onset=onset_roll,
        )
        print(f"[{i}/{len(selected)}] {split:5s} {tid}  cqt={cqt.shape}")

    print("Bitti. Onbellek:", CACHE_DIR)


def load_cached(split: str):
    """Onbellekteki kayitlari tek tek verir: (track_id, cqt, frame_roll, onset_roll)."""
    pattern = os.path.join(CACHE_DIR, split, "*.npz")
    for path in sorted(glob.glob(pattern)):
        d = np.load(path)
        tid = os.path.basename(path)[:-4]
        yield tid, d["cqt"], d["frame"], d["onset"]


if __name__ == "__main__":
    build()
