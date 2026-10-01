"""
Katman 3.7 — Veri genişletme: GuitarSet 'comp' (akor eşliği) kayıtlarını önbelleğe al.

NEDEN: Katman 3.6b teşhisi darboğazın AKUSTİK tel ayrımı olduğunu gösterdi
(oracle notalarda CRNN teli %74 doğru seçiyor). Bugüne kadar sadece 180 'solo'
kayıt kullanıldı; 180 'comp' kaydı diskte hazır ama hiç kullanılmadı. Aynı
gitaristler, aynı gitar, aynı mikrofon -> her telin tınısı çok daha fazla
bağlamda (akorlar dahil) görülür. Tel-başına softmax şeması akorları zaten
destekliyor (bkz. gtab/tab_labels.py) -> model değişmez.

KARŞILAŞTIRILABİLİRLİK: Mevcut data/cache/{train,val} DOKUNULMAZ.
  oyuncu 00-04 comp -> data/cache/train_comp/
  oyuncu 05    comp -> data/cache/val_comp/   (sadece ek rapor; ana val hâlâ 05 solo)
Böylece yeni modeller eski sayılarla (CRNN greedy 0.664) birebir kıyaslanır.

Tek geçişte CQT + frame/onset roll + tel/fret ('tab') etiketi yazılır.

Çalıştırma:
    python -m scripts.data.build_comp
"""

import os
import numpy as np

from gtab.data.features import cqt_from_file
from gtab.data.labels import notes_to_rolls
from gtab.data.tab_labels import build_tab_targets
from gtab.data.guitarset import is_comp, player_of, track_to_transcription
from gtab.core.instrument import STANDARD_6
from gtab.config import GUITARSET_DATA_HOME
from gtab.paths import CACHE_DIR


def build(val_player: str = "05", overwrite: bool = False):
    import mirdata
    gset = mirdata.initialize("guitarset", data_home=GUITARSET_DATA_HOME)
    tracks = gset.load_tracks()

    selected = sorted(t for t in tracks if is_comp(t))
    print(f"{len(selected)} comp kaydi islenecek.")

    for i, tid in enumerate(selected, 1):
        split = "val_comp" if player_of(tid) == val_player else "train_comp"
        out_dir = os.path.join(CACHE_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, tid + ".npz")
        if os.path.exists(out) and not overwrite:
            continue                      # yarıda kalırsa kaldığı yerden devam

        track = tracks[tid]
        cqt = cqt_from_file(track.audio_mic_path)          # solo ile AYNI modalite (mic)
        n_frames = cqt.shape[0]
        frame_roll, onset_roll = notes_to_rolls(track_to_transcription(track), n_frames)
        tab = build_tab_targets(track.notes, n_frames, STANDARD_6)

        np.savez_compressed(out, cqt=cqt, frame=frame_roll, onset=onset_roll, tab=tab)
        print(f"[{i}/{len(selected)}] {split:10s} {tid}  cqt={cqt.shape}  "
              f"aktif_tel_kare={int((tab > 0).sum())}")

    print("Bitti. Onbellek:", CACHE_DIR, "(train_comp / val_comp)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    build(overwrite=ap.parse_args().overwrite)
