"""
Katman 3.8a — GAPS (Guitar-Aligned Performance Scores) v1.1 indirme.

Kaynak: https://huggingface.co/datasets/xavriley/GAPS  (MIT, ses DAHİL, ~15.3 GB)
Zenodo'daki v1 sürümünde ses yok ve dosya adlarında büyük/küçük harf çakışması
var (Windows'ta dosyalar birbirinin üzerine yazılır). v1.1 ikisini de düzeltir;
dosyalar "001_mvswc" gibi numara önekli.

Kesintiye dayanıklı: zaten inen dosyalar atlanır, tekrar çalıştırmak güvenli.

Çalıştırma:
    python -m scripts.data.get_gaps                      # tamamı, ~15 GB
    python -m scripts.data.get_gaps --splits train test  # sadece bölünmüş kayıtlar (önerilen)
    python -m scripts.data.get_gaps --limit 5            # hızlı deneme
"""

import argparse
import os

from huggingface_hub import snapshot_download

from gtab.data.gaps import REPO, GAPS_DIR, META, load_metadata


def main(splits=None, limit=None):
    meta = load_metadata()
    if splits:
        meta = meta[meta["split"].isin(splits)]
    if limit:
        meta = meta.head(limit)
    ids = list(meta["id"])
    print(f"{len(ids)} kayit indirilecek -> {GAPS_DIR}")

    patterns = []
    for i in ids:
        patterns += [f"audio/{i}.wav", f"midi/{i}.mid", f"musicxml/{i}.xml",
                     f"syncpoints/{i}.json", f"match/{i}.match"]
    snapshot_download(REPO, repo_type="dataset", local_dir=GAPS_DIR,
                      allow_patterns=patterns + ["README.md", META])

    missing = [i for i in ids if not os.path.exists(os.path.join(GAPS_DIR, "audio", f"{i}.wav"))]
    print(f"Bitti. Eksik ses: {len(missing)}" + (f" -> {missing[:5]}" if missing else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=None, help="train test (bos = hepsi)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.splits, a.limit)
