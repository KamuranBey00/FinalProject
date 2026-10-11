"""
Katman 4 Faz 2A — GuitarSet konturu -> `data/cache/contour_guitarset/` (yeni dizin).

Mevcut önbelleklerin (train / val / gaps_*) hiçbirine dokunulmaz. Kaynak: GuitarSet JAMS
(`pitch_contour` + `note_midi`); ses dosyası okunmaz, indirme yapılmaz.

Kayıt başına `.npz`:
    t_<tel>      (int32)  zaman indeksi (saniye = indeks x 5.805 ms)
    c_<tel>      (float32) cent; geçersiz örnek NaN
    f_<tel>      (uint8)  bayrak maskesi (gtab.data.contour)
    s_<tel>      (int32)  segment kimliği
    notes_<tel>  (float32) (N,3) [başlangıç, süre, MIDI]
    nstat_<tel>  (uint8)  nota başına durum kodu (RELIABLE/SUSPECT/INVALID/EMPTY)
    ncov_<tel>   (float32) nota başına kapsama; nnet_<tel> (float32) net kayma (cent)
+ `summary.csv`: kayıt başına kalite istatistikleri.

Kontur DÜZELTİLMEZ: yalnız bayraklanır (bkz. gtab/data/contour.py).

Çalıştırma:
    python -m scripts.data.build_contour --dry-run        # boyut + kalite raporu, YAZMAZ
    python -m scripts.data.build_contour                  # 360 kayit -> onbellek
    python -m scripts.data.build_contour --limit 5        # deneme
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from gtab.config import GUITARSET_DATA_HOME
from gtab.data.contour import (EMPTY, INVALID, RELIABLE, STEP_SEC, SUSPECT, iter_strings,
                               jump_report, load_jams_contours, note_contours, quality_stats)
from gtab.paths import CACHE_DIR

OUT_DIR = Path(CACHE_DIR) / "contour_guitarset"
STATUS_CODE = {RELIABLE: 0, SUSPECT: 1, INVALID: 2, EMPTY: 3}
SUMMARY_FIELDS = ["id", "ornek", "gecerli", "sifir", "aralik_disi", "unvoiced", "nonfinite",
                  "sicrama", "segment", "nota", RELIABLE, SUSPECT, INVALID, EMPTY]


def pack(contours, notes) -> Dict[str, np.ndarray]:
    """
    Kayıt -> npz sözlüğü. Zaman, 5.805 ms adımına göre tam sayı indeks olarak saklanır.
    Teller `iter_strings` ile dolaşılır: `note_midi` olup `pitch_contour` olmayan telin notaları
    da yazılır (aksi hâlde önbellekte hiç görünmezdi).
    """
    out: Dict[str, np.ndarray] = {}
    for s, c, nt in iter_strings(contours, notes):
        out[f"t_{s}"] = np.round(c.time / STEP_SEC).astype(np.int32)
        out[f"c_{s}"] = c.cents.astype(np.float32)
        out[f"f_{s}"] = c.flags.astype(np.uint8)
        out[f"s_{s}"] = c.segment.astype(np.int32)
        ncs = note_contours(c, nt, s)
        out[f"notes_{s}"] = nt.astype(np.float32)
        out[f"nstat_{s}"] = np.array([STATUS_CODE[n.status] for n in ncs], np.uint8)
        out[f"ncov_{s}"] = np.array([n.coverage for n in ncs], np.float32)
        out[f"nnet_{s}"] = np.array([n.net_cents for n in ncs], np.float32)
    return out


def write_summary(rows: List[dict]) -> None:
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)


def build(limit: Optional[int], dry_run: bool, overwrite: bool) -> int:
    files = sorted(Path(GUITARSET_DATA_HOME, "annotation").glob("*.jams"))[:limit]
    if not files:
        print(f"HATA: JAMS bulunamadi: {GUITARSET_DATA_HOME}")
        return 1
    free_gb = shutil.disk_usage(CACHE_DIR).free / 1e9
    print(f"{len(files)} JAMS | hedef {OUT_DIR} | bos disk {free_gb:.1f} GB | "
          f"{'DENEME (yazilmaz)' if dry_run else 'yazilacak'}\n")
    if not dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    total, rows, jumps, bytes_written = Counter(), [], Counter(), 0
    for k, path in enumerate(files, 1):
        contours, notes = load_jams_contours(path)
        st = quality_stats(contours, notes)
        total.update(st)
        rows.append({"id": path.stem, **{f: st[f] for f in SUMMARY_FIELDS[1:]}})
        jr = jump_report(contours)
        if jr:
            jumps.update({"n": jr["n"], ">=600c": jr[">=600c"], ">=300c": jr[">=300c"],
                          ">=100c": jr[">=100c"]})
            jumps["max"] = max(jumps.get("max", 0.0), jr["max"])
        data = pack(contours, notes)
        out = OUT_DIR / f"{path.stem}.npz"
        if dry_run:
            bytes_written += sum(a.nbytes for a in data.values())
        elif out.exists() and not overwrite:
            bytes_written += out.stat().st_size
        else:
            np.savez_compressed(out, **data)
            bytes_written += out.stat().st_size
        if k % 60 == 0:
            print(f"  ... {k}/{len(files)}", flush=True)

    report(total, jumps, bytes_written, len(files), dry_run)
    if not dry_run:
        write_summary(rows)
        print(f"\nYazildi: {OUT_DIR} ({len(files)} npz + summary.csv)")
    return 0


def report(total: Counter, jumps: Counter, nbytes: int, n_files: int, dry_run: bool) -> None:
    """İşlem öncesi/sonrası kalite istatistikleri (yüzdelerle)."""
    n = max(total["ornek"], 1)
    nn = max(total["nota"], 1)
    print("== Ornek duzeyi (islem oncesi ham -> bayraklanmis) ==")
    print(f"  toplam ornek      {total['ornek']:9d}")
    print(f"  gecerli           {total['gecerli']:9d}  (%{100 * total['gecerli'] / n:.3f})")
    for key, label in (("sifir", "frequency == 0"), ("aralik_disi", "tel araligi disi"),
                       ("nonfinite", "NaN / inf"), ("unvoiced", "voiced == False")):
        print(f"  {label:22s} {total[key]:7d}  (%{100 * total[key] / n:.3f})  -> GECERSIZ")
    print(f"  {'sicrama (supheli)':22s} {total['sicrama']:7d}  (%{100 * total['sicrama'] / n:.3f})"
          f"  -> ISARETLI, silinmedi")
    print(f"  {'segment basi':22s} {total['segment']:7d}  (takipci kesintisi / zaman boslugu)")
    if jumps:
        print(f"  |delta cent| dagilimi: >=100c {jumps['>=100c']} | >=300c {jumps['>=300c']} | "
              f">=600c {jumps['>=600c']} | max {jumps.get('max', 0):.0f}  (n={jumps['n']})")
    print("\n== Nota duzeyi (teknik karari icin uygunluk) ==")
    for key in (RELIABLE, SUSPECT, INVALID, EMPTY):
        print(f"  {key:9s} {total[key]:7d}  (%{100 * total[key] / nn:.2f})")
    print(f"  toplam nota {total['nota']}  | UNKNOWN olacak (RELIABLE olmayan): "
          f"{total['nota'] - total[RELIABLE]} (%{100 * (total['nota'] - total[RELIABLE]) / nn:.2f})")
    print(f"\n== Boyut ==\n  {nbytes / 1e6:.1f} MB / {n_files} kayit "
          f"({nbytes / max(n_files, 1) / 1e3:.0f} KB ortalama)"
          + ("  [sikistirilmamis tahmin]" if dry_run else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="islenecek JAMS sayisi")
    ap.add_argument("--dry-run", action="store_true", help="yazmadan rapor ver")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    sys.exit(build(a.limit, a.dry_run, a.overwrite))
