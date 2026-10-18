"""
Katman 3.13 P3 — Çalınabilirlik ölçümü: üç yöntemin karşılaştırması (SALT OKUNUR).

Üç yaklaşım aynı akor gruplaması ve aynı metrik tanımlarıyla ölçülür:
  1. `model`   — mevcut BiLSTM edisyon modeli (`checkpoints/edition_seq.pt`), konumu kendi seçer
  2. `naif`    — en düşük perde konumu + `edition.finger_baseline`
  3. `editor`  — GAPS partisyonundaki editör parmaklandırması (referans, **kusursuz varsayılmaz**)

ÜÇ PAYDA ayrı ayrı raporlanır (karıştırılmaz):
  1) **kendi kapsamı**    — her yöntem kendi değerlendirebildiği akorlarda (paydalar FARKLI)
  2) **editör alt kümesi** — editörün parmağı tam bilinen akorlar (yöntemler yine düşebilir)
  3) **GERÇEK KESİŞİM**   — üç yöntemin de değerlendirilebilir olduğu akorlar (payda AYNI)

Zaman örtüşmesi: A/B/C/E metrikleri yalnız zamanda örtüşen nota çiftlerinde tanımlıdır
(bkz. `gtab/decoding/playability.py`). Örtüşme gözetmeyen eski tanım karşılaştırma için
"grup tanımı" satırlarında gösterilir. Akor gruplama kodu DEĞİŞTİRİLMEDİ.

Protokol: yalnız `gaps_val` (ya da `gaps_fit`); `gaps_test` reddedilir. Dosya listesi
`protocol.gaps_training_files()` ile alınır (karantina düşer). Eksik parmak etiketi
**tahmin edilmez**; o akorlar "değerlendirilemez" sayılır.

Çalıştırma:
    python -m scripts.eval.eval_playability
    python -m scripts.eval.eval_playability --verify-span 12 --worst 15
"""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from gtab.core.instrument import STANDARD_6
from gtab.data.gaps import GAPS_TAB_DIR
from gtab.decoding.edition import finger_baseline, label_sequence
from gtab.decoding.playability import (SPAN_THRESHOLDS, ChordShape, PlayabilityReport,
                                       evaluate_shapes, finger_conflicts,
                                       finger_order_inversions, group_span, hand_span,
                                       shapes_from_notes)
from gtab.evaluation.protocol import ProtocolError, assert_may_select, gaps_training_files
from gtab.paths import ckpt_path
from gtab.utils import get_device

INSTR = STANDARD_6
METHODS = ("model", "naif", "editor")


def naive_positions(notes: np.ndarray) -> List:
    """En düşük perde kuralı (`eval_edition.naive` ile aynı tanım); çalınamayan perde -> None."""
    out = []
    for _, _, p in notes:
        cands = INSTR.pitch_to_positions(int(p))
        out.append(min(cands, key=lambda sf: sf[1]) if cands else None)
    return out


def record_shapes(path: Path, net, device) -> Dict[str, List[ChordShape]]:
    """Bir kayıt -> yöntem başına akor şekilleri (aynı gruplama, aynı sıra, aynı zamanlar)."""
    from gtab.models.edition_net import predict as seq_predict
    notes, chord, gs, gf, gg = label_sequence(path)
    ms, mf, mg = seq_predict(net, notes, chord, device)
    npos = naive_positions(notes)
    nb = finger_baseline(notes, npos)
    rows = {"model": (ms, mf, mg),
            "naif": ([p[0] if p else -1 for p in npos], [p[1] if p else -1 for p in npos],
                     [g if g is not None else -1 for g in nb]),
            "editor": (gs, gf, gg)}
    return {m: shapes_from_notes(notes, chord, *rows[m]) for m in METHODS}, notes


def masks(shapes: Dict[str, List[ChordShape]]) -> Dict[str, List[bool]]:
    """-> alt küme maskeleri. `kesisim` = üç yöntemin DE değerlendirilebilir olduğu akorlar."""
    n = len(shapes["editor"])
    ed = [shapes["editor"][k].evaluable for k in range(n)]
    strict = [all(shapes[m][k].evaluable for m in METHODS) for k in range(n)]
    return {"kendi": [True] * n, "editor": ed, "kesisim": strict}


def merge(a: PlayabilityReport, b: PlayabilityReport) -> PlayabilityReport:
    """Kayıt raporlarını toplar (paylar ve paydalar toplanır; listeler birleşir)."""
    for k in ("n_groups", "n_evaluable", "n_unevaluable", "n_empty", "n_fretted_notes",
              "n_overlap_groups", "n_overlap_pairs", "n_nonoverlap_pairs",
              "groups_with_nonoverlap", "conflicts", "groups_with_conflict", "group_conflicts",
              "barre_clean", "barre_ambiguous", "inversions", "groups_with_inversion"):
        setattr(a, k, getattr(a, k) + getattr(b, k))
    a.spans += b.spans
    a.group_spans += b.group_spans
    a.spreads += b.spreads
    for k, v in b.shifts.items():
        a.shifts[k] = a.shifts.get(k, 0.0) + v
    return a


def print_table(title: str, reps: Dict[str, PlayabilityReport]) -> None:
    print(f"\n{title}")
    print(f"  {'metrik':<44}" + "".join(f"{m:>14}" for m in METHODS))
    print("  " + "-" * (44 + 14 * len(METHODS)))
    gsp = lambda r: np.array(r.group_spans) if r.group_spans else np.zeros(0)
    rows = [
        ("akor grubu (toplam)", lambda r: f"{r.n_groups}"),
        ("PAYDA-1 degerlendirilebilir akor", lambda r: f"{r.n_evaluable}"),
        ("  etiketi eksik -> degerlendirilemez", lambda r: f"{r.n_unevaluable}"),
        ("  basili notasi yok (yalniz bos tel)", lambda r: f"{r.n_empty}"),
        ("PAYDA-2 basili nota", lambda r: f"{r.n_fretted_notes}"),
        ("PAYDA-3 ortusen basili cifti olan akor", lambda r: f"{r.n_overlap_groups}"),
        ("  ortusen cift / ortusmeyen cift",
         lambda r: f"{r.n_overlap_pairs}/{r.n_nonoverlap_pairs}"),
        ("A) cakisma (ortusen cift; KESIN)", lambda r: f"{r.conflicts}"),
        ("A) cakismali akor orani (PAYDA-3)", lambda r: f"{r.rates()['cakisma_akor_orani']:.4f}"),
        ("A) grup tanimi (ortusme gozetmez)", lambda r: f"{r.group_conflicts}"),
        ("B) aciklik medyan / maks (ortusen)",
         lambda r: f"{r.rates()['aciklik_medyan']:.1f} / {r.rates()['aciklik_maks']:.0f}"),
        ("B) grup tanimi medyan / maks",
         lambda r: (f"{np.median(gsp(r)):.1f} / {gsp(r).max():.0f}" if len(gsp(r)) else "- / -")),
        ("C) ters parmak sirasi (ERGONOMIK RISK)", lambda r: f"{r.inversions}"),
        ("C) ters siralı akor orani (PAYDA-3)",
         lambda r: f"{r.rates()['ters_sira_akor_orani']:.4f}"),
        ("E) barre (temiz / belirsiz)", lambda r: f"{r.barre_clean} / {r.barre_ambiguous}"),
        ("D) gecis / kayma (|dh| >= 1)",
         lambda r: f"{int(r.shifts.get('gecis', 0))} / {int(r.shifts.get('kayma', 0))}"),
        ("D) ortalama hareket (yarim ses)",
         lambda r: f"{r.shifts.get('toplam_hareket', 0) / max(r.shifts.get('gecis', 0), 1):.2f}"),
        ("D) uzun bosluklu gecis (serbest tasima)",
         lambda r: f"{int(r.shifts.get('uzun_bosluk_gecis', 0))}"),
    ]
    for name, fn in rows:
        print(f"  {name:<44}" + "".join(f"{fn(reps[m]):>14}" for m in METHODS))
    print(f"  B) aciklik dagilimi (PAYDA-3 orani, ortusen tanim):")
    for t in SPAN_THRESHOLDS:
        print(f"    >= {t} yarim ses{'':<29}" + "".join(f"{reps[m].rates()[f'aciklik>={t}']:>14.4f}"
                                                        for m in METHODS))


def overlap_diagnostics(reps: Dict[str, PlayabilityReport]) -> None:
    """Madde 4: 30 ms gruplamanin yan etkisi — ortusmeyen ciftlerin metriklere etkisi."""
    print("\n4) ZAMAN ORTUSMESI TANILAMASI (order_notes 30 ms gruplamasi degistirilmedi)")
    print(f"  {'olcut':<44}" + "".join(f"{m:>14}" for m in METHODS))
    print("  " + "-" * (44 + 14 * len(METHODS)))
    for name, fn in [
        ("ortusmeyen cift orani (tum ciftlerde)",
         lambda r: f"{r.rates()['ortusmeyen_cift_orani']:.4f}"),
        ("ortusmeyen cifti olan akor", lambda r: f"{r.groups_with_nonoverlap}"),
        ("A) cakisma: ortusen -> grup tanimi",
         lambda r: f"{r.conflicts} -> {r.group_conflicts}"),
        ("B) maks aciklik: ortusen -> grup tanimi",
         lambda r: (f"{max(r.spans, default=0)} -> {max(r.group_spans, default=0)}")),
    ]:
        print(f"  {name:<44}" + "".join(f"{fn(reps[m]):>14}" for m in METHODS))
    print("  -> Ortusme gozetmeyen tanim, ayni anda basilmayan notalari da sayarak metrigi")
    print("     sisiriyor. Raporun ana sayilari ORTUSEN tanimdan alinmistir.")


def verify_spans(rows: List[dict], limit: int) -> None:
    """Madde 5: asiri acikliklarin ham veriyle dogrulanmasi."""
    rows.sort(key=lambda r: -r["aciklik"])
    print(f"\n5) ASIRI ACIKLIK DOGRULAMASI (ortusen tanimda aciklik >= {limit}; model cikti)")
    if not rows:
        print("  ornek yok")
        return
    for r in rows[:4]:
        print(f"  {r['id']} | grup basi {r['onset']:.3f} sn | aciklik {r['aciklik']} "
              f"(grup tanimi {r['grup_aciklik']}) | cakisma {r['cakisma']}")
        for on, off, p, s, f, g in r["notalar"]:
            print(f"      onset {on:8.3f}  bitis {off:8.3f}  perde {int(p):3d}  ->  "
                  f"tel {int(s)} perde {int(f):2d} parmak {int(g)}")
        print(f"      ortusen cift sayisi {r['ortusen']} | en genis ortusen cift "
              f"{r['en_genis']}")


def main(split: str, worst: int, verify: int, seq: str = "edition_seq.pt") -> None:
    try:
        assert_may_select(split)
    except ProtocolError as e:
        print(f"PROTOKOL: {e}")
        return
    import torch
    from gtab.models.edition_net import EditionNet
    device = get_device()
    net = EditionNet().to(device)
    net.load_state_dict(torch.load(ckpt_path(seq), map_location=device)["model"])

    fit, val = gaps_training_files()
    files = [Path(f) for f in (val if split == "gaps_val" else fit)]
    labeled = [f for f in files if (Path(GAPS_TAB_DIR) / f.name).exists()]
    print(f"Katman 3.13 P3 — calinabilirlik olcumu | split {split} | {len(labeled)}/{len(files)} "
          f"etiketli kayit | cihaz {device}")

    reps = {k: {m: PlayabilityReport(m) for m in METHODS} for k in ("kendi", "editor", "kesisim")}
    span_rows, worst_rows, n_notes = [], [], 0
    for f in labeled:
        shapes, notes = record_shapes(Path(GAPS_TAB_DIR) / f.name, net, device)
        mk = masks(shapes)
        n_notes += len(notes)
        for key, mask in mk.items():
            for m in METHODS:
                merge(reps[key][m], evaluate_shapes([s for s, k in zip(shapes[m], mask) if k], m))
        for s in shapes["model"]:
            if not s.evaluable or not s.overlap_pairs():
                continue
            c, _ = finger_conflicts(s)
            sp, inv = hand_span(s), finger_order_inversions(s)
            if c or (sp or 0) >= 5 or inv:
                worst_rows.append(dict(id=f.stem, onset=s.onset, cakisma=c, aciklik=sp, ters=inv,
                                       sekil=" ".join(f"{a}:{b}:{g}" for a, b, g
                                                      in zip(s.strings, s.frets, s.fingers))))
            if (sp or 0) >= verify:
                pairs = s.overlap_pairs()
                span_rows.append(dict(
                    id=f.stem, onset=s.onset, aciklik=sp, grup_aciklik=group_span(s), cakisma=c,
                    ortusen=len(pairs),
                    en_genis=max(abs(s.frets[i] - s.frets[j]) for i, j in pairs),
                    notalar=[(s.onsets[i], s.offsets[i],
                              s.frets[i] + INSTR.tuning[s.strings[i]] if s.strings[i] >= 0 else -1,
                              s.strings[i], s.frets[i], s.fingers[i]) for i in range(len(s.frets))]))

    ed = reps["kendi"]["editor"]
    print(f"\nKapsam: {n_notes} nota, {ed.n_groups} akor grubu. Editorun parmagi tam bilinen akor "
          f"{ed.n_evaluable} (%{100 * ed.n_evaluable / max(ed.n_groups, 1):.1f}); "
          f"model/naif her akorda parmak uretir.")
    print(f"GERCEK KESISIM (uc yontem de degerlendirilebilir): "
          f"{reps['kesisim']['editor'].n_groups} akor")
    print_table("1) KENDI KAPSAMI (paydalar yontemlere gore FARKLI)", reps["kendi"])
    print_table("2) EDITOR ALT KUMESI (editorun parmagi tam bilinen akorlar)", reps["editor"])
    print_table("3) GERCEK KESISIM (uc yontem de degerlendirilebilir; PAYDA-1 AYNI)",
                reps["kesisim"])
    overlap_diagnostics(reps["kendi"])
    verify_spans(span_rows, verify)
    worst_rows.sort(key=lambda r: (-r["cakisma"], -(r["aciklik"] or 0), -r["ters"]))
    print(f"\n6) EN SORUNLU {min(worst, len(worst_rows))} AKOR (model, kendi kapsami)")
    print(f"  {'kayit':<14}{'sn':>8}{'cakisma':>9}{'aciklik':>9}{'ters':>6}  sekil tel:perde:parmak")
    for r in worst_rows[:worst]:
        print(f"  {r['id']:<14}{r['onset']:>8.1f}{r['cakisma']:>9}{str(r['aciklik']):>9}"
              f"{r['ters']:>6}  {r['sekil']}")
    print("\nNOT: A) kesin fiziksel sorun (yalniz ortusen ciftlerde); C) ergonomik risk; "
          "D) el pozisyonu YAKLASIK; E) barre hata degil.")
    print("Bu kosu yeni kisit uygulamaz, model secimi yapmaz, hicbir dosyaya yazmaz.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="gaps_val", help="gaps_val | gaps_fit (gaps_test reddedilir)")
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--seq", default="edition_seq.pt", help="edisyon modeli (checkpoints/)")
    ap.add_argument("--verify-span", type=int, default=12, help="bu aciklik ve ustu dogrulanir")
    a = ap.parse_args()
    main(a.split, a.worst, a.verify_span, a.seq)
