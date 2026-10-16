"""
Katman 4 Faz 1 — Teknik kapsamı ve etiket envanteri (SALT OKUNUR; eğitim yok, indirme yok).

Soru: hangi çalma tekniği için elimizde etiket var, hangisi türetilebilir, hangisi yok?

İki kaynak taranır:
  1. GuitarSet JAMS (360 kayıt, yerel): namespace sayımı + tel başına `pitch_contour` (5.8 ms,
     Hz) ve `note_midi` (kesirli MIDI) üzerinden ADAY teknik olayları.
  2. GAPS MusicXML (yalnız gaps_fit + gaps_val; test ve karantina dışarıda): nota düzeyinde
     teknik/artikülasyon etiketi sayımı.

Üretilen sayılar **aday** sayılarıdır, gerçek etiket değildir: GuitarSet'te teknik anotasyonu
yoktur, buradaki kurallar kontur hareketinden çıkarım yapar (Faz 3'ün sözde etiket tohumu).
Eşikler aşağıda sabit ve raporda yazılı; iki ek eşik setiyle duyarlılık da raporlanır.

Çalıştırma:
    python -m scripts.eval.inventory_technique                 # tam tarama (~3-5 dk)
    python -m scripts.eval.inventory_technique --limit 20      # hizli deneme
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from gtab.config import GUITARSET_DATA_HOME
from gtab.data.gaps import CACHE_META, GAPS_DIR
from gtab.evaluation.protocol import QUARANTINE_GAPS_IDS, gaps_training_files

# --- aday kuralı eşikleri (raporda aynen yazılı) ---------------------------
MIN_DUR = 0.25          # sn; daha kısa notada kontur şekli güvenilmez
VIB_P2P = 40.0          # cent; tepe-tepe salınım (p95 - p5)
VIB_RATE = (3.0, 9.0)   # Hz; klasik gitar vibratosu bu aralıkta
VIB_CROSS = 4           # en az yön değişimi
GLIDE_CENTS = 100.0     # cent; nota içi net perde kayması (>= 1 yarım ses)
GLIDE_MONO = 0.8        # net kayma / toplam değişim (tek yönlülük)
LEG_GAP = 0.05          # sn; iki nota arası boşluk üst sınırı
LEG_SEMI = (1, 5)       # yarım ses; komşu perde aralığı
LEG_CONTOUR_GAP = 0.03  # sn; sınırda konturun kesintisiz sayılma eşiği
SEMI_TOL = 20.0         # cent; "tam yarım sese indi" toleransı (slide / bend ayrımı ipucu)

NAMESPACE_WHITELIST = ("pitch_contour", "note_midi", "beat_position", "tempo", "chord", "key_mode")
GAPS_TAGS = {
    "slide": ".//slide", "glissando": ".//glissando", "slur": ".//slur",
    "harmonic": ".//harmonic", "tremolo": ".//tremolo", "trill-mark": ".//trill-mark",
    "mordent": ".//mordent", "turn": ".//turn", "staccato": ".//staccato",
    "accent": ".//accent", "tenuto": ".//tenuto", "fermata": ".//fermata",
    "hammer-on": ".//hammer-on", "pull-off": ".//pull-off", "bend": ".//bend",
}


# ----------------------------------------------------------------- GuitarSet
def _dense(data) -> Tuple[np.ndarray, np.ndarray]:
    """JAMS yoğun biçim -> (zaman, MIDI). Seslendirilmemiş örnekler atılır."""
    t = np.asarray(data["time"], float)
    f = np.array([v["frequency"] if v["voiced"] else 0.0 for v in data["value"]], float)
    ok = f > 0
    return t[ok], 69.0 + 12.0 * np.log2(np.maximum(f[ok], 1e-9) / 440.0)


def load_jams(path: Path) -> Tuple[Counter, Dict[str, Tuple]]:
    """-> (namespace sayımı, tel -> (notalar (N,3), kontur zamanı, kontur MIDI))."""
    with path.open(encoding="utf-8") as fh:
        doc = json.load(fh)
    ns = Counter(a["namespace"] for a in doc["annotations"])
    notes: Dict[str, np.ndarray] = {}
    contour: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for a in doc["annotations"]:
        src = a["annotation_metadata"].get("data_source", "")
        if a["namespace"] == "note_midi":
            d = a["data"]
            notes[src] = (np.array([[n["time"], n["duration"], n["value"]] for n in d], float)
                          if d else np.zeros((0, 3)))
        elif a["namespace"] == "pitch_contour" and a["data"]:
            contour[src] = _dense(a["data"])
    strings = {s: (notes.get(s, np.zeros((0, 3))), *contour.get(s, (np.zeros(0), np.zeros(0))))
               for s in notes}
    return ns, strings


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Kayan ortalama, KENAR KORUMALI. `np.convolve(..., "same")` kenarlarda sıfırla doldurur;
    mutlak cent değerlerinde (~6000) bu yapay rampa üretip tek yönlülük ve salınım
    ölçülerini bozuyordu. `uniform_filter1d(mode="nearest")` kenarı tekrarlar
    (çözümlemede `decode.time_smooth_probs` de aynı yolu kullanıyor).
    """
    if len(x) < k:
        return x
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, size=k, mode="nearest")


def classify_note(cents: np.ndarray, dur: float, p2p: float = VIB_P2P,
                  glide: float = GLIDE_CENTS) -> Tuple[Optional[str], float]:
    """
    Bir notanın kontur sapması (cent) -> (aday sınıf | None, net kayma).
    Sıra: glide (tek yönlü büyük kayma) önce, sonra vibrato (salınım).
    """
    if len(cents) < 10 or dur < MIN_DUR:
        return None, 0.0
    sm = _smooth(cents)
    head, tail = np.median(sm[:max(2, len(sm) // 5)]), np.median(sm[-max(2, len(sm) // 5):])
    net = tail - head
    tv = float(np.abs(np.diff(sm)).sum())
    if abs(net) >= glide and tv > 0 and abs(net) / tv >= GLIDE_MONO:
        return "GLIDE", net
    centred = sm - np.median(sm)
    cross = int((np.diff(np.sign(centred)) != 0).sum())
    rng = float(np.percentile(sm, 95) - np.percentile(sm, 5))
    rate = cross / (2.0 * dur) if dur > 0 else 0.0
    if rng >= p2p and cross >= VIB_CROSS and VIB_RATE[0] <= rate <= VIB_RATE[1]:
        return "VIBRATO", rng
    return None, net


def scan_track(strings: Dict[str, Tuple], p2p: float = VIB_P2P,
               glide: float = GLIDE_CENTS) -> Counter:
    """Bir kayıt -> aday sayımları (nota düzeyi)."""
    c = Counter()
    for notes, ct, cm in strings.values():
        c["nota"] += len(notes)
        for on, dur, _ in notes:
            m = (ct >= on) & (ct <= on + dur)
            cls, net = classify_note(cm[m] * 100.0 - np.median(cm[m] * 100.0) if m.any()
                                     else np.zeros(0), float(dur), p2p, glide)
            if cls:
                c[cls] += 1
                if cls == "GLIDE":
                    c["GLIDE_yarim_ses"] += int(abs(abs(net) % 100.0 - 50.0) >= 50.0 - SEMI_TOL)
        c.update(legato_pairs(notes, ct))
    return c


def legato_pairs(notes: np.ndarray, ct: np.ndarray) -> Counter:
    """Aynı telde ardışık, bitişik perdeli ve konturu kesintisiz nota çiftleri (hammer-on / pull-off)."""
    c = Counter()
    order = np.argsort(notes[:, 0]) if len(notes) else []
    for a, b in zip(order[:-1], order[1:]):
        on_a, dur_a, p_a = notes[a]
        on_b, _, p_b = notes[b]
        gap = on_b - (on_a + dur_a)
        step = int(round(p_b - p_a))
        if not (-LEG_GAP <= gap <= LEG_GAP and LEG_SEMI[0] <= abs(step) <= LEG_SEMI[1]):
            continue
        seg = ct[(ct >= on_a + dur_a - LEG_CONTOUR_GAP) & (ct <= on_b + LEG_CONTOUR_GAP)]
        if len(seg) >= 2 and float(np.diff(seg).max()) <= LEG_CONTOUR_GAP:
            c["HAMMER_ON" if step > 0 else "PULL_OFF"] += 1
    return c


def scan_guitarset(limit: Optional[int] = None) -> Tuple[Counter, Dict[str, Counter], Dict[str, set]]:
    """-> (namespace sayımı, oyuncu -> aday sayımları, aday sınıf -> kayıt kümesi)."""
    files = sorted(Path(GUITARSET_DATA_HOME, "annotation").glob("*.jams"))[:limit]
    ns_total, per_player, tracks = Counter(), defaultdict(Counter), defaultdict(set)
    for k, path in enumerate(files, 1):
        ns, strings = load_jams(path)
        ns_total.update(ns)
        player = path.stem.split("_")[0]
        counts = scan_track(strings)
        per_player[player].update(counts)
        for key in ("VIBRATO", "GLIDE", "HAMMER_ON", "PULL_OFF"):
            if counts[key]:
                tracks[key].add(path.stem)
        if k % 60 == 0:
            print(f"    ... {k}/{len(files)} JAMS", flush=True)
    return ns_total, per_player, tracks


def sensitivity(limit: Optional[int] = None) -> List[Tuple[str, Counter]]:
    """Aynı veride gevşek / sıkı eşikle aday sayıları (tek eşiğe bağlı kalmamak için)."""
    files = sorted(Path(GUITARSET_DATA_HOME, "annotation").glob("*.jams"))[:limit]
    out = []
    for name, p2p, gl in (("gevsek (30 cent / 75 cent)", 30.0, 75.0),
                          ("varsayilan (40 / 100)", VIB_P2P, GLIDE_CENTS),
                          ("siki (60 cent / 150 cent)", 60.0, 150.0)):
        total = Counter()
        for path in files:
            _, strings = load_jams(path)
            total.update(scan_track(strings, p2p, gl))
        out.append((name, total))
    return out


# ----------------------------------------------------------------- GAPS
def pitch_move_distribution(limit: Optional[int] = None) -> Dict[str, float]:
    """
    Nota İÇİ perde hareketinin dağılımı: |net kayma| ve tepe-tepe salınım (cent).
    "GLIDE adayı yok" sonucunu eşik seçimine değil veriye bağlamak için.
    """
    files = sorted(Path(GUITARSET_DATA_HOME, "annotation").glob("*.jams"))[:limit]
    nets, p2ps = [], []
    for path in files:
        _, strings = load_jams(path)
        for notes, ct, cm in strings.values():
            for on, dur, _ in notes:
                m = (ct >= on) & (ct <= on + dur)
                if m.sum() < 10 or dur < MIN_DUR:
                    continue
                sm = _smooth(cm[m] * 100.0)
                k = max(2, len(sm) // 5)
                nets.append(abs(float(np.median(sm[-k:]) - np.median(sm[:k]))))
                p2ps.append(float(np.percentile(sm, 95) - np.percentile(sm, 5)))
    net, p2p = np.array(nets), np.array(p2ps)
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
    return {"nota": len(net), "net_p50": q(net, 50), "net_p90": q(net, 90), "net_p99": q(net, 99),
            "net_max": float(net.max()) if len(net) else float("nan"),
            "net>=50c": int((net >= 50).sum()), "net>=100c": int((net >= 100).sum()),
            "net>=150c": int((net >= 150).sum()),
            "p2p_p50": q(p2p, 50), "p2p_p90": q(p2p, 90), "p2p_p99": q(p2p, 99)}


def scan_gaps(ids: Iterable[str]) -> Tuple[Counter, Counter, int, int]:
    """-> (etiket -> nota/işaret sayısı, etiket -> kayıt sayısı, taranan kayıt, toplam nota)."""
    tag_n, tag_rec, n_rec, n_note = Counter(), Counter(), 0, 0
    for tid in ids:
        path = Path(GAPS_DIR, "musicxml", f"{tid}.xml")
        if not path.exists():
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            tag_n["XML_HATASI"] += 1
            continue
        n_rec += 1
        n_note += sum(1 for _ in root.iter("note"))
        for name, xp in GAPS_TAGS.items():
            k = len(root.findall(xp))
            if k:
                tag_n[name] += k
                tag_rec[name] += 1
    return tag_n, tag_rec, n_rec, n_note


def gaps_inventory_ids() -> Tuple[List[str], int]:
    """Protokol: yalnız gaps_fit + gaps_val (karantina çıkarılmış); gaps_test hiç okunmaz."""
    fit, val = gaps_training_files()
    ids = [Path(f).stem for f in fit + val]
    meta = pd.read_csv(CACHE_META).set_index("id")
    performers = {str(meta.performer.get(i, "?")) for i in ids}
    return ids, len(performers - {"nan"})


# ----------------------------------------------------------------- rapor
def main(limit: Optional[int], sens_limit: Optional[int]) -> None:
    print("Katman 4 Faz 1 — teknik etiket envanteri (salt okunur)\n")
    print("== GuitarSet (JAMS) ==")
    ns, per_player, tracks = scan_guitarset(limit)
    print(f"  namespace sayimi: {dict(ns)}")
    extra = set(ns) - set(NAMESPACE_WHITELIST)
    print(f"  teknik namespace'i: {sorted(extra) if extra else 'YOK (teknik anotasyonu yok)'}")
    total = Counter()
    for c in per_player.values():
        total.update(c)
    print(f"  nota: {total['nota']} | oyuncu: {len(per_player)}")
    for key in ("VIBRATO", "GLIDE", "HAMMER_ON", "PULL_OFF"):
        n = total[key]
        pl = sum(1 for c in per_player.values() if c[key])
        print(f"  {key:11} aday {n:6d} (%{100 * n / max(total['nota'], 1):5.2f}) | "
              f"oyuncu {pl}/{len(per_player)} | kayit {len(tracks[key])}")
    print(f"  GLIDE adaylarindan tam yarim sese inen: {total['GLIDE_yarim_ses']} "
          f"/ {total['GLIDE']} (slide profili; kalani bend/portamento profili)")
    print("  oyuncu kirilimi (nota / VIBRATO / GLIDE / HAMMER_ON / PULL_OFF):")
    for p in sorted(per_player):
        c = per_player[p]
        print(f"    {p}: {c['nota']:6d} / {c['VIBRATO']:5d} / {c['GLIDE']:4d} / "
              f"{c['HAMMER_ON']:4d} / {c['PULL_OFF']:4d}")

    print("\n== Nota ici perde hareketi dagilimi (cent) ==")
    dist = pitch_move_distribution(sens_limit)
    print(f"  {dist['nota']} nota | |net kayma| p50 {dist['net_p50']:.1f} p90 {dist['net_p90']:.1f} "
          f"p99 {dist['net_p99']:.1f} max {dist['net_max']:.1f}")
    print(f"  |net| >= 50c: {dist['net>=50c']} | >= 100c: {dist['net>=100c']} | >= 150c: {dist['net>=150c']}")
    print(f"  tepe-tepe salinim p50 {dist['p2p_p50']:.1f} p90 {dist['p2p_p90']:.1f} p99 {dist['p2p_p99']:.1f}")

    print("\n== Esik duyarliligi (ayni veri, uc esik seti) ==")
    for name, c in sensitivity(sens_limit):
        print(f"  {name:28} VIBRATO {c['VIBRATO']:6d} | GLIDE {c['GLIDE']:5d} "
              f"| nota {c['nota']}")

    print("\n== GAPS MusicXML (yalniz gaps_fit + gaps_val; gaps_test okunmadi) ==")
    ids, n_perf = gaps_inventory_ids()
    tag_n, tag_rec, n_rec, n_note = scan_gaps(ids)
    print(f"  taranan kayit: {n_rec}/{len(ids)} | icraci: {n_perf} | nota: {n_note} | "
          f"karantina disinda: {list(QUARANTINE_GAPS_IDS)}")
    for name in GAPS_TAGS:
        print(f"  {name:12} isaret {tag_n[name]:5d} | kayit {tag_rec[name]:3d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="GuitarSet JAMS sayisi (deneme icin)")
    ap.add_argument("--sens-limit", type=int, default=60, help="duyarlilik taramasi kayit sayisi")
    a = ap.parse_args()
    main(a.limit, a.sens_limit)
