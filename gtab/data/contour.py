"""
Katman 4 Faz 2A — Sürekli F0 (kontur) altyapısı: güvenilir okuma, hizalama, anomali İŞARETLEME.

Teknik tespitinin girdisi tel başına sürekli F0'dır. Bu modül yalnız **okur, hizalar ve
işaretler**; hiçbir örneği düzeltmez, silmez, yumuşatmaz. Gerekçe: büyük bir frekans değişimi
gerçek bir slide/vibrato de olabilir, perde takipçisinin hatası da; ikisini ayırmak Faz 2B'nin
(insan doğrulaması) işi. Bu yüzden çıktı "temizlenmiş kontur" değil, **güvenilirlik maskeli**
konturdur.

GuitarSet JAMS'te ölçülen gerçekler (Faz 2A, 24 dosyalık keşif + tam tarama):
  - `pitch_contour` tel başına, adım 5.805 ms (256/44100); adımların %98'i tam bu değerde.
  - `voiced` alanı veride HER ZAMAN True; sessiz bölgeler örnek olarak yok, zaman boşluğu olarak var.
  - `frequency == 0` örnekleri VAR (voiced=True olmasına rağmen) -> geçersiz, log alınamaz.
  - `value.index` alanı takipçinin kontur segment kimliği: her zaman boşluğunda değişiyor
    (3236/3236) ama boşluk olmadan da değişebiliyor -> takipçinin izi kaybettiği yerler.
  - `data_source` "0".."5" = tel indeksi ve `STANDARD_6.tuning[0..5]` ile birebir uyumlu
    (ölçüldü: src 0 en pes, nota perdesi min 39.8 ≈ E2 40) -> yeniden indeksleme gerekmez.
  - Tel aralığı dışı örnekler VAR (ör. 5. telde 61.7 Hz = açık telin ~29 yarım ses altı).

Bayraklar örnek başına bit maskesidir; biri NORMAL/teknik kararı vermez — yalnız kaydı anlatır.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from gtab.core.instrument import Instrument, STANDARD_6

STEP_SEC = 256.0 / 44100.0      # 5.805 ms; JAMS'te ölçülen adım
GAP_SEC = 0.02                  # bundan büyük adım = voiced blok kesintisi (adımın ~3.4 katı)
RANGE_PAD = 0.5                 # yarım ses; tel aralığı payı (akort sapması + bend payı)

# Katman 4 Faz 2A önerisinden (README17 Ek A) başlatılan eşikler. DOĞRULANMIŞ EVRENSEL EŞİK DEĞİLDİR:
# `jump_report` ile dağılım ölçülüp raporlanır, Faz 2B'de insan doğrulamasıyla sınanır.
JUMP_CENTS = 600.0              # ardışık iki örnek arası sıçrama (5.8 ms'de yarım oktav)
NET_CENTS = 700.0               # nota içi net kayma üst sınırı

# Örnek bayrakları (bit maskesi)
F_ZERO = 1                      # frequency <= 0 -> geçersiz
F_NONFINITE = 2                 # NaN / inf -> geçersiz
F_RANGE = 4                     # telin fiziksel perde aralığı dışı -> geçersiz
F_UNVOICED = 8                  # voiced == False -> geçersiz (bu veride görülmedi)
F_JUMP = 16                     # ardışık sıçrama > jump_cents -> ŞÜPHELİ (silinmez)
F_SEGSTART = 32                 # segment başlangıcı (zaman boşluğu veya takipçi index değişimi)
INVALID_MASK = F_ZERO | F_NONFINITE | F_RANGE | F_UNVOICED

# Nota düzeyi kontur durumu
RELIABLE = "RELIABLE"           # teknik kararı verilebilir
SUSPECT = "SUSPECT"             # şüpheli örnek/sıçrama/kapsama eksiği var
INVALID = "INVALID"             # geçerli örnek çok az
EMPTY = "EMPTY"                 # hiç kontur örneği yok
UNKNOWN = "UNKNOWN"             # teknik etiketi üretilemeyen nota (NORMAL İLE AYNI ŞEY DEĞİL)

MIN_SAMPLES = 10                # nota için en az geçerli örnek
MIN_COVERAGE = 0.6              # nota süresinin en az bu kadarı örneklenmiş olmalı


def hz_to_cents(hz: np.ndarray) -> np.ndarray:
    """
    Hz -> cent (MIDI x 100). Geçersiz (<= 0, NaN, inf) girişler **NaN** olur; log(0) alınmaz.
    A4 = 440 Hz = MIDI 69 = 6900 cent.
    """
    hz = np.asarray(hz, dtype=np.float64)
    out = np.full(hz.shape, np.nan)
    ok = np.isfinite(hz) & (hz > 0)
    out[ok] = (69.0 + 12.0 * np.log2(hz[ok] / 440.0)) * 100.0
    return out


def string_range_cents(string: int, instrument: Instrument = STANDARD_6,
                       pad: float = RANGE_PAD) -> Tuple[float, float]:
    """Bir telin fiziksel perde aralığı (cent): açık tel - pad .. açık tel + perde sayısı + pad."""
    open_midi = instrument.tuning[string]
    return (open_midi - pad) * 100.0, (open_midi + instrument.num_frets + pad) * 100.0


@dataclass
class StringContour:
    """Bir telin konturu. `cents` ASLA düzeltilmez; geçersiz örnekler NaN + bayrak."""
    string: int
    time: np.ndarray            # (N,) saniye
    cents: np.ndarray           # (N,) cent, geçersiz = NaN
    flags: np.ndarray           # (N,) uint8 bit maskesi
    segment: np.ndarray         # (N,) int32 segment kimliği

    @property
    def valid(self) -> np.ndarray:
        return (self.flags & INVALID_MASK) == 0

    @property
    def reliable(self) -> np.ndarray:
        """Geçerli ve sıçrama işaretlenmemiş örnekler."""
        return self.valid & ((self.flags & F_JUMP) == 0)

    def __len__(self) -> int:
        return len(self.time)


def segment_ids(time: np.ndarray, index: np.ndarray, gap_sec: float = GAP_SEC) -> np.ndarray:
    """
    Segment kimliği: zaman boşluğu VEYA takipçinin `index` değişimi yeni segment başlatır.
    İki ölçüt birlikte kullanılır; `index` takipçinin izi kaybettiği yeri de gösteriyor.
    """
    if len(time) == 0:
        return np.zeros(0, np.int32)
    gap = np.r_[True, np.diff(time) > gap_sec]
    newix = np.r_[True, np.diff(index) != 0]
    return np.cumsum(gap | newix).astype(np.int32) - 1


def flag_samples(cents: np.ndarray, hz: np.ndarray, voiced: np.ndarray, string: int,
                 segment: np.ndarray, instrument: Instrument = STANDARD_6,
                 jump_cents: float = JUMP_CENTS, pad: float = RANGE_PAD) -> np.ndarray:
    """
    Örnek başına bayrak maskesi. Hiçbir değer değiştirilmez.
    Sıçrama yalnız AYNI segment içindeki ardışık örnekler arasında aranır (segment sınırındaki
    büyük fark zaten takipçi kesintisi, gerçek bir perde hareketi değil).
    """
    n = len(cents)
    flags = np.zeros(n, np.uint8)
    flags[~np.asarray(voiced, bool)] |= F_UNVOICED
    hz = np.asarray(hz, float)
    flags[hz <= 0] |= F_ZERO
    flags[~np.isfinite(hz)] |= F_NONFINITE
    lo, hi = string_range_cents(string, instrument, pad)
    with np.errstate(invalid="ignore"):
        flags[np.isfinite(cents) & ((cents < lo) | (cents > hi))] |= F_RANGE
    if n:
        start = np.r_[True, segment[1:] != segment[:-1]]
        flags[start] |= F_SEGSTART
        d = np.abs(np.diff(cents))
        same = segment[1:] == segment[:-1]
        jump = np.r_[False, same & np.isfinite(d) & (d > jump_cents)]
        flags[jump] |= F_JUMP
    return flags


def jump_report(contours: Dict[int, StringContour]) -> Dict[str, float]:
    """
    Ardışık örnek farkının (|Δcent|, aynı segment içinde) dağılımı. Eşiğin kuyrukta mı yoksa
    gerçek hareketin içinde mi kaldığını göstermek için — eşik gerekçesi veriye dayansın.
    """
    deltas = []
    for c in contours.values():
        if len(c) < 2:
            continue
        d = np.abs(np.diff(c.cents))
        same = c.segment[1:] == c.segment[:-1]
        deltas.append(d[same & np.isfinite(d)])
    if not deltas:
        return {}
    d = np.concatenate(deltas)
    q = lambda p: float(np.percentile(d, p))
    return {"n": int(d.size), "p50": q(50), "p99": q(99), "p999": q(99.9),
            "max": float(d.max()), ">=600c": int((d >= JUMP_CENTS).sum()),
            ">=300c": int((d >= 300).sum()), ">=100c": int((d >= 100).sum())}


@dataclass
class NoteContour:
    """Bir notanın kontur özeti. Teknik etiketi ÜRETİLMEZ; yalnız uygunluk değerlendirilir."""
    string: int
    onset: float
    offset: float
    midi: float
    n_samples: int              # pencereye düşen TÜM örnekler (geçersizler dahil; bağlam için)
    n_valid: int
    n_jump: int
    coverage: float             # GEÇERLİ örneklenmiş süre / nota süresi (bkz. note_contours)
    net_cents: float            # ilk %20 -> son %20 medyan farkı (yalnız geçerli örnekler)
    status: str

    @property
    def eligible(self) -> bool:
        return self.status == RELIABLE


def note_status(n_valid: int, n_jump: int, coverage: float, net_cents: float,
                min_samples: int = MIN_SAMPLES, min_coverage: float = MIN_COVERAGE,
                net_limit: float = NET_CENTS) -> str:
    """
    Nota kontur durumu. Sıra önemli: veri yokluğu INVALID, şüphe SUSPECT.
    **Şüpheli nota NORMAL'a düşürülmez**; Faz 3'te UNKNOWN olarak kalır.
    """
    if n_valid == 0:
        return EMPTY
    if n_valid < min_samples:
        return INVALID
    if n_jump > 0 or coverage < min_coverage or abs(net_cents) > net_limit:
        return SUSPECT
    return RELIABLE


def fallback_label(status: str) -> Optional[str]:
    """
    Teknik etiketi üretilemeyen notanın etiketi. RELIABLE -> None (kararı Faz 3 verir),
    diğer her durum -> UNKNOWN. **Hiçbir koşulda NORMAL döndürmez** (test ile sabitlendi).
    """
    return None if status == RELIABLE else UNKNOWN


def note_contours(contour: StringContour, notes: np.ndarray, string: int) -> List[NoteContour]:
    """
    (N,3) nota dizisi [(başlangıç, süre, MIDI)] + telin konturu -> nota başına özet.
    Hizalama: notanın [başlangıç, bitiş] penceresine düşen örnekler; tel kimliği korunur.

    `coverage` YALNIZ GEÇERLİ örnekler üzerinden hesaplanır. Sebebi: kapsama bir güvenilirlik
    kapısıdır ("notanın ne kadarını ölçebiliyoruz?"), oysa geçersiz örnek (0 Hz, tel aralığı dışı)
    hiçbir perde bilgisi taşımaz. Tüm örnekler sayılırsa, konturu büyük ölçüde geçersiz olan bir
    nota kapsama kapısından geçer ve sessizce RELIABLE olur.
    """
    out = []
    for onset, dur, midi in np.asarray(notes, float).reshape(-1, 3):
        offset = onset + dur
        m = (contour.time >= onset) & (contour.time <= offset)
        idx = np.flatnonzero(m)
        valid = idx[contour.valid[idx]] if len(idx) else idx
        cov = float(len(valid) * STEP_SEC / dur) if dur > 0 else 0.0
        cents = contour.cents[valid]
        net = 0.0
        if len(cents) >= 4:
            k = max(2, len(cents) // 5)
            net = float(np.median(cents[-k:]) - np.median(cents[:k]))
        n_jump = int((contour.flags[idx] & F_JUMP).astype(bool).sum()) if len(idx) else 0
        out.append(NoteContour(string, float(onset), float(offset), float(midi), len(idx),
                               len(valid), n_jump, min(cov, 1.0), net,
                               note_status(len(valid), n_jump, min(cov, 1.0), net)))
    return out


def load_jams_contours(path, instrument: Instrument = STANDARD_6, jump_cents: float = JUMP_CENTS
                       ) -> Tuple[Dict[int, StringContour], Dict[int, np.ndarray]]:
    """
    GuitarSet JAMS -> (tel -> StringContour, tel -> notalar (N,3)).
    `data_source` tel indeksidir (0 = en pes; ölçümle doğrulandı). Ses dosyası okunmaz.
    """
    import json
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    contours: Dict[int, StringContour] = {}
    notes: Dict[int, np.ndarray] = {}
    for ann in doc["annotations"]:
        src = ann["annotation_metadata"].get("data_source", "")
        if not src.isdigit():
            continue
        s = int(src)
        if ann["namespace"] == "note_midi":
            d = ann["data"]
            notes[s] = (np.array([[n["time"], n["duration"], n["value"]] for n in d], float)
                        if d else np.zeros((0, 3)))
        elif ann["namespace"] == "pitch_contour":
            d = ann["data"]
            if not d:
                contours[s] = StringContour(s, np.zeros(0), np.zeros(0),
                                            np.zeros(0, np.uint8), np.zeros(0, np.int32))
                continue
            t = np.asarray(d["time"], float)
            hz = np.array([v["frequency"] for v in d["value"]], float)
            voiced = np.array([bool(v["voiced"]) for v in d["value"]])
            index = np.array([int(v["index"]) for v in d["value"]])
            seg = segment_ids(t, index)
            cents = hz_to_cents(hz)
            flags = flag_samples(cents, hz, voiced, s, seg, instrument, jump_cents)
            cents = np.where((flags & INVALID_MASK) == 0, cents, np.nan)   # geçersiz = NaN
            contours[s] = StringContour(s, t, cents, flags, seg)
    return contours, notes


def empty_contour(string: int) -> StringContour:
    """Örneksiz kontur (ör. telde `pitch_contour` anotasyonu hiç yok)."""
    return StringContour(string, np.zeros(0), np.zeros(0), np.zeros(0, np.uint8),
                         np.zeros(0, np.int32))


def iter_strings(contours: Dict[int, StringContour], notes: Dict[int, np.ndarray]):
    """
    Kontur ve nota anahtarlarının BİRLEŞİMİ üzerinde (tel, kontur, notalar) üretir.
    Yalnız `contours` üzerinde dolaşmak, `note_midi` olup `pitch_contour` olmayan bir telin
    notalarını sessizce atlardı; o notaların da sayılması ve EMPTY olarak işaretlenmesi gerekir
    (atlanan nota = görünmeyen nota; UNKNOWN bile olamaz).
    """
    for s in sorted(set(contours) | set(notes)):
        yield s, contours.get(s, empty_contour(s)), notes.get(s, np.zeros((0, 3)))


def quality_stats(contours: Dict[int, StringContour], notes: Dict[int, np.ndarray]
                  ) -> Dict[str, int]:
    """İşlem öncesi/sonrası sayımlar: örnek bayrakları + nota durumları."""
    st: Dict[str, int] = {k: 0 for k in
                          ("ornek", "gecerli", "sifir", "nonfinite", "aralik_disi", "unvoiced",
                           "sicrama", "segment", "nota", RELIABLE, SUSPECT, INVALID, EMPTY)}
    for s, c, nt in iter_strings(contours, notes):
        st["ornek"] += len(c)
        st["gecerli"] += int(c.valid.sum())
        st["sifir"] += int((c.flags & F_ZERO).astype(bool).sum())
        st["nonfinite"] += int((c.flags & F_NONFINITE).astype(bool).sum())
        st["aralik_disi"] += int((c.flags & F_RANGE).astype(bool).sum())
        st["unvoiced"] += int((c.flags & F_UNVOICED).astype(bool).sum())
        st["sicrama"] += int((c.flags & F_JUMP).astype(bool).sum())
        st["segment"] += int((c.flags & F_SEGSTART).astype(bool).sum())
        for nc in note_contours(c, nt, s):
            st["nota"] += 1
            st[nc.status] += 1
    return st
