"""
Katman 3.13 P3 — Fiziksel çalınabilirlik ÖLÇÜMÜ (yalnız analiz; hiçbir kısıt uygulanmaz).

Bu modül parmaklandırma çıktısını **değiştirmez, filtrelemez, düzeltmez**. Yalnız sayar ve
sınıflandırır. Mevcut çıkarım hattından bağımsızdır: `edition.py` / `edition_net.py` bu modülü
import etmez.

**ZAMAN ÖRTÜŞMESİ (P3 son doğrulaması).** `edition.order_notes` akorları yalnız BAŞLANGIÇ
yakınlığına göre gruplar (<= 30 ms). Başlangıcı yakın iki nota zamanda örtüşmek zorunda değildir
(ör. 25 ms sonra başlayan nota, ilki bittikten sonra çalınıyor olabilir). Fiziksel ihlal ancak
notalar AYNI ANDA basılıyken anlamlıdır; bu yüzden A/B/C/E metrikleri **zamanda örtüşen nota
çiftleri** üzerinden hesaplanır (`min(bitiş) − max(başlangıç) > 0`). Gruplama kodu değişmedi;
örtüşme bu modülde ayrıca denetleniyor. Karşılaştırma için örtüşme gözetmeyen eski tanım
`group_span()` / `group_finger_conflicts()` olarak korunuyor.

Ölçülenler:
  A. Parmak çakışması — aynı parmak, ÖRTÜŞEN iki nota, FARKLI perdeler    -> KESİN fiziksel sorun
  B. El açıklığı — örtüşen basılı çiftlerin en büyük perde farkı           -> dağılım, eşik dayatılmaz
  C. Parmak sırası — örtüşen çiftte küçük parmak daha yüksek perdede       -> ERGONOMİK RİSK
  D. El pozisyonu değişimi — ardışık gruplar arası tahmini taşıma          -> YAKLAŞIK ölçü
  E. Barre — aynı parmak, aynı perde, örtüşen birden çok tel               -> ayrı sayılır, hata değil

Sözleşmeler (mevcut koddan alındı, değiştirilmedi):
  - Nota = (başlangıç sn, **bitiş** sn, MIDI perdesi).
  - Akor grubu = `order_notes` çıktısındaki `chord` bayrağı (yeni gruplama üretilmez).
  - Tel 0 = en pes (E2); perde = MIDI − açık tel; parmak 0 = boş tel, 1–4 basılı, −1 bilinmiyor.
  - Uzun boşluk eşiği `LONG_GAP_SEC` = 0,5 sn: bundan uzun arada el serbestçe taşınabilir.

Pay/payda: sayımlar pay; paydalar `n_evaluable` (akor), `n_fretted_notes` (nota),
`n_overlap_groups` (örtüşen basılı çifti olan akor), `gecis` (pozisyon geçişi).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gtab.decoding.transitions import LONG_GAP_SEC

FINGERS = (1, 2, 3, 4)
SPAN_THRESHOLDS = (3, 4, 5, 6)        # yarım ses; yalnız dağılım için, imkânsızlık iddiası değil
UNKNOWN_FINGER = -1
OPEN_FINGER = 0


@dataclass(frozen=True)
class ChordShape:
    """
    Bir akor grubunun parmak şekli + nota zamanları.
    `fingers`: 0 = boş tel, 1–4 basılı, −1 = bilinmiyor. Boş teller ve bilinmeyenler
    çakışma/açıklık hesabına girmez (parmak tutmazlar).
    """
    onset: float
    strings: Tuple[int, ...]
    frets: Tuple[int, ...]
    fingers: Tuple[int, ...]
    onsets: Tuple[float, ...] = ()
    offsets: Tuple[float, ...] = ()

    def __post_init__(self):
        if not self.onsets:                      # zaman verilmediyse: hepsi örtüşüyor varsayılır
            object.__setattr__(self, "onsets", tuple(self.onset for _ in self.frets))
            object.__setattr__(self, "offsets", tuple(self.onset + 1.0 for _ in self.frets))

    @property
    def fretted(self) -> List[int]:
        """Parmağı BİLİNEN basılı notaların indeksleri."""
        return [i for i, (f, g) in enumerate(zip(self.frets, self.fingers))
                if f > 0 and g in FINGERS]

    @property
    def n_unknown(self) -> int:
        """
        Şekli belirsiz bırakan nota sayısı: konumu bilinmeyen (perde < 0) ya da basılı olup
        parmağı bilinmeyen notalar. İkisi de akoru değerlendirilemez yapar — gizli bir nota
        çakışmayı ya da açıklığı değiştirebilir, eksik etiketten sonuç üretilmez.
        """
        return sum(1 for f, g in zip(self.frets, self.fingers)
                   if f < 0 or (f > 0 and g not in FINGERS))

    @property
    def evaluable(self) -> bool:
        """Grupta basılı nota var ve hepsinin parmağı/konumu biliniyor."""
        return self.n_unknown == 0 and len(self.fretted) > 0

    def overlap(self, i: int, j: int) -> float:
        """İki notanın ortak süresi (sn); <= 0 ise aynı anda basılmıyorlar."""
        return min(self.offsets[i], self.offsets[j]) - max(self.onsets[i], self.onsets[j])

    def overlap_pairs(self) -> List[Tuple[int, int]]:
        """Zamanda gerçekten örtüşen basılı nota çiftleri."""
        idx = self.fretted
        return [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:] if self.overlap(i, j) > 0]

    def nonoverlap_pairs(self) -> List[Tuple[int, int]]:
        """Aynı grupta olup örtüşmeyen basılı çiftler (gruplama toleransının yan etkisi)."""
        idx = self.fretted
        return [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:] if self.overlap(i, j) <= 0]


def shapes_from_notes(notes: np.ndarray, chord: Sequence[bool], strings: Sequence[int],
                      frets: Sequence[int], fingers: Sequence[int]) -> List[ChordShape]:
    """
    `order_notes` sırası + konum/parmak dizileri -> akor şekilleri (zamanlar nota dizisinden).
    Gruplama `chord` bayrağından okunur (False = yeni grup); yeni gruplama kuralı ÜRETİLMEZ.
    """
    notes = np.asarray(notes, float).reshape(-1, 3)
    chord = list(chord)
    bounds = [i for i, c in enumerate(chord) if not c] + [len(chord)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = range(a, b)
        out.append(ChordShape(float(notes[a][0]),
                              tuple(int(strings[i]) for i in idx),
                              tuple(int(frets[i]) for i in idx),
                              tuple(int(fingers[i]) for i in idx),
                              tuple(float(notes[i][0]) for i in idx),
                              tuple(float(notes[i][1]) for i in idx)))
    return out


# ----------------------------------------------------------------- A) parmak çakışması
def finger_conflicts(shape: ChordShape) -> Tuple[int, int]:
    """
    -> (çakışan çift sayısı, barre çifti sayısı). YALNIZ örtüşen çiftler sayılır.
    Aynı parmak + örtüşme: perdeler farklıysa **çakışma** (bir parmak aynı anda iki perdede
    olamaz), aynıysa **barre adayı** (hata değil).
    """
    conflicts = barres = 0
    for i, j in shape.overlap_pairs():
        if shape.fingers[i] != shape.fingers[j]:
            continue
        if shape.frets[i] == shape.frets[j]:
            barres += 1
        else:
            conflicts += 1
    return conflicts, barres


def group_finger_conflicts(shape: ChordShape) -> int:
    """
    Örtüşme GÖZETMEYEN eski tanım (karşılaştırma için): aynı parmağın grupta kullandığı
    farklı perde sayısı − 1. Örtüşmeyen notaları da sayar, bu yüzden üst sınırdır.
    """
    total = 0
    for g in FINGERS:
        frets = {shape.frets[i] for i in shape.fretted if shape.fingers[i] == g}
        total += max(len(frets) - 1, 0)
    return total


# ----------------------------------------------------------------- B) el açıklığı
def hand_span(shape: ChordShape) -> Optional[int]:
    """Örtüşen basılı çiftlerin en büyük perde farkı (elin aynı anda germesi gereken aralık)."""
    pairs = shape.overlap_pairs()
    return max((abs(shape.frets[i] - shape.frets[j]) for i, j in pairs), default=None)


def group_span(shape: ChordShape) -> Optional[int]:
    """Örtüşme gözetmeyen eski tanım: gruptaki tüm basılı notaların maks − min perdesi."""
    frets = [shape.frets[i] for i in shape.fretted]
    return (max(frets) - min(frets)) if len(frets) >= 2 else None


# ----------------------------------------------------------------- C) parmak sırası
def finger_order_inversions(shape: ChordShape) -> int:
    """
    Ters çift sayısı: örtüşen iki notada küçük parmak numarası daha YÜKSEK perdede.
    Aynı perdeli (barre) ve aynı parmaklı çiftler sayılmaz. **Kesin ihlal değil** — klasik
    icrada geçerli çapraz parmak kullanımı vardır; ergonomik risk göstergesi.
    """
    inv = 0
    for i, j in shape.overlap_pairs():
        gi, gj, fi, fj = shape.fingers[i], shape.fingers[j], shape.frets[i], shape.frets[j]
        if gi == gj or fi == fj:
            continue
        if (gi < gj) != (fi < fj):
            inv += 1
    return inv


# ----------------------------------------------------------------- D) el pozisyonu
def hand_position(shape: ChordShape) -> Optional[float]:
    """
    Tahmini el pozisyonu = medyan(perde − parmak + 1) (işaret parmağının varsayılan perdesi).
    **Yaklaşık değer**: gerçek el pozisyonu ölçülmüyor, parmak-perde ilişkisinden türetiliyor.
    Grup düzeyindedir (örtüşme ayrımı yapılmaz), çünkü taşıma gruplar arası ölçülür.
    """
    vals = [shape.frets[i] - shape.fingers[i] + 1 for i in shape.fretted]
    return float(np.median(vals)) if vals else None


def position_spread(shape: ChordShape) -> Optional[int]:
    """Aynı akordaki pozisyon tahminlerinin yayılımı; büyük değer = tutarsız parmak şekli."""
    vals = [shape.frets[i] - shape.fingers[i] + 1 for i in shape.fretted]
    return (max(vals) - min(vals)) if len(vals) >= 2 else None


def position_shifts(shapes: Sequence[ChordShape], long_gap: float = LONG_GAP_SEC) -> Dict[str, float]:
    """
    Ardışık (değerlendirilebilir) gruplar arası taşıma. Uzun boşluklu geçişler ayrı sayılır:
    0,5 sn'den uzun arada el serbestçe taşınabilir, bu bir maliyet sayılmamalı.
    """
    seq = [(s.onset, hand_position(s)) for s in shapes if s.evaluable and hand_position(s) is not None]
    tot = n_shift = n_trans = 0.0
    tot_s = n_shift_s = n_trans_s = 0.0
    for (t0, p0), (t1, p1) in zip(seq[:-1], seq[1:]):
        d = abs(p1 - p0)
        n_trans += 1; tot += d; n_shift += d >= 1
        if t1 - t0 <= long_gap:
            n_trans_s += 1; tot_s += d; n_shift_s += d >= 1
    return {"gecis": n_trans, "kayma": n_shift, "toplam_hareket": tot,
            "ortalama_hareket": tot / n_trans if n_trans else float("nan"),
            "kisa_bosluk_gecis": n_trans_s, "kisa_bosluk_kayma": n_shift_s,
            "kisa_bosluk_toplam": tot_s, "uzun_bosluk_gecis": n_trans - n_trans_s}


# ----------------------------------------------------------------- E) barre
def barre_detail(shape: ChordShape) -> List[Dict[str, object]]:
    """
    Barre adayları: aynı parmak + aynı perde + ÖRTÜŞEN birden çok tel.
    `belirsiz` nedenleri: teller bitişik değil; ya da aradaki bir telde daha DÜŞÜK perdeli
    (örtüşen) bir nota var. Hüküm verilmez, yalnız işaretlenir.
    """
    out = []
    for g in FINGERS:
        idx = [i for i in shape.fretted if shape.fingers[i] == g]
        for fret in sorted({shape.frets[i] for i in idx}):
            same = [i for i in idx if shape.frets[i] == fret]
            held = [(i, j) for a, i in enumerate(same) for j in same[a + 1:] if shape.overlap(i, j) > 0]
            if not held:
                continue
            ss = sorted({shape.strings[i] for p in held for i in p})
            gaps = [s for s in range(min(ss), max(ss) + 1) if s not in ss]
            blocked = [i for i in shape.fretted
                       if shape.strings[i] in gaps and shape.frets[i] < fret]
            reason = ("teller bitisik degil" if gaps and not blocked else
                      "arada daha dusuk perde var" if blocked else None)
            out.append({"parmak": g, "perde": fret, "teller": tuple(ss), "belirsiz": reason})
    return out


# ----------------------------------------------------------------- toplama
@dataclass
class PlayabilityReport:
    """Paylar (sayımlar) + paydalar (`n_*`). Oranlar `rates()` ile üretilir."""
    method: str
    n_groups: int = 0                 # tüm akor grupları
    n_evaluable: int = 0              # payda: parmağı/konumu tam bilinen, basılı notası olan gruplar
    n_unevaluable: int = 0            # eksik etiket yüzünden değerlendirilemeyen gruplar
    n_empty: int = 0                  # basılı notası olmayan gruplar (yalnız boş tel)
    n_fretted_notes: int = 0          # payda: değerlendirilebilir gruplardaki basılı notalar
    n_overlap_groups: int = 0         # payda: örtüşen basılı çifti OLAN gruplar (A/B/C/E tanımlı)
    n_overlap_pairs: int = 0          # örtüşen basılı çift sayısı
    n_nonoverlap_pairs: int = 0       # aynı grupta olup örtüşmeyen çiftler
    groups_with_nonoverlap: int = 0
    conflicts: int = 0                # A (örtüşen çiftlerde)
    groups_with_conflict: int = 0
    group_conflicts: int = 0          # A, örtüşme gözetmeyen eski tanım (üst sınır)
    barre_clean: int = 0              # E
    barre_ambiguous: int = 0
    inversions: int = 0               # C
    groups_with_inversion: int = 0
    spans: List[int] = field(default_factory=list)          # B (örtüşme tabanlı)
    group_spans: List[int] = field(default_factory=list)    # B (eski tanım)
    spreads: List[int] = field(default_factory=list)
    shifts: Dict[str, float] = field(default_factory=dict)  # D

    def rates(self) -> Dict[str, float]:
        ev, fr = max(self.n_evaluable, 1), max(self.n_fretted_notes, 1)
        ov = max(self.n_overlap_groups, 1)
        sp = np.array(self.spans) if self.spans else np.zeros(0)
        out = {"degerlendirilebilir_oran": self.n_evaluable / max(self.n_groups, 1),
               "cakisma_akor_orani": self.groups_with_conflict / ov,
               "cakisma_nota_orani": self.conflicts / fr,
               "ters_sira_akor_orani": self.groups_with_inversion / ov,
               "barre_akor_orani": (self.barre_clean + self.barre_ambiguous) / ev,
               "ortusmeyen_cift_orani": self.n_nonoverlap_pairs /
                                        max(self.n_overlap_pairs + self.n_nonoverlap_pairs, 1),
               "aciklik_medyan": float(np.median(sp)) if len(sp) else float("nan"),
               "aciklik_maks": float(sp.max()) if len(sp) else float("nan")}
        for t in SPAN_THRESHOLDS:
            out[f"aciklik>={t}"] = float((sp >= t).sum()) / ov if len(sp) else 0.0
        return out


def evaluate_shapes(shapes: Sequence[ChordShape], method: str) -> PlayabilityReport:
    """Akor şekilleri -> rapor. Hiçbir şekil atılmaz; değerlendirilemeyenler ayrıca sayılır."""
    r = PlayabilityReport(method=method, n_groups=len(shapes))
    for s in shapes:
        if s.n_unknown:
            r.n_unevaluable += 1
            continue
        if not s.fretted:
            r.n_empty += 1
            continue
        r.n_evaluable += 1
        r.n_fretted_notes += len(s.fretted)
        r.group_conflicts += group_finger_conflicts(s)
        gs = group_span(s)
        if gs is not None:
            r.group_spans.append(gs)
        no = s.nonoverlap_pairs()
        r.n_nonoverlap_pairs += len(no)
        r.groups_with_nonoverlap += len(no) > 0
        pairs = s.overlap_pairs()
        r.n_overlap_pairs += len(pairs)
        for b in barre_detail(s):
            if b["belirsiz"]:
                r.barre_ambiguous += 1
            else:
                r.barre_clean += 1
        if not pairs:
            continue                                   # A/B/C tanımsız (eşzamanlı basılı çift yok)
        r.n_overlap_groups += 1
        c, _ = finger_conflicts(s)
        r.conflicts += c
        r.groups_with_conflict += c > 0
        r.spans.append(hand_span(s))
        sp = position_spread(s)
        if sp is not None:
            r.spreads.append(sp)
        inv = finger_order_inversions(s)
        r.inversions += inv
        r.groups_with_inversion += inv > 0
    r.shifts = position_shifts(shapes)
    return r
