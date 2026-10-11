"""
Katman 4 Faz 0 — Teknik tanıma ölçütleri (sunum slayt 12: "per-class P/R/F1 + confusion matrix").

Değerlendirme birimi: **NOTA OLAYI**. Bir teknik, nota olayına atfedilir; kare düzeyi
teknik metriği tanımlanmaz (vibrato/legato gibi teknikler nota boyunca sürer, karelerde
sayılırsa uzun notalar metriği domine eder).

Eşleştirme: gerçek ve tahmin notaları **aynı perde + |Δonset| <= 50 ms** ile bire bir
eşlenir (mir_eval nota toleransı ve `eval_edition`'daki uçtan uca eşleme ile aynı kural);
en küçük |Δt| önce bağlanır.

İki raporlama modu (karıştırılmaz):
  MODE_MATCHED  "eslenmis"  : yalnız eşleşen notalar -> teknik modülünün kendi doğruluğu
  MODE_E2E      "uctan-uca" : eşleşmeyen gerçek nota = ilgili sınıfa FN, eşleşmeyen tahmin
                              = ilgili sınıfa FP -> ürün sayısı (nota bulma hatası dahil)

Ana ölçüt **macro-F1**'dir: NORMAL sınıfı baskın olduğu için (GuitarSet/GAPS'te notaların
büyük çoğunluğu teknik taşımaz) weighted-F1 ve doğruluk yanıltıcıdır. Teknik kararları için
ayrıca `macro_f1_no_normal` raporlanır.

Boş sınıf kuralı (raporda aynen kullanılır):
  support > 0                -> ölçülür ("olculdu"), macro-F1'e girer. Hiç tahmin yoksa P = 0.
  support = 0, tahmin yok    -> "tanimsiz": P/R/F1 = None, macro-F1'e GİRMEZ, ayrıca listelenir.
  support = 0, tahmin var    -> "hayalet": P = 0, R tanımsız, F1 = 0; ana macro-F1'e girmez,
                                `macro_f1_strict`'e 0 olarak girer ve confusion matrix'te görünür.

Etiket kaynağı (`label_source`) zorunludur ve rapor başlığında taşınır: hex pickup'tan veya
elle etiketlenmiş "gercek" etiket ile kuraldan türetilmiş "turetilmis" (sözde) etiket
**asla aynı tabloda** raporlanmaz.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

NORMAL = "NORMAL"
MISSING = "yok"                       # eşleşmeyen taraf (confusion matrix'te son satır/sütun)
ONSET_TOL = 0.05                      # sn; mir_eval / eval_edition ile aynı
EPS = 1e-9                            # kayan nokta temsil payı: 1.0 + 0.05 tam 1.05 değildir;
                                      # ölçüm çözünürlüğünün (ms) çok altında, kuralı gevşetmez
EVAL_UNIT = "nota olayi (ayni perde + |dt| <= 50 ms ile bire bir eslenir)"

MODE_MATCHED = "eslenmis"
MODE_E2E = "uctan-uca"
MODES = (MODE_MATCHED, MODE_E2E)

SOURCE_REAL = "gercek"                # hex pickup / insan etiketi
SOURCE_DERIVED = "turetilmis"         # kuraldan uretilmis sozde etiket
LABEL_SOURCES = (SOURCE_REAL, SOURCE_DERIVED)

STATE_SCORED = "olculdu"
STATE_UNDEFINED = "tanimsiz"
STATE_PHANTOM = "hayalet"


@dataclass(frozen=True)
class ClassScore:
    label: str
    tp: int
    fp: int
    fn: int
    support: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    state: str


@dataclass(frozen=True)
class TechniqueReport:
    classes: Tuple[str, ...]
    per_class: Dict[str, ClassScore]
    macro_f1: float
    macro_f1_strict: float
    macro_f1_no_normal: float
    weighted_f1: float
    confusion: np.ndarray                 # (C+1, C+1); satir = gercek, sutun = tahmin, son = MISSING
    confusion_labels: Tuple[str, ...]
    label_source: str
    mode: str
    unit: str
    tol: float
    n_pairs: int
    missed: int                           # eslesmeyen gercek nota
    phantom: int                          # eslesmeyen tahmin notasi
    undefined_classes: Tuple[str, ...]
    phantom_classes: Tuple[str, ...]


def match_events(gold: Sequence[Tuple[float, int, str]], pred: Sequence[Tuple[float, int, str]],
                 tol: float = ONSET_TOL) -> List[Tuple[int, int]]:
    """
    (onset, perde, etiket) listeleri -> bire bir eşleşen (gercek_i, tahmin_j) çiftleri.
    Aynı perde ve |Δonset| <= tol şartı; en küçük |Δt| önce bağlanır (gaps_score.match_notes
    ile aynı kural). Sınır dahildir (EPS = kayan nokta temsil payı).
    """
    cand = []
    for i, (t_g, p_g, _) in enumerate(gold):
        for j, (t_p, p_p, _) in enumerate(pred):
            if p_g == p_p and abs(t_p - t_g) <= tol + EPS:
                cand.append((abs(t_p - t_g), i, j))
    cand.sort()
    used_g, used_p, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j); pairs.append((i, j))
    return sorted(pairs)


def align_labels(gold: Sequence[Tuple[float, int, str]], pred: Sequence[Tuple[float, int, str]],
                 mode: str = MODE_MATCHED, tol: float = ONSET_TOL
                 ) -> Tuple[List[str], List[str], int, int]:
    """
    -> (gercek etiketler, tahmin etiketleri, kacan, hayalet). MODE_E2E'de eşleşmeyen taraflar
    MISSING ile doldurulur; MODE_MATCHED'de atılır (sayıları yine döndürülür).
    """
    if mode not in MODES:
        raise ValueError(f"bilinmeyen mod: {mode!r}")
    pairs = match_events(gold, pred, tol)
    g_lab = [gold[i][2] for i, _ in pairs]
    p_lab = [pred[j][2] for _, j in pairs]
    missed = [i for i in range(len(gold)) if i not in {a for a, _ in pairs}]
    phantom = [j for j in range(len(pred)) if j not in {b for _, b in pairs}]
    if mode == MODE_E2E:
        g_lab += [gold[i][2] for i in missed] + [MISSING] * len(phantom)
        p_lab += [MISSING] * len(missed) + [pred[j][2] for j in phantom]
    return g_lab, p_lab, len(missed), len(phantom)


def _counts(g_lab: Sequence[str], p_lab: Sequence[str], label: str) -> Tuple[int, int, int, int]:
    g = np.asarray(g_lab, dtype=object); p = np.asarray(p_lab, dtype=object)
    is_g, is_p = g == label, p == label
    tp = int((is_g & is_p).sum())
    return tp, int(is_p.sum()) - tp, int(is_g.sum()) - tp, int(is_g.sum())


def _score(label: str, tp: int, fp: int, fn: int, support: int) -> ClassScore:
    if support == 0:
        state = STATE_PHANTOM if fp > 0 else STATE_UNDEFINED
        prec = 0.0 if state == STATE_PHANTOM else None
        f1 = 0.0 if state == STATE_PHANTOM else None
        return ClassScore(label, tp, fp, fn, support, prec, None, f1, state)
    prec = tp / (tp + fp) if tp + fp else 0.0          # hic tahmin yok -> P = 0 (zero_division=0)
    rec = tp / support
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return ClassScore(label, tp, fp, fn, support, prec, rec, f1, STATE_SCORED)


def confusion(g_lab: Sequence[str], p_lab: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """(C+1, C+1) sayım matrisi; satır = gerçek, sütun = tahmin, son indeks = MISSING."""
    labels = list(classes) + [MISSING]
    idx = {c: k for k, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for a, b in zip(g_lab, p_lab):
        if a not in idx or b not in idx:
            raise ValueError(f"sinif listesinde olmayan etiket: {a!r} / {b!r}")
        m[idx[a], idx[b]] += 1
    return m


def evaluate(gold: Sequence[Tuple[float, int, str]], pred: Sequence[Tuple[float, int, str]],
             classes: Sequence[str], label_source: str, mode: str = MODE_MATCHED,
             tol: float = ONSET_TOL) -> TechniqueReport:
    """Nota olayı listeleri -> TechniqueReport. `classes` NORMAL dahil tüm tanımlı sınıflar."""
    if label_source not in LABEL_SOURCES:
        raise ValueError(f"label_source {LABEL_SOURCES} icinden olmali: {label_source!r}")
    classes = tuple(classes)
    if len(set(classes)) != len(classes) or MISSING in classes:
        raise ValueError("sinif listesi tekrarsiz olmali ve MISSING icermemeli")
    g_lab, p_lab, missed, phantom = align_labels(gold, pred, mode, tol)
    per = {c: _score(c, *_counts(g_lab, p_lab, c)) for c in classes}
    scored = [s for s in per.values() if s.state == STATE_SCORED]
    strict = scored + [s for s in per.values() if s.state == STATE_PHANTOM]
    no_norm = [s for s in scored if s.label != NORMAL]
    total = sum(s.support for s in scored)
    return TechniqueReport(
        classes=classes, per_class=per,
        macro_f1=float(np.mean([s.f1 for s in scored])) if scored else float("nan"),
        macro_f1_strict=float(np.mean([s.f1 for s in strict])) if strict else float("nan"),
        macro_f1_no_normal=float(np.mean([s.f1 for s in no_norm])) if no_norm else float("nan"),
        weighted_f1=float(sum(s.f1 * s.support for s in scored) / total) if total else float("nan"),
        confusion=confusion(g_lab, p_lab, classes), confusion_labels=classes + (MISSING,),
        label_source=label_source, mode=mode, unit=EVAL_UNIT, tol=tol, n_pairs=len(g_lab),
        missed=missed, phantom=phantom,
        undefined_classes=tuple(s.label for s in per.values() if s.state == STATE_UNDEFINED),
        phantom_classes=tuple(s.label for s in per.values() if s.state == STATE_PHANTOM))


def majority_baseline(gold: Sequence[Tuple[float, int, str]], classes: Sequence[str],
                      label_source: str, mode: str = MODE_MATCHED) -> TechniqueReport:
    """Taban: her notaya NORMAL de. Kabul kuralı bu tabana göre kurulur (slayt 12 yanlılığı)."""
    pred = [(t, p, NORMAL) for t, p, _ in gold]
    return evaluate(gold, pred, classes, label_source, mode)


def assert_not_mixed(sources: Iterable[str]) -> None:
    """Gerçek ve türetilmiş etiket aynı tabloda raporlanamaz."""
    uniq = set(sources)
    if len(uniq) > 1:
        raise ValueError(f"etiket kaynaklari karistirilamaz: {sorted(uniq)}")


def format_report(rep: TechniqueReport) -> str:
    """Konsol tablosu: başlıkta kaynak/mod/birim, sonra sınıf satırları ve toplamlar."""
    w = max(len(c) for c in rep.classes) + 1
    lines = [f"etiket kaynagi: {rep.label_source} | mod: {rep.mode} | birim: {rep.unit}",
             f"eslesen nota: {rep.n_pairs} | kacan: {rep.missed} | hayalet: {rep.phantom}",
             f"{'sinif':{w}} {'P':>7} {'R':>7} {'F1':>7} {'support':>8}  durum"]
    for c in rep.classes:
        s = rep.per_class[c]
        f = lambda v: f"{v:7.3f}" if v is not None else f"{'-':>7}"
        lines.append(f"{c:{w}} {f(s.precision)} {f(s.recall)} {f(s.f1)} {s.support:8d}  {s.state}")
    lines.append(f"macro-F1 (ANA) {rep.macro_f1:.3f} | macro-F1 (NORMAL haric) {rep.macro_f1_no_normal:.3f} "
                 f"| macro-F1 (hayalet dahil) {rep.macro_f1_strict:.3f} | weighted-F1 {rep.weighted_f1:.3f}")
    if rep.undefined_classes:
        lines.append(f"tanimsiz (support 0, tahmin yok): {', '.join(rep.undefined_classes)}")
    if rep.phantom_classes:
        lines.append(f"hayalet (support 0, tahmin var): {', '.join(rep.phantom_classes)}")
    return "\n".join(lines)
