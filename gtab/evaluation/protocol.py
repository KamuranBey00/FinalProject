"""
Katman 4 Faz 0 — Veri protokolü: hangi split ne için kullanılabilir (TEK KAYNAK).

Neden kod? Katman 3.13'e kadar "test yalnız raporlama" kuralı yalnızca devlog'larda
yazılıydı ve bir kez ihlal edildi (README14 Adım 2 kabul kuralı gaps_test üzerinde
kuruluydu). Bu modül kuralı çalıştırılabilir hale getirir: Katman 4'ün yeni eğitim /
kalibrasyon / raporlama yolları split'i burada doğrular, yanlış kullanım hata verir.

Üç kullanım izni ayrı tutulur:
    may_train  : ağırlık güncellemesinde kullanılabilir
    may_select : eşik / hiperparametre / epoch seçiminde kullanılabilir (kalibrasyon)
    may_report : bir sayı olarak raporlanabilir (report_label ile birlikte)

Katman 4'ün yeni eğitim/ölçüm yolları GAPS dosya listelerini `gaps_training_files()` ile alır
(`gaps_files` yerine): kaynak split doğrulanır, roller kontrol edilir ve karantina kayıtları
(`QUARANTINE_GAPS_IDS`) listelerden çıkarılır.

"Bağımsız test" ayrı bir alan (independent_test): yalnız gaps_test bu niteliği taşır.
GuitarSet oyuncu 05 (val / val_comp) kalibrasyon setidir; sayıları raporlanır ama
**validation** olarak etiketlenir, bağımsız test sayılmaz (oyuncu 05 train_onset'in
seçim skorunda da kullanıldı).
"""

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

# Kullanım rolleri (rapor metinlerinde aynen kullanılır)
EGITIM = "egitim"
SECIM = "secim"
RAPOR = "nihai-rapor"
KESIF = "kesif"
YASAK = "yasak"


class ProtocolError(Exception):
    """Protokol ihlali: bir split izinsiz bir amaçla kullanılmak istendi."""


@dataclass(frozen=True)
class SplitRule:
    name: str
    dataset: str
    content: str
    role: str
    may_train: bool
    may_select: bool
    may_report: bool
    independent_test: bool
    report_label: str
    note: str


def _rule(name, dataset, content, role, may_train, may_select, may_report,
          independent_test, report_label, note) -> SplitRule:
    return SplitRule(name, dataset, content, role, may_train, may_select,
                     may_report, independent_test, report_label, note)


# --- Protokol tablosu -------------------------------------------------------
# GuitarSet: oyuncu 00-04 eğitim, oyuncu 05 kalibrasyon (icracı-ayrık; 6 oyuncu var,
# bu yüzden eğitimden ve kalibrasyondan tamamen bağımsız bir GuitarSet oyuncusu YOK).
# GAPS: resmi train icracı-ayrık ikiye bölünür (gaps_fit eğitim, gaps_val kalibrasyon);
# gaps_test (30 kayıt, 27 icracı) yalnız nihai rapor.
SPLITS: Dict[str, SplitRule] = {r.name: r for r in [
    _rule("train", "GuitarSet", "oyuncu 00-04 solo (150 kayit)", EGITIM,
          True, False, False, False, "egitim verisi",
          "tab (tel/perde) denetimi; celik tel"),
    _rule("train_comp", "GuitarSet", "oyuncu 00-04 akorlu (150 kayit)", EGITIM,
          True, False, False, False, "egitim verisi",
          "polifoni denetimi; celik tel"),
    _rule("val", "GuitarSet", "oyuncu 05 solo (30 kayit)", SECIM,
          False, True, True, False, "validation (bagimsiz test degil)",
          "esik/cozumleme kalibrasyonu + train_onset secim skoru; D1 0.924 / 0.777 buradan"),
    _rule("val_comp", "GuitarSet", "oyuncu 05 akorlu (30 kayit)", SECIM,
          False, True, True, False, "validation (bagimsiz test degil)",
          "polifoni kalibrasyonu; D1 0.807 buradan"),
    _rule("val_half_a", "GuitarSet", "oyuncu 05 solo, parca-ayrik yari A", KESIF,
          False, False, False, False, "kesif / duyarlilik analizi",
          "yalnız duyarlilik analizi: yarida secip diger yarida raporlamak YASAK"),
    _rule("val_half_b", "GuitarSet", "oyuncu 05 solo, parca-ayrik yari B", KESIF,
          False, False, False, False, "kesif / duyarlilik analizi",
          "yalnız duyarlilik analizi: yarida secip diger yarida raporlamak YASAK"),
    _rule("gaps_fit", "GAPS", "gaps_train'in egitim kismi (icraci-ayrik)", EGITIM,
          True, False, False, False, "egitim verisi",
          "perde denetimi + (3.13) partisyon TAB / parmak etiketleri"),
    _rule("gaps_val", "GAPS", "gaps_train'in dogrulama kismi (icraci-ayrik)", SECIM,
          False, True, True, False, "validation (bagimsiz test degil)",
          "GAPS alanindaki tum esik/model secimi burada yapilir"),
    _rule("gaps_test", "GAPS", "resmi GAPS test (30 kayit, 27 icraci)", RAPOR,
          False, False, True, True, "bagimsiz test (icraci-ayrik)",
          "YALNIZ nihai raporlama; model/esik/hiperparametre secimi YASAK"),
    _rule("gaps_skipped", "GAPS", "test icracilariyla ortusen 39 train kaydi", YASAK,
          False, False, False, False, "kullanilmaz",
          "build_gaps --disjoint tarafindan onbellege hic alinmaz; sizinti kaynagi"),
]}

# --- Karantina: icracı kimliği doğrulanamayan kayıtlar --------------------
# `395_w41wc`: GAPS ham metadata'sında performer_name, channel_name, ss_artist, video_title
# alanlarının hepsi boş; elde yalnız yt_id var. Test icracılarından biri olup olmadığı
# doğrulanamıyor -> Katman 4'ün eğitim ve model seçiminde kullanılmaz. Önbellekten
# SİLİNMEZ (mevcut sonuçların yeniden üretilebilirliği bozulmasın); yalnız yeni
# eğitim/seçim yolları bu listeyi dışarıda bırakır.
QUARANTINE_GAPS_IDS: Tuple[str, ...] = ("395_w41wc",)

TRAINING_SPLITS: Tuple[str, ...] = tuple(n for n, r in SPLITS.items() if r.may_train)
SELECTION_SPLITS: Tuple[str, ...] = tuple(n for n, r in SPLITS.items() if r.may_select)
REPORT_ONLY_SPLITS: Tuple[str, ...] = tuple(n for n, r in SPLITS.items() if r.independent_test)


def rule(split: str) -> SplitRule:
    """Bilinmeyen split adı sessizce geçmesin: protokol dışı ad = hata."""
    try:
        return SPLITS[split]
    except KeyError:
        raise ProtocolError(f"protokolde tanimsiz split: {split!r}") from None


def assert_may_train(splits: Sequence[str]) -> None:
    for s in splits:
        if not rule(s).may_train:
            raise ProtocolError(f"{s}: egitimde kullanilamaz ({rule(s).role})")


def assert_may_select(split: str) -> None:
    """Eşik / hiperparametre / epoch seçimi kapısı."""
    r = rule(split)
    if not r.may_select:
        raise ProtocolError(f"{split}: secim/kalibrasyonda kullanilamaz ({r.role}); "
                            f"izinli: {', '.join(SELECTION_SPLITS)}")


def assert_may_report(split: str) -> None:
    r = rule(split)
    if not r.may_report:
        raise ProtocolError(f"{split}: raporlanamaz ({r.role})")


def report_label(split: str) -> str:
    """Her sayının yanında yazılacak split cinsi (validation / bagimsiz test / kesif)."""
    return rule(split).report_label


def is_independent_test(split: str) -> bool:
    return rule(split).independent_test


# --- Sızıntı kontrolleri (saf fonksiyonlar; veri seti gerektirmez) ----------
def split_intersection(a: Iterable[str], b: Iterable[str]) -> Set[str]:
    """İki split'in kayıt kimliği kesişimi (boş olmalı)."""
    return set(a) & set(b)


def performer_overlap(a: Mapping[str, str], b: Mapping[str, str]) -> Set[str]:
    """kayit->icraci eşlemeleri -> iki split'te birlikte geçen icracılar (boş olmalı)."""
    return set(a.values()) & set(b.values())


def normalize_performer(name: object) -> str:
    """
    İcracı adı -> karşılaştırma anahtarı: aksan kaldırılır, küçük harf, noktalama atılır,
    kelimeler sıralanır. GAPS metadata'sında aynı kişi farklı yazılabiliyor (ölçüldü:
    'Ana Vidovic' / 'Ana Vidović', 'Tengyue "TY" Zhang' / 'Tengyue "Ty" Zhang'); ham ad
    eşleşmesi böyle bir varyantı kaçırır ve test icracısı eğitime sızar.
    """
    if name is None or name != name:                  # None / NaN
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(re.sub(r"[^a-z ]", " ", s).split()))


def forbidden_gaps_ids(rows: Iterable[Mapping[str, object]]) -> List[str]:
    """
    GAPS ham metadata satırları -> test icracılarıyla örtüşen train kayıtlarının kimlikleri.
    Beklenen alanlar: id, split ('train'|'test'|NaN), performer_name.
    Eşleşme normalize ad üzerinden (yazım varyantı da yakalanır).
    Bu kayıtlar ne eğitime ne kalibrasyona girebilir (README9: 39 kayıt).
    """
    rows = list(rows)
    test_keys = {normalize_performer(r.get("performer_name")) for r in rows
                 if r.get("split") == "test"} - {""}
    return sorted(str(r["id"]) for r in rows if r.get("split") == "train"
                  and normalize_performer(r.get("performer_name")) in test_keys)


def unnamed_gaps_ids(rows: Iterable[Mapping[str, object]]) -> List[str]:
    """
    İcracı adı boş olan train kayıtları: test icracısı olup olmadıkları AD ile doğrulanamaz
    (kalan risk; kanal adıyla elde doğrulanır).
    """
    return sorted(str(r["id"]) for r in rows if r.get("split") == "train"
                  and not normalize_performer(r.get("performer_name")))


def channel_leakage_suspects(rows: Iterable[Mapping[str, object]]) -> List[Tuple[str, List[str]]]:
    """
    Kanal düzeyi kontrol: bir kanal train ve test'te birlikte geçiyorsa ya (a) tek icracının
    kişisel kanalıdır -> o kanalın TÜM train kayıtları dışlanmış olmalı, ya da (b) çok icracılı
    bir platform kanalıdır (ör. Guitar Salon International, SiccasGuitars) -> örtüşme beklenir.
    -> kuralı ihlal eden (kanal, eğitimde kalan kayıt kimlikleri) listesi (boş olmalı).
    """
    rows = [r for r in rows if r.get("channel_name") and r.get("channel_name") == r.get("channel_name")]
    banned = set(forbidden_gaps_ids(rows))
    by_channel: Dict[str, List[Mapping[str, object]]] = {}
    for r in rows:
        by_channel.setdefault(str(r["channel_name"]), []).append(r)
    out = []
    for ch, items in sorted(by_channel.items()):
        splits = {r.get("split") for r in items}
        if not {"train", "test"} <= splits:
            continue
        people = {normalize_performer(r.get("performer_name")) for r in items} - {""}
        kept = [str(r["id"]) for r in items if r.get("split") == "train" and str(r["id"]) not in banned]
        if len(people) <= 1 and kept:                 # kişisel kanal ama eğitimde kayıt kaldı
            out.append((ch, sorted(kept)))
    return out


def is_quarantined(track_id: str) -> bool:
    """Dosya yolu ya da kayıt kimliği -> karantinada mı."""
    return os.path.splitext(os.path.basename(str(track_id)))[0] in QUARANTINE_GAPS_IDS


def drop_quarantined(items: Iterable[str]) -> List[str]:
    """Dosya yolu / kimlik listesinden karantina kayıtlarını çıkarır (sıra korunur)."""
    return [i for i in items if not is_quarantined(i)]


def gaps_training_files(split: str = "gaps_train") -> Tuple[List[str], List[str]]:
    """
    Katman 4'ün YENİ eğitim ve ölçüm yolları `gaps_files` yerine BUNU çağırır:
      - kaynak split gaps_train değilse (ör. gaps_test) ProtocolError,
      - sonuçtaki rollerin izinleri doğrulanır (gaps_fit eğitim, gaps_val seçim),
      - karantina kayıtları iki listeden de çıkarılır.
    -> (gaps_fit dosyaları, gaps_val dosyaları)
    """
    if split != "gaps_train":
        raise ProtocolError(f"egitim/dogrulama yalniz gaps_train'den bolunur: {split!r}")
    from gtab.data.gaps import gaps_files          # tembel: protocol pandas'a bağlanmasın
    assert_may_train(["gaps_fit"])
    assert_may_select("gaps_val")
    fit, val = gaps_files(split)
    return drop_quarantined(fit), drop_quarantined(val)


def player05_halves(ids: Sequence[str]) -> Tuple[List[str], List[str]]:
    """
    Oyuncu 05 kayıtlarını parça-ayrık iki yarıya böler (yalnız KEŞİF; bağımsız test değil).

    Aynı akor dizisinin iki tempo kaydı (ör. BN1-129-Eb / BN1-147-Gb) birlikte kalır,
    yoksa iki yarı birbirinin neredeyse kopyası olur. Yarı = (stil sırası + dizi no) % 2;
    böylece her stil iki yarıda da temsil edilir ve bölme deterministiktir.
    """
    def parse(fid: str) -> Tuple[str, int]:
        piece = fid.split("_")[1].split("-")[0]              # ör. "BN1", "Funk2"
        digits = "".join(c for c in piece if c.isdigit())
        return piece.rstrip("0123456789"), int(digits or 0)

    styles = sorted({parse(i)[0] for i in ids})
    a, b = [], []
    for fid in sorted(ids):
        style, prog = parse(fid)
        (a if (styles.index(style) + prog) % 2 == 0 else b).append(fid)
    return a, b
