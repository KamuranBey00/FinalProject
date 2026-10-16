"""
Katman 4 Faz 0 — D1 referans tablosu (regresyon koruması).

Tablo verisi: `docs/baselines/d1_baseline.json`. Bu modül yalnız okur, doğrular ve
karşılaştırır; Katman 4 boyunca referans değerler DEĞİŞTİRİLMEZ (yeni katmanın sonucu
kötüyse tablo düzeltilmez, sonuç raporlanır).

Üç koruma:
  1. Her metrik bir `grup` taşır; farklı grupların sayıları karşılaştırılamaz
     (ör. 0.815 "oracle nota + edisyon" ile 0.655 "uçtan uca" aynı şey değil).
  2. Her metrik `split` taşır ve split cinsi protokolden okunur (validation / bağımsız test).
  3. Her metrik `kosullu` alanında kendisine verilen bilgiyi (oracle nota / oracle konum)
     açıkça yazar; "oracle" sayılar ürün sayısı olarak sunulamaz.
"""

import json
from pathlib import Path
from typing import Dict, List

from gtab.evaluation.protocol import report_label, rule
from gtab.paths import ROOT

BASELINE_PATH = Path(ROOT) / "docs" / "baselines" / "d1_baseline.json"

REQUIRED_FIELDS = ("id", "aciklama", "deger", "veri_seti", "split", "birim",
                   "kosullu", "cozumleme", "grup", "cekirdek", "kaynak")
MODE_RERUN = "yeniden-kosu"
MODE_RETRAIN = "yeniden-egitim"
EPS = 1e-9                  # kayan nokta temsil payı (0.779 + 0.002 tam 0.781 değildir);
                            # metrikler 3 haneye raporlanıyor, toleransı gevşetmez


def load_baselines(path: Path = BASELINE_PATH) -> Dict[str, object]:
    """JSON -> doğrulanmış tablo. Her metriğe protokolden `split_rolu` eklenir."""
    with path.open(encoding="utf-8") as f:
        table = json.load(f)
    seen = set()
    for m in table["metrikler"]:
        missing = [k for k in REQUIRED_FIELDS if k not in m]
        if missing:
            raise ValueError(f"{m.get('id')}: eksik alan {missing}")
        if m["id"] in seen:
            raise ValueError(f"tekrar eden metrik id: {m['id']}")
        seen.add(m["id"])
        rule(m["split"])                                  # protokolde tanimsiz split -> hata
        m["split_rolu"] = report_label(m["split"])
    for mode in (MODE_RERUN, MODE_RETRAIN):
        if mode not in table["toleranslar"]:
            raise ValueError(f"tolerans tanimi eksik: {mode}")
    return table


def metrics() -> Dict[str, Dict[str, object]]:
    return {m["id"]: m for m in load_baselines()["metrikler"]}


def metric(metric_id: str) -> Dict[str, object]:
    try:
        return metrics()[metric_id]
    except KeyError:
        raise KeyError(f"referans tabloda yok: {metric_id!r}") from None


def core_metrics() -> List[Dict[str, object]]:
    """Kullanıcının D1 listesi (6 metrik); kapanış raporunun çekirdeği."""
    return [m for m in load_baselines()["metrikler"] if m["cekirdek"]]


def tolerance(mode: str = MODE_RERUN) -> float:
    return float(load_baselines()["toleranslar"][mode])


def compare(metric_id: str, value: float, mode: str = MODE_RERUN) -> Dict[str, object]:
    """
    Ölçülen değeri referansla karşılaştırır.
    -> dict(ok, delta, tol, ...). `ok` False ise regresyon (ya da açıklanması gereken sapma).
    """
    m = metric(metric_id)
    tol = tolerance(mode)
    delta = float(value) - float(m["deger"])
    return {"id": metric_id, "beklenen": float(m["deger"]), "olculen": float(value),
            "delta": delta, "tol": tol, "mod": mode, "ok": abs(delta) <= tol + EPS,
            "split": m["split"], "split_rolu": m["split_rolu"], "grup": m["grup"],
            "kosullu": m["kosullu"]}


def assert_comparable(metric_id_a: str, metric_id_b: str) -> None:
    """Farklı grup / farklı koşul = karşılaştırılamaz (metrik karıştırma koruması)."""
    a, b = metric(metric_id_a), metric(metric_id_b)
    if a["grup"] != b["grup"]:
        raise ValueError(f"{metric_id_a} ({a['grup']}) ile {metric_id_b} ({b['grup']}) "
                         "karsilastirilamaz: farkli metrik grubu")


def format_table(only_core: bool = False) -> str:
    """Konsol/devlog tablosu: deger, split cinsi, birim, kosullu bilgi."""
    rows = core_metrics() if only_core else load_baselines()["metrikler"]
    lines = [f"{'metrik':28} {'deger':>6}  {'split':10} {'cins':34} birim / kosullu"]
    for m in rows:
        lines.append(f"{m['id']:28} {m['deger']:6.3f}  {m['split']:10} {m['split_rolu']:34} "
                     f"{m['birim']} | {m['kosullu']}")
    return "\n".join(lines)
