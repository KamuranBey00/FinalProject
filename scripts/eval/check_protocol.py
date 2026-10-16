"""
Katman 4 Faz 0 — Gerçek önbellek üzerinde veri protokolü / sızıntı denetimi (salt okunur).

Birim testler (tests/) protokol mantığını sentetik veriyle doğrular; bu script aynı kuralları
DİSKTEKİ gerçek önbellekte çalıştırır:
  1. GuitarSet: oyuncu 00-04 (egitim) ve 05 (kalibrasyon) kayit/icraci kesisimi bos mu
  2. GAPS: gaps_train / gaps_test icraci ve kayit kesisimi bos mu
  3. Test icracilariyla ortusen 39 GAPS train kaydi onbellekte YOK mu (ham metadata'dan);
     ad yazim varyanti (normalize ad) ve kanal duzeyi ortusme ayrica kontrol edilir
  4. Karantina kayitlari (icraci kimligi dogrulanamayan) egitim/secim listelerinden cikti mi
  5. gaps_fit / gaps_val icraci-ayrik mi (train_onset ve train_edition_seq bu bolmeyi kullanir)
  6. gaps_tab etiketleri: hangi split'te kac kayit (gaps_test etiketleri yalniz raporlama)
  7. Oyuncu 05 kesif yarilari (parca-ayrik; BAGIMSIZ TEST DEGIL)
  8. Veri envanteri: beklenen dosya sayilari korunmus mu

Hiçbir dosya yazılmaz. Çıkış kodu: 0 = tüm kontroller geçti.

Çalıştırma:
    python -m scripts.eval.check_protocol
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from gtab.data.gaps import CACHE_META, GAPS_DIR, GAPS_TAB_DIR, META, gaps_files
from gtab.evaluation.protocol import (QUARANTINE_GAPS_IDS, SPLITS, channel_leakage_suspects,
                                      forbidden_gaps_ids, gaps_training_files, performer_overlap,
                                      player05_halves, split_intersection, unnamed_gaps_ids)
from gtab.paths import CACHE_DIR

EXPECTED_COUNTS = {"train": 150, "val": 30, "train_comp": 150, "val_comp": 30,
                   "gaps_train": 231, "gaps_test": 30}

results: list = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"[{'GECTI' if ok else 'BASARISIZ'}] {name}: {detail}")


def ids_of(split: str) -> list:
    return sorted(Path(p).stem for p in glob.glob(str(Path(CACHE_DIR) / split / "*.npz")))


def check_guitarset() -> None:
    tr, va = ids_of("train"), ids_of("val")
    perf = lambda ids: {i: i.split("_")[0] for i in ids}
    check("GuitarSet kayit kesisimi (train/val)", not split_intersection(tr, va),
          f"train {len(tr)} | val {len(va)} | kesisim {len(split_intersection(tr, va))}")
    ov = performer_overlap(perf(tr), perf(va))
    check("GuitarSet icraci-ayrikligi", not ov,
          f"train icracilari {sorted(set(perf(tr).values()))} | val {sorted(set(perf(va).values()))}")
    trc, vac = ids_of("train_comp"), ids_of("val_comp")
    ovc = performer_overlap({**perf(tr), **perf(trc)}, {**perf(va), **perf(vac)})
    check("GuitarSet solo+akorlu birlesik icraci-ayrikligi", not ovc, f"ortak icraci: {sorted(ovc) or 'yok'}")


def check_gaps_splits() -> None:
    meta = pd.read_csv(CACHE_META)
    tr = meta[meta.split == "gaps_train"]; te = meta[meta.split == "gaps_test"]
    pm = lambda df: dict(zip(df.id.astype(str), df.performer.astype(str)))
    ov = performer_overlap(pm(tr), pm(te))
    check("GAPS icraci-ayrikligi (train/test)", not ov,
          f"train {len(tr)} kayit / {tr.performer.nunique()} icraci | "
          f"test {len(te)} kayit / {te.performer.nunique()} icraci | ortak icraci {len(ov)}")
    check("GAPS kayit kesisimi (train/test)", not split_intersection(tr.id, te.id),
          f"kesisim {len(split_intersection(tr.id, te.id))}")


def check_forbidden_39(gaps_dir: Path) -> None:
    raw = gaps_dir / META
    if not raw.exists():
        check("Yasakli 39 GAPS kaydi", False, f"ham metadata bulunamadi: {raw} (kontrol yapilamadi)")
        return
    rows = pd.read_csv(raw).to_dict("records")
    bad = forbidden_gaps_ids(rows)
    cached = set(ids_of("gaps_train")) | set(ids_of("gaps_test"))
    labeled = {Path(p).stem for p in glob.glob(str(Path(GAPS_TAB_DIR) / "*.npz"))}
    leaked = sorted(set(bad) & cached)
    check("Yasakli 39 GAPS kaydi onbellekte yok", not leaked,
          f"test icracilariyla ortusen train kaydi: {len(bad)} | onbellekte bulunan: {len(leaked)}")
    check("Yasakli kayitlarin TAB etiketi uretilmemis", not set(bad) & labeled,
          f"gaps_tab icinde: {len(set(bad) & labeled)}")
    # build_gaps ham adla disliyor; normalize ad varyant da yakalar -> ikisi ayni cikmali
    raw_names = {r["performer_name"] for r in rows if r.get("split") == "test"}
    raw_excl = {str(r["id"]) for r in rows if r.get("split") == "train"
                and r.get("performer_name") in raw_names}
    check("Yazim varyanti sizintisi yok (normalize ad)", set(bad) == raw_excl,
          f"ham ad: {len(raw_excl)} kayit | normalize ad: {len(bad)} kayit | "
          f"normalize ile yeni yakalanan: {sorted(set(bad) - raw_excl) or 'yok'}")
    suspects = channel_leakage_suspects(rows)
    check("Kanal duzeyi sizinti yok (kisisel kanal tamamen dislanmis)", not suspects,
          f"ihlal: {suspects or 'yok'} (cok icracili platform kanallari beklenen ortusme)")
    unnamed = [i for i in unnamed_gaps_ids(rows) if i in cached]
    by_id = {str(r["id"]): r for r in rows}
    te_ch = {r.get("channel_name") for r in rows if r.get("split") == "test"}
    unverifiable = [i for i in unnamed if not isinstance(by_id[i].get("channel_name"), str)]
    by_channel = [i for i in unnamed if i not in unverifiable]
    check("Icraci adi bos kayitlar kanal adiyla dogrulandi veya karantinada",
          all(by_id[i].get("channel_name") not in te_ch for i in by_channel)
          and set(unverifiable) <= set(QUARANTINE_GAPS_IDS),
          f"onbellekte {len(unnamed)} adsiz kayit | kanal adiyla temiz: {sorted(by_channel)} | "
          f"kanal adi da bos -> karantina: {sorted(unverifiable)}")


def check_quarantine() -> None:
    """Karantina kayitlari onbellekte DURUR ama egitim/secim listelerine girmez."""
    fit, val = gaps_training_files("gaps_train")
    raw_fit, raw_val = gaps_files("gaps_train")
    vetted = {Path(f).stem for f in fit + val}
    cached = {i for i in QUARANTINE_GAPS_IDS if Path(CACHE_DIR, "gaps_train", f"{i}.npz").exists()}
    dropped = {Path(f).stem for f in raw_fit + raw_val} - vetted
    check("Karantina kayitlari egitim/secim listelerinde yok", not (vetted & set(QUARANTINE_GAPS_IDS)),
          f"karantina: {list(QUARANTINE_GAPS_IDS)} | onbellekte duruyor (silinmedi): {sorted(cached)} | "
          f"listelerden cikarilan: {sorted(dropped)}")
    check("Karantina yalnizca hedef kayitlari cikariyor", dropped <= set(QUARANTINE_GAPS_IDS),
          f"ham liste {len(raw_fit) + len(raw_val)} -> denetimli liste {len(fit) + len(val)} kayit")


def check_fit_val() -> None:
    fit, val = gaps_files("gaps_train")
    meta = pd.read_csv(CACHE_META).set_index("id")
    perf = lambda files: {Path(f).stem: str(meta.performer.get(Path(f).stem, "?")) for f in files}
    ov = performer_overlap(perf(fit), perf(val))
    check("gaps_fit / gaps_val icraci-ayrikligi", not ov,
          f"fit {len(fit)} kayit / {len(set(perf(fit).values()))} icraci | "
          f"val {len(val)} kayit / {len(set(perf(val).values()))} icraci | ortak {len(ov)}")
    check("gaps_val egitimde kullanilmiyor (bolme kaynagi tek)", not split_intersection(
        [Path(f).stem for f in fit], [Path(f).stem for f in val]), "kayit kesisimi 0")


def check_gaps_tab() -> None:
    summary = Path(GAPS_TAB_DIR) / "summary.csv"
    if not summary.exists():
        check("gaps_tab etiket dagilimi", False, "summary.csv yok")
        return
    s = pd.read_csv(summary)
    used = s[s.use]
    n_files = len(glob.glob(str(Path(GAPS_TAB_DIR) / "*.npz")))
    per_split = used.groupby("split").size().to_dict()
    check("gaps_tab etiket dagilimi", n_files == len(used),
          f"dosya {n_files} = kullanilan {len(used)} ({per_split}); "
          f"gaps_test etiketleri YALNIZ raporlama")


def check_halves() -> None:
    a, b = player05_halves(ids_of("val"))
    piece = lambda ids: {i.split("_")[1].split("-")[0] for i in ids}
    check("Oyuncu 05 kesif yarilari parca-ayrik", not (piece(a) & piece(b)),
          f"yari A {len(a)} kayit {sorted(piece(a))} | yari B {len(b)} kayit {sorted(piece(b))} "
          f"-> KESIF/duyarlilik; bagimsiz test DEGIL")


def check_inventory() -> None:
    counts = {s: len(ids_of(s)) for s in EXPECTED_COUNTS}
    bad = {s: (counts[s], n) for s, n in EXPECTED_COUNTS.items() if counts[s] != n}
    check("Veri envanteri korunmus", not bad,
          f"{counts} | beklenenden farkli: {bad or 'yok'}")


def main(gaps_dir: Path) -> int:
    print("Katman 4 Faz 0 — protokol denetimi (salt okunur)\n")
    print("Split rolleri:")
    for r in SPLITS.values():
        print(f"  {r.name:14} {r.dataset:10} egitim={int(r.may_train)} secim={int(r.may_select)} "
              f"rapor={int(r.may_report)} bagimsiz_test={int(r.independent_test)}  <- {r.report_label}")
    print()
    check_guitarset(); check_gaps_splits(); check_forbidden_39(gaps_dir)
    check_quarantine(); check_fit_val(); check_gaps_tab(); check_halves(); check_inventory()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} kontrol gecti"
          + (f" | BASARISIZ: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps-dir", default=GAPS_DIR, help="GAPS ham veri dizini (metadata icin)")
    a = ap.parse_args()
    sys.exit(main(Path(a.gaps_dir)))
