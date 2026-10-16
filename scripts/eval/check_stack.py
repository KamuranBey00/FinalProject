"""
Aktif model yığını + bağımlılık denetimi (gtab/models/stack.py). Taban ses modeli değişince kalibrasyon, BeatNet ve
regresyon referansının da yenilenmesi gerekir; bu script hangisinin eski kaldığını ve yeniden üretme komutunu yazar.

Çalıştırma:
    python -m scripts.eval.check_stack                                   # denetle (hata varsa çıkış kodu 1)
    python -m scripts.eval.check_stack --set acoustic=yeni.pt cal=yeni.cal.json   # yeni taban seç
"""

import argparse
import sys

from gtab.models.stack import ACTIVE_PATH, active, check, set_active


def main(a):
    if a.set:
        set_active(**dict(kv.split("=", 1) for kv in a.set))
        print(f"yazildi: {ACTIVE_PATH}")
    s = active()
    print("Aktif yigin:\n" + "\n".join(f"  {k:<12}{v}" for k, v in s.items()))
    rows = check(s)
    print("\nBagimliliklar:")
    for name, st, msg, fix in rows:
        print(f"  [{st:^8}] {name:<12}{msg}")
        if fix:
            print(f"             -> {fix}")
    bad = [r for r in rows if r[1] in ("eksik", "uyumsuz")]
    print("\n" + ("YIGIN TUTARLI." if not bad else f"{len(bad)} sorun: yukaridaki komutlari sirayla calistirin."))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", default=None, help="girdi=dosya ..., orn. acoustic=yeni.pt cal=yeni.cal.json")
    sys.exit(main(ap.parse_args()))
