"""
Aktif model yığını + bağımlılık denetimi (model seçimi ve kalibrasyon için tek kaynak).

Yığın `checkpoints/active.json` (yoksa DEFAULT): hangi checkpoint ürün / ölçüm varsayılanı. Taban ses modeli
(acoustic) değişince ona bağlı her şey yeniden üretilmeli; bağımlılıklar burada tek yerde:
  cal          <- acoustic   kalibrasyon (<acoustic>.cal.json; iki alan: gaps_val, val)
  beat_feats   <- acoustic   BeatNet eğitim önbelleği (data/cache/beat_feats/<acoustic adı>/)
  beat_net     <- acoustic   BeatNet, acoustic'in `encode` temsiliyle eğitilir (checkpoint "feats" alanı)
  fingerprint  <- acoustic   çıkarım regresyon referansı (docs/baselines/inference_fingerprint.json "ckpt")
Sembolik modeller (rhythm_q, edition, transitions) ses modelinden bağımsız; yalnız dosyanın varlığı denetlenir.

Yeni taban seçmek: `python -m scripts.eval.check_stack --set acoustic=<yeni>.pt cal=<yeni>.cal.json`, sonra
denetimin yazdığı komutlar sırayla çalıştırılır (bkz. scripts/eval/check_stack.py).
"""

import json
import os

from gtab.paths import CACHE_DIR, ROOT, ckpt_path

ACTIVE_PATH = ckpt_path("active.json")
DEFAULT = dict(acoustic="tabcrnn_deep2.pt", cal="tabcrnn_deep2.cal.json", beat_net="beat_net.pt",
               rhythm_q="rhythm_q.pt", edition="edition_seq.pt", transitions="transitions.npz")
FINGERPRINT = os.path.join(ROOT, "docs", "baselines", "inference_fingerprint.json")


def active():
    """Aktif yığın: DEFAULT + checkpoints/active.json'daki değişiklikler."""
    s = dict(DEFAULT)
    if os.path.exists(ACTIVE_PATH):
        with open(ACTIVE_PATH, encoding="utf-8") as f:
            s.update(json.load(f))
    return s


def set_active(**kw):
    """Yığında verilen girdileri değiştirip checkpoints/active.json'a yazar."""
    unknown = set(kw) - set(DEFAULT)
    if unknown:
        raise KeyError(f"bilinmeyen yigin girdisi: {sorted(unknown)} (gecerli: {sorted(DEFAULT)})")
    s = dict(active(), **kw)
    with open(ACTIVE_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    return s


def _stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def check(s=None):
    """-> [(ad, durum 'ok' | 'eksik' | 'uyumsuz' | 'uyari', açıklama, düzeltme komutu | None)]."""
    import torch
    s = s or active()
    ac = s["acoustic"]
    out = []
    for k in ("acoustic", "cal", "beat_net", "rhythm_q", "edition", "transitions"):
        if not os.path.exists(ckpt_path(s[k])):
            out.append((k, "eksik", ckpt_path(s[k]), None))
    if os.path.exists(ckpt_path(s["cal"])):
        with open(ckpt_path(s["cal"]), encoding="utf-8") as f:
            domains = set(json.load(f))
        if _stem(s["cal"]) != _stem(ac) + ".cal":
            out.append(("cal", "uyumsuz", f"{s['cal']} adi {ac} ile eslesmiyor",
                        f"python -m scripts.eval.eval_pitch --ckpt {ac} --splits gaps_val val val_comp "
                        f"--save-cal checkpoints/{_stem(ac)}.cal.json && "
                        f"python -m scripts.eval.check_stack --set cal={_stem(ac)}.cal.json"))
        elif not {"gaps_val", "val"} <= domains:
            out.append(("cal", "uyumsuz", f"alanlar {sorted(domains)} (gaps_val + val gerekli)", None))
        else:
            out.append(("cal", "ok", f"{s['cal']} <- {ac}", None))
    if os.path.exists(ckpt_path(s["beat_net"])):
        feats = torch.load(ckpt_path(s["beat_net"]), map_location="cpu").get("feats")
        out.append(("beat_net", "ok", f"{s['beat_net']} <- {feats}", None) if feats == ac else
                   ("beat_net", "uyumsuz", f"{s['beat_net']} {feats} temsiliyle egitildi, aktif taban {ac}",
                    f"python -m scripts.data.build_beat_feats --ckpt {ac} && "
                    f"python -m scripts.train.train_beat --ckpt {ac} --out beat_net_{_stem(ac)}.pt && "
                    f"python -m scripts.eval.check_stack --set beat_net=beat_net_{_stem(ac)}.pt"))
    fd = os.path.join(CACHE_DIR, "beat_feats", _stem(ac))
    out.append(("beat_feats", "ok", fd, None) if os.path.isdir(fd) else
               ("beat_feats", "uyari", f"{fd} yok (yalniz BeatNet egitimi icin gerekli)",
                f"python -m scripts.data.build_beat_feats --ckpt {ac}"))
    if os.path.exists(FINGERPRINT):
        with open(FINGERPRINT, encoding="utf-8") as f:
            ref = json.load(f).get("ckpt")
        out.append(("fingerprint", "ok", f"referans <- {ref}", None) if ref == os.path.basename(ac) else
                   ("fingerprint", "uyumsuz", f"referans {ref}, aktif taban {ac}",
                    f"python -m scripts.eval.check_inference_regression --ckpt {ac} --save --force"))
    return out
