"""
Proje yolları — tek kaynak.

Script'ler hangi klasörden çalıştırılırsa çalıştırılsın aynı dosyaları bulsun diye
tüm yollar proje kökünden (bu dosyanın iki üstü) türetilir.

    data/raw/       ham veri setleri (GAPS, SynthTab zip'leri)   -- git'e girmez
    data/cache/     önbellek (CQT + etiket .npz)
    checkpoints/    eğitilmiş modeller (*.pt) ve geçiş modeli (transitions.npz)
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
CKPT_DIR = os.path.join(ROOT, "checkpoints")


def ckpt_path(name: str) -> str:
    """
    Checkpoint adı ya da yolu -> tam yol.
    'tabcrnn_comp.pt' gibi çıplak bir ad checkpoints/ altında aranır;
    klasör içeren ya da mevcut bir yol olduğu gibi kullanılır.
    """
    if os.path.isabs(name) or os.path.dirname(name) or os.path.exists(name):
        return name
    return os.path.join(CKPT_DIR, name)


def split_dir(split: str) -> str:
    return os.path.join(CACHE_DIR, split)
