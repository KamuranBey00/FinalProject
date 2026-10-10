"""
GAPS (Guitar-Aligned Performance Scores) v1.1 yardımcıları — klasik gitar.

Kaynak: https://huggingface.co/datasets/xavriley/GAPS (MIT, ses dahil).
Ham veri: data/raw/gaps_hf (GAPS_DATA_HOME ile değiştirilebilir).
Önbellek: data/cache/gaps_{train,test} + data/cache/gaps_meta.csv
"""

import glob
import os

import pandas as pd

from gtab.core.note_event import NoteEvent, Transcription
from gtab.paths import RAW_DIR, CACHE_DIR

REPO = "xavriley/GAPS"
GAPS_DIR = os.environ.get("GAPS_DATA_HOME", os.path.join(RAW_DIR, "gaps_hf"))
META = "gaps_metadata_with_splits.csv"
CACHE_META = os.path.join(CACHE_DIR, "gaps_meta.csv")
GAPS_TAB_DIR = os.path.join(CACHE_DIR, "gaps_tab")     # Katman 3.13: partisyon TAB etiketleri (build_gaps_tab)


def load_metadata(gaps_dir=GAPS_DIR) -> pd.DataFrame:
    path = os.path.join(gaps_dir, META)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, META, repo_type="dataset", local_dir=gaps_dir)
    return pd.read_csv(path)


def midi_to_transcription(path) -> Transcription:
    """Hizalı MIDI -> NoteEvent listesi (tüm enstrüman izleri birleştirilir)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(path)
    notes = [NoteEvent(float(n.start), float(n.end), int(n.pitch))
             for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    return Transcription(notes=notes).sort()


def gaps_files(split="gaps_train", val_every=8):
    """
    Önbellekteki bir GAPS split'i -> (eğitim_dosyaları, doğrulama_dosyaları).
    Doğrulama, icracıya göre ayrılır (sıralı icracı listesinde her 'val_every'
    icracıdan biri) -> deterministik ve performer-disjoint.
    """
    files = sorted(glob.glob(os.path.join(CACHE_DIR, split, "*.npz")))
    if not files:
        raise FileNotFoundError(f"{CACHE_DIR}/{split} bos. build_gaps calisti mi?")
    meta = pd.read_csv(CACHE_META).set_index("id")
    tid = lambda f: os.path.basename(f)[:-4]
    perf = {tid(f): str(meta.performer.get(tid(f), "?")) for f in files}
    val_perf = set(sorted(set(perf.values()))[::val_every])
    tr = [f for f in files if perf[tid(f)] not in val_perf]
    va = [f for f in files if perf[tid(f)] in val_perf]
    return tr, va
