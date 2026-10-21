"""Fetch the guitar-sound test data that the repo may not redistribute (scripts/eval/gv, README22).

classtab.org tablature and YouTube performances are downloaded into data/raw/gv/ (git-ignored); the sample
libraries need a (free) account on their sites, so for those the script only prints where to get them and where to
unpack them. Wikimedia Commons recordings (CC BY-SA / public domain) and the Ichiyanagi SoundFont (CC BY-NC 4.0)
are already in eval_data/ with their licenses.

    python -m scripts.data.fetch_gv_data                 # classtab + YouTube, then the library checklist
    python -m scripts.data.fetch_gv_data --only classtab

YouTube needs `pip install yt-dlp imageio-ffmpeg` (ffmpeg converts to the 22050 Hz mono WAV the tests used).
"""
import argparse
import io
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from scripts.eval.gv.paths import CLASSTAB, LIBRARIES, YOUTUBE

CLASSTAB_ZIP = "https://www.classtab.org/zip/classtab.zip"
# (classtab piece, YouTube id): the Test F set; file name <piece>__<id>.wav is what pairs_youtube.tsv expects
VIDEOS = [
    ("carcassi_op60_no01_study_in_c", "4yd1CA3VsjM"), ("carcassi_op60_no07_study_in_am", "RRAVzEeXEX4"),
    ("sor_op31_no01_in_c_andante", "sPaWdh1Ic_A"), ("sor_op35_no19_moderato_in_c", "PBnGRg2Nzpw"),
    ("sor_op35_no19_moderato_in_c", "j7AngJtbHbY"), ("tarrega_estudio_en_forma_de_minuetto", "b9kTKz2t0ko"),
    ("tarrega_lagrima", "DwME8Xp9WxA"), ("tarrega_lagrima", "GQSm36DkMCs"), ("tarrega_lagrima", "Mlrj7mNKuYE"),
    ("tarrega_marieta_mazurka", "O6saw4NqrvI"), ("tarrega_marieta_mazurka", "PDg3kbx3lFM"),
    ("tarrega_marieta_mazurka", "lk39KR-Kcl8"), ("tarrega_recuerdos_de_la_alhambra", "fwjX-m4LkYk"),
]
# name, where to get it, what test_b_resynth looks for under data/raw/gv/libraries/
LIBRARY_CHECKLIST = [
    ("MG Soft Nylon Guitar (Lite) (Mike Georgiades)", "https://www.pianobook.co.uk/packs/mg-soft-nylon-guitar-lite/",
     "unpacked folder with MG_NylonGuitar*.wav"),
    ("RJS LoFi Nylon v1.1 (Risto Sipola)", "https://www.pianobook.co.uk/packs/rjs-lofi-nylon/",
     "unpacked folder with Nylon_*.wav"),
    ("MDM Classical Guitar v1.01 (Mica de Mijatovic y Senic)",
     "https://www.pianobook.co.uk/packs/mdm-classical-guitar-v1-01/",
     "unpacked folder 'MDM Classical Guitar v1.01' incl. its .sfz"),
    ("Custom Classical Guitar", "https://www.polyphone.io/en/soundfonts/guitars/606-custom-classical-guitar",
     "'Custom Classical Guitar.sf2'"),
]


def fetch_classtab() -> None:
    CLASSTAB.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CLASSTAB_ZIP) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    # the archive nests everything under classtab/; flatten so pairs files can name data/raw/gv/classtab/<piece>.txt
    for info in z.infolist():
        name = Path(info.filename).name
        if name.lower().endswith((".txt", ".mid")):
            (CLASSTAB / name).write_bytes(z.read(info))
    print(f"classtab: {len(list(CLASSTAB.glob('*.txt')))} tabs in {CLASSTAB}")


def ffmpeg_args() -> list[str]:
    try:
        import imageio_ffmpeg
    except ImportError:
        return []  # fall back to an ffmpeg on PATH
    return ["--ffmpeg-location", imageio_ffmpeg.get_ffmpeg_exe()]


def fetch_youtube() -> None:
    YOUTUBE.mkdir(parents=True, exist_ok=True)
    for piece, vid in VIDEOS:
        target = YOUTUBE / f"{piece}__{vid}.wav"
        if target.exists():
            continue
        cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-playlist", "-f", "bestaudio", "-x",
               "--audio-format", "wav", "--postprocessor-args", "ffmpeg:-ac 1 -ar 22050", *ffmpeg_args(),
               "-o", str(YOUTUBE / f"{piece}__{vid}.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"]
        ok = subprocess.run(cmd).returncode == 0
        print(f"youtube {vid}: {'ok' if ok else 'FAILED (video removed or yt-dlp outdated?)'}")


def library_checklist() -> None:
    print(f"\nSample libraries (manual download, unpack into {LIBRARIES}):")
    for name, source, expected in LIBRARY_CHECKLIST:
        print(f"  - {name}\n      from: {source}\n      expect: {expected}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["classtab", "youtube", "libraries"])
    only = ap.parse_args().only
    if only in (None, "classtab"):
        fetch_classtab()
    if only in (None, "youtube"):
        fetch_youtube()
    if only in (None, "libraries"):
        library_checklist()


if __name__ == "__main__":
    main()
