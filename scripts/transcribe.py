"""
WAV -> TAB: tek komutla uçtan uca transkripsiyon (mevcut bileşenleri BAĞLAR, yeniden yazmaz).

Zincir (hepsi var olan kod):
    ses dosyası
      -> gtab.data.features.cqt_from_file            (22050 Hz mono, log-CQT, hop 512)
      -> gtab.models.inference.load_model/predict_heads   (checkpoints/tabcrnn_deep2.pt)
      -> gtab.decoding.viterbi.energy_rise + segment      (kalibre çözümleme: .cal.json)
      -> gtab.decoding.transitions.build_lattice/decode_lattice   (tel ataması, transitions.npz)
      -> gtab.core.note_event.NoteEvent / Transcription
      -> gtab.decoding.decode.render_ascii_tab            (6 telli ASCII TAB)

Kalibrasyon `checkpoints/tabcrnn_deep2.cal.json` içinden alınır; **yeni seçim yapılmaz**:
  --domain gaps_val  (varsayılan) klasik gitar (naylon tel) için doğrulanmış ayar
  --domain val                    GuitarSet (çelik telli akustik) için doğrulanmış ayar
Alan seçimi çıktıda yazılır; iki ayarın sayıları birbirinin yerine kullanılamaz.

`--fingering` (seçimlik): `checkpoints/edition_seq.pt` ile sol el parmağı eklenir. Parmak modeli
konumu **değiştirmez**: tel/perde yukarıdaki ses zincirinden gelir ve modele `fixed` olarak
verilir (eval_edition'daki "partisyon konumu verilerek" ölçümüyle aynı kullanım). Modelin kendi
konum tercihi yalnız BİLGİ olarak sayılır, uygulanmaz.

Çalıştırma:
    python -m scripts.transcribe --audio "ornek.wav"
    python -m scripts.transcribe --audio "ornek.wav" --domain val --fingering
    python -m scripts.transcribe --audio "ornek.wav" --output tab.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.core.note_event import NoteEvent, Transcription
from gtab.paths import ckpt_path

INSTR = STANDARD_6
STRING_NAMES = ("E", "A", "D", "G", "B", "e")
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DEFAULT_CKPT = "tabcrnn_deep2.pt"
DEFAULT_CAL = "tabcrnn_deep2.cal.json"
DOMAINS = {"gaps_val": "klasik gitar (GAPS doğrulamasında seçildi)",
           "val": "çelik telli akustik (GuitarSet oyuncu 05 doğrulamasında seçildi)"}


class TranscribeError(Exception):
    """Kullanıcıya gösterilecek, açık nedenli hata (sahte çıktı üretmek yerine durur)."""


def note_name(midi: int) -> str:
    """MIDI numarası -> nota adı (ör. 64 -> E4)."""
    return f"{PITCH_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


def resolve_audio(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise TranscribeError(f"ses dosyasi bulunamadi: {p}")
    if p.is_dir():
        raise TranscribeError(f"dizin verildi, dosya bekleniyor: {p}")
    if p.stat().st_size == 0:
        raise TranscribeError(f"ses dosyasi bos: {p}")
    return p


def resolve_file(name: str, what: str) -> Path:
    """Checkpoint / kalibrasyon yolu (çıplak ad -> checkpoints/ altında)."""
    p = Path(ckpt_path(name))
    if not p.exists():
        raise TranscribeError(f"{what} bulunamadi: {p}")
    return p


def load_calibration(cal_path: Path, domain: str) -> Tuple[float, float, Dict]:
    """
    `.cal.json` -> (perde eşiği, onset eşiği, çözümleme kuralları). Yeni arama YAPILMAZ.
    """
    from gtab.evaluation.pitch_eval import load_selection
    sel = load_selection(str(cal_path))
    if domain not in sel:
        raise TranscribeError(f"kalibrasyonda '{domain}' yok; mevcut: {sorted(sel)}")
    _, thr, onset_thr, dec = sel[domain]
    return float(thr), float(onset_thr), dict(dec)


def build_events(segs: Sequence[Tuple[int, int, int]], assign: Sequence[Tuple],
                 frame_rate: float = FRAME_RATE) -> Transcription:
    """(kare başı, kare sonu, perde) + (tel, perde) -> Transcription (saniye)."""
    notes = []
    for (a, b, pitch), sf in zip(segs, assign):
        ev = NoteEvent(a / frame_rate, b / frame_rate, int(pitch))
        if sf is not None and sf[0] is not None:
            ev.string, ev.fret = int(sf[0]), int(sf[1])
        notes.append(ev)
    return Transcription(notes=notes).sort()


def transcribe_cqt(cqt: np.ndarray, ckpt: Path, thr: float, onset_thr: float, dec: Dict,
                   w_transition: float, device) -> Transcription:
    """CQT -> Transcription. Model, çözümleme ve tel ataması mevcut kodla yapılır."""
    from gtab.data.torch_dataset import _normalize
    from gtab.decoding.transitions import TransitionModel, build_lattice, decode_lattice
    from gtab.decoding.viterbi import energy_rise, segment
    from gtab.models.inference import load_model, predict_heads
    model, _, _ = load_model("crnn", device, str(ckpt))
    probs, ons, offs, pons = predict_heads(model, _normalize(cqt), device, with_pitch_onset=True)
    segs = segment(probs, ons, thr, onset_thr, INSTR, rise=energy_rise(cqt, INSTR), offsets=offs,
                   pitch_onsets=pons if dec.get("onset_source") == "pitch" else None, **dec)
    if not segs:
        return Transcription(notes=[])
    tm = TransitionModel.load(str(resolve_file("transitions.npz", "gecis modeli")), INSTR)
    asg = decode_lattice(build_lattice(segs, probs, tm), 1.0, w_transition)
    return build_events(segs, asg)


def add_fingering(tr: Transcription, ckpt: Path, device) -> Tuple[List[int], int]:
    """
    Mevcut tel/perde KORUNARAK parmak ekler (`fixed` konum). -> (nota başına parmak, farklı
    konum sayısı). İkinci değer yalnız bilgi: parmak modeli serbest bırakılsa kaç notada başka
    konum seçerdi (uygulanmaz).
    """
    import torch
    from gtab.decoding.edition import order_notes
    from gtab.models.edition_net import EditionNet, predict as ed_predict
    net = EditionNet().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    notes = np.array([[n.onset, n.offset, n.pitch] for n in tr.notes], float)
    order, chord = order_notes(notes)
    fixed = np.array([tr.notes[i].string if tr.notes[i].string is not None else -1 for i in order])
    _, _, fingers = ed_predict(net, notes[order], chord, device, fixed=fixed)
    free_s, _, _ = ed_predict(net, notes[order], chord, device)
    out = [0] * len(tr.notes)
    for k, i in enumerate(order):
        out[i] = int(fingers[k])
    differs = int(sum(1 for k, i in enumerate(order)
                      if tr.notes[i].string is not None and int(free_s[k]) != tr.notes[i].string))
    return out, differs


def note_table(tr: Transcription, fingers: Optional[Sequence[int]] = None, limit: int = 0) -> str:
    """Nota tablosu: başlangıç, bitiş, MIDI, nota adı, tel, perde (+ parmak)."""
    head = f"  {'#':>4} {'baslangic':>10} {'bitis':>8} {'MIDI':>5} {'nota':>5} {'tel':>4} {'perde':>6}"
    rows = [head + ("  parmak" if fingers is not None else ""),
            "  " + "-" * (len(head) + (8 if fingers is not None else 0))]
    shown = tr.notes if limit <= 0 else tr.notes[:limit]
    for k, n in enumerate(shown):
        s = STRING_NAMES[n.string] if n.string is not None else "?"
        f = f"{n.fret}" if n.fret is not None else "?"
        line = (f"  {k + 1:>4} {n.onset:>10.3f} {n.offset:>8.3f} {n.pitch:>5d} "
                f"{note_name(n.pitch):>5} {s:>4} {f:>6}")
        if fingers is not None:
            line += f"  {fingers[k] if k < len(fingers) else '?':>6}"
        rows.append(line)
    if limit > 0 and len(tr.notes) > limit:
        rows.append(f"  ... ({len(tr.notes) - limit} nota daha; tamami icin --limit 0)")
    return "\n".join(rows)


def main(args) -> int:
    from gtab.data.features import cqt_from_file
    from gtab.decoding.decode import render_ascii_tab
    from gtab.utils import get_device
    audio = resolve_audio(args.audio)
    ckpt = resolve_file(args.ckpt, "model checkpoint'i")
    cal = resolve_file(args.cal, "kalibrasyon dosyasi")
    thr, onset_thr, dec = load_calibration(cal, args.domain)
    try:
        cqt = cqt_from_file(str(audio))
    except Exception as e:                      # bozuk/desteklenmeyen ses: sahte cikti uretme
        raise TranscribeError(f"ses okunamadi ({type(e).__name__}): {e}") from None
    dur = cqt.shape[0] / FRAME_RATE
    device = get_device()
    print(f"Ses       : {audio.name}  ({dur:.1f} sn, {cqt.shape[0]} kare)")
    print(f"Model     : {ckpt.name} | cihaz {device}")
    print(f"Kalibrasyon: {cal.name} [{args.domain}] = {DOMAINS[args.domain]}")
    print(f"             perde esigi {thr:.3f} | onset esigi {onset_thr} | kurallar {dec}")
    tr = transcribe_cqt(cqt, ckpt, thr, onset_thr, dec, args.w_transition, device)
    fingers, differs = (None, 0)
    if args.fingering and tr.notes:
        fingers, differs = add_fingering(tr, resolve_file(args.fingering_ckpt, "parmak modeli"),
                                         device)
    print(f"\nTespit edilen nota: {len(tr.notes)}")
    if not tr.notes:
        print("  (nota bulunamadi — esik/alan ayari bu kayit icin uygun olmayabilir)")
    else:
        print(note_table(tr, fingers, args.limit))
    if fingers is not None:
        print(f"\n  Parmak modeli konumu DEGISTIRMEDI (tel/perde ses zincirinden). Bilgi: serbest "
              f"birakilsa {differs}/{len(tr.notes)} notada baska konum secerdi.")
    tab = render_ascii_tab(tr, INSTR, col_seconds=args.col_seconds)
    print(f"\nASCII TAB ({args.col_seconds} sn/kolon):\n")
    print(tab)
    if args.output:
        Path(args.output).write_text(tab + "\n", encoding="utf-8")
        print(f"\nTAB yazildi: {args.output}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WAV -> gitar TAB (mevcut modellerle)")
    ap.add_argument("--audio", required=True, help="ses dosyasi (.wav)")
    ap.add_argument("--domain", default="gaps_val", choices=sorted(DOMAINS),
                    help="kalibrasyon alani (varsayilan: gaps_val = klasik gitar)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--cal", default=DEFAULT_CAL)
    # 0 = greedy, the setting every reported "heard position" number was measured with
    # (eval_hmm selects w_tr=0: tab F1 0.777 vs 0.751 at 1.0; eval_gaps_tab / eval_edition are greedy)
    ap.add_argument("--w-transition", type=float, default=0.0,
                    help="ogrenilen gecis agirligi (0 = greedy, olculen ayar; eski varsayilan 1.0)")
    ap.add_argument("--fingering", action="store_true", help="sol el parmagi ekle (edition_seq.pt)")
    ap.add_argument("--fingering-ckpt", default="edition_seq.pt")
    ap.add_argument("--limit", type=int, default=40, help="nota tablosunda gosterilecek satir (0 = hepsi)")
    ap.add_argument("--col-seconds", type=float, default=0.15, help="ASCII TAB kolon suresi")
    ap.add_argument("--output", default=None, help="TAB'i .txt olarak kaydet (yalniz verilirse)")
    try:
        sys.exit(main(ap.parse_args()))
    except TranscribeError as e:
        print(f"HATA: {e}")
        sys.exit(2)
