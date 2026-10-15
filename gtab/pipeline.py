"""
Katman 5 Adım 2 — Uçtan uca transkripsiyon: CQT -> notalar (deep2) -> tel/perde -> vuruşlar (yarı otomatik) -> konumlar (1q)
-> ölçüler. Komut satırı (scripts/app/transcribe.py) ve ölçüm (scripts/eval/eval_score.py) aynı kodu kullanır.

Tel/perde iki mod (karıştırılmaz, README14): "ses" = duyulan konum (deep2 tel olasılıkları + öğrenilen geçiş modeli,
kafes Viterbi), "edisyon" = klasik edisyon önerisi (EditionNet: konum + sol el parmağı).
Ritim yarı otomatik (README15 Adım 1b): kullanıcı BPM (vuruş birimine göre) + ölçü imi verir; DBN o ölçüde, BPM ± USER_TOL
aralığında rubato'yu sesten (BeatNet) izler; ayarlar gaps_val'de seçildi (README15 B3). Tap modu (README17): kullanıcının
vuruş dokunuşları -> vuruşlar (gtab/decoding/taps.py). analyze (ses, bir kez) + rhythm (vuruş kaynağına göre) = transcribe.
"""

import numpy as np
import torch

from gtab.config import FRAME_RATE
from gtab.core.instrument import STANDARD_6
from gtab.data.torch_dataset import _normalize
from gtab.decoding.beat_dbn import dbn_decode
from gtab.decoding.edition import order_notes
from gtab.decoding.taps import meter_phase, taps_to_beats
from gtab.decoding.transitions import TransitionModel, build_lattice, decode_lattice
from gtab.decoding.viterbi import energy_rise, segment
from gtab.evaluation.pitch_eval import load_selection
from gtab.models.beat_net import activations, encode_features, load_beat_net
from gtab.models.edition_net import EditionNet
from gtab.models.edition_net import predict as edition_predict
from gtab.models.inference import load_model, predict_heads
from gtab.models.rhythm_net import load_rhythm_net
from gtab.models.rhythm_net import predict as rhythm_predict
from gtab.paths import ckpt_path
from gtab.render.score import build_measures, meter_of_signature

INSTR = STANDARD_6
USER_TOL, USER_TRANS = 0.35, 10.0                    # README15 B3 (gaps_val uçtan uca IOI seçimi)


def load_models(device, ckpt="tabcrnn_deep2.pt", cal="checkpoints/tabcrnn_deep2.cal.json", qnet="rhythm_q.pt",
                beatnet="beat_net.pt", edition="edition_seq.pt", domain="gaps_val"):
    """Tüm ağlar + çözümleme seçimi (domain: 'gaps_val' = klasik gitar, 'val' = GuitarSet)."""
    model, _, _ = load_model("crnn", device, ckpt)
    q, lam = load_rhythm_net(qnet, device)
    b, dbn = load_beat_net(beatnet, device)
    ed = EditionNet().to(device)
    ed.load_state_dict(torch.load(ckpt_path(edition), map_location=device)["model"])
    tm = TransitionModel.load(ckpt_path("transitions.npz"), INSTR)
    return dict(model=model, sel=load_selection(cal)[domain], qnet=q, lam=lam, beatnet=b, dbn=dbn, edition=ed.eval(),
                tm=tm, device=device)


def analyze(cqt, m, mode="ses"):
    """Ses aşaması (bir kez): -> dict(notes (N, 3) sn, string, fret, finger (N,) (-1 = yok), act (T, 3) vuruş aktivasyonu)."""
    dev = m["device"]
    _, thr, ot, rules = m["sel"]
    x = _normalize(cqt)
    probs, ons, offs, pons = predict_heads(m["model"], x, dev, with_pitch_onset=True)
    segs = segment(probs, ons, thr, ot, INSTR, rise=energy_rise(cqt, INSTR), offsets=offs,
                   pitch_onsets=pons if rules.get("onset_source") == "pitch" else None, **rules)
    notes = np.array([(a / FRAME_RATE, b / FRAME_RATE, p) for a, b, p in segs], float).reshape(-1, 3)
    N = len(notes)
    string, fret, finger = np.full(N, -1), np.full(N, -1), np.full(N, -1)
    if N and mode == "ses":
        for i, (s, f) in enumerate(decode_lattice(build_lattice(segs, probs, m["tm"]), 1.0, 1.0)):
            if s is not None:
                string[i], fret[i] = s, f
    elif N and mode == "edisyon":
        o, chord = order_notes(notes)
        s, f, g = edition_predict(m["edition"], notes[o], chord, dev)
        string[o], fret[o], finger[o] = s, f, np.where(s >= 0, g, -1)
    elif N:
        raise ValueError(mode)
    act = activations(m["beatnet"], encode_features(m["model"], x, dev), dev)
    return dict(notes=notes, string=string, fret=fret, finger=finger, act=act)


def rhythm(an, m, sig, bpm=None, taps=None, tol=USER_TOL, trans=USER_TRANS, shift=0):
    """
    Ritim aşaması: vuruşlar -> 1q konumları -> ölçüler. Vuruş kaynağı: taps (sn; tap modu, ölçü başı = ilk dokunuş,
    README17) ya da bpm (yarı otomatik: DBN verilen ölçüde, BPM ± tol). shift: ölçü çizgilerini k vuruş ileri kaydır.
    -> dict(beats, downbeats (sn), q (N,) vuruş konumu, measures, first_beat, sig, beat_q, downbeat_index).
    """
    M, beat_q = meter_of_signature(sig)
    if taps is not None:
        taps = np.asarray(taps, float)
        if len(an["notes"]):                                       # müzikten önceki dokunuşlar (sayım, deneme) atılır
            taps = taps[taps >= an["notes"][:, 0].min() - 0.15]
        beats, downs = meter_phase(taps_to_beats(taps, an["act"]), an["act"], M)
        d0 = int(np.searchsorted(beats, downs[0])) if len(downs) else 0
    else:
        beats, downs, _ = dbn_decode(an["act"], meters=(M,), min_bpm=bpm / (1 + tol), max_bpm=bpm * (1 + tol),
                                     trans_lambda=trans, obs_lambda=m["dbn"].get("obs_lambda", 16))
        d0 = int(np.searchsorted(beats, downs[0])) if len(downs) else 0
    q = rhythm_predict(m["qnet"], an["notes"], beats, m["device"], m["lam"])
    measures, first = build_measures(q, d0 + shift, M, beat_q)
    return dict(beats=beats, downbeats=downs, q=q, measures=measures, first_beat=first, sig=sig, beat_q=beat_q,
                downbeat_index=d0 + shift)


def transcribe(cqt, m, bpm, sig, mode="ses", tol=USER_TOL, trans=USER_TRANS, shift=0, taps=None):
    """
    cqt (T, n_bins) dB, m = load_models(...), bpm = vuruş/dk (vuruş birimi: x/4 çeyrek, x/2 ikilik, bileşik x/8 noktalı
    çeyrek; taps verilirse kullanılmaz), sig = (pay, payda) -> analyze + rhythm sözlükleri birleşik.
    """
    an = analyze(cqt, m, mode)
    return dict(an, **rhythm(an, m, sig, bpm, taps, tol, trans, shift))
