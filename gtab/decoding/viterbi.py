"""
Katman 3.6 — Viterbi tel ataması (string assignment).

TEŞHİS: pitch F1 ~0.83 (perde doğru) ama tab F1 ~0.58 (tel yanlış). Yani asıl
kayıp telde. Karışık mikrofon sesinde tel kimliği fiziksel olarak zayıf bir
sinyal; onu sadece akustikten zorlamak yerine, GİTARİSTİN KISITINI kullanırız:
el rastgele zıplamaz, ardışık notalar birbirine yakın pozisyonlarda çalınır.

YAKLAŞIM (iki sinyali BİRLEŞTİRİR, kuralla değiştirmez):
  emisyon  = modelin o (tel, fret) için verdiği log-olasılık   [öğrenilen]
  geçiş    = el hareketi maliyeti (fret mesafesi, tel atlama)  [müzikal önsel]
  Viterbi  = toplam skoru maksimize eden ÇALINABİLİR yolu bulur

Bu yüzden model güçlendikçe (veri zenginleştikçe) emisyon ağırlığı artırılarak
kurallar geri çekilebilir -- `w_emission` tam bunun için var. Yani bu çözüm
"şimdilik idare etsin" değil; modelin gelişimiyle birlikte ölçeklenir.

Retrain GEREKTİRMEZ: mevcut model çıktıları (softmax olasılıkları) üstünde çalışır.
"""

import numpy as np
from typing import List, Tuple

from gtab.core.note_event import NoteEvent, Transcription
from gtab.core.instrument import Instrument, STANDARD_6
from gtab.config import FRAME_RATE


# ---------------------------------------------------------------- perde matrisi
def pitch_matrix(probs: np.ndarray, instrument: Instrument = STANDARD_6):
    """
    probs: (T, S, ncls) softmax. -> (pitch_mat (T, P), pitch_lo)
    Her perde için: o perdeyi üretebilen TÜM (tel, fret) pozisyonları arasından
    EN YÜKSEK olasılık. Teli marjinalize eder; geriye güçlü perde sinyali kalır.
    """
    T, S, ncls = probs.shape
    lo, hi = instrument.pitch_range()
    P = hi - lo + 1
    out = np.zeros((T, P), dtype=np.float32)
    for p in range(P):
        pitch = lo + p
        for (s, f) in instrument.pitch_to_positions(pitch):
            if f + 1 < ncls:
                np.maximum(out[:, p], probs[:, s, f + 1], out=out[:, p])
    return out, lo


def segment_notes(pitch_mat: np.ndarray, pitch_lo: int, threshold=0.5,
                  frame_rate: float = FRAME_RATE, min_frames: int = 2
                  ) -> List[Tuple[int, int, int]]:
    """
    (T,P) perde matrisi -> [(start_frame, end_frame, pitch), ...]
    Ardışık aktif kareleri nota yapar.
    """
    active = pitch_mat > threshold
    T, P = active.shape
    notes = []
    for p in range(P):
        t = 0
        while t < T:
            if active[t, p]:
                s = t
                while t < T and active[t, p]:
                    t += 1
                if t - s >= min_frames:
                    notes.append((s, t, pitch_lo + p))
            else:
                t += 1
    notes.sort(key=lambda n: (n[0], n[2]))
    return notes


# ---------------------------------------------------------------- maliyetler
def transition_cost(prev, cur, gap_sec, w_fret=1.0, w_string=0.35,
                    open_discount=0.3, gap_relax=0.5):
    """
    İki ardışık nota pozisyonu arasındaki EL HAREKETİ maliyeti.
    prev/cur: (string, fret)
    - fret mesafesi ana maliyet (el kaydırmak zor)
    - tel atlama daha ucuz (el aynı pozisyonda kalır)
    - açık tel (fret 0) eli bağlamaz ama BEDAVA DEĞİL: open_discount ile indirimli.
      DİKKAT: bedava yapılırsa "hep açık tel" yolu dejenere bir çekim merkezi olur
      ve modelin emisyon sinyalini tamamen ezer (test edilip görüldü).
    - aradaki süre uzunsa elin taşınacak zamanı var -> maliyet azalır
    """
    ps, pf = prev
    cs, cf = cur
    fret_cost = abs(pf - cf)
    if pf == 0 or cf == 0:
        fret_cost *= open_discount
    cost = w_fret * fret_cost + w_string * abs(ps - cs)
    # zaman gevşemesi: uzun boşlukta el rahat taşınır
    cost /= (1.0 + gap_relax * max(0.0, gap_sec))
    return cost


def emission_logprob(probs, start, end, s, f, eps=1e-8):
    """Nota süresince modelin (tel s, fret f) için ortalama log-olasılığı."""
    return float(np.log(probs[start:end, s, f + 1] + eps).mean())


# ---------------------------------------------------------------- Viterbi
def viterbi_assign(notes, probs, instrument: Instrument = STANDARD_6,
                   frame_rate: float = FRAME_RATE,
                   w_emission=1.0, w_transition=0.6,
                   w_fret=1.0, w_string=0.35, open_discount=0.3, gap_relax=0.5,
                   high_fret_penalty=0.02):
    """
    notes: [(start, end, pitch)] zaman sırasına göre.
    -> her nota için seçilen (string, fret) listesi.

    Skor = w_emission * Σ emisyon  -  w_transition * Σ geçiş maliyeti
    high_fret_penalty: eşit koşulda daha düşük pozisyonu tercih (çok küçük tutulur).
    """
    n = len(notes)
    if n == 0:
        return []

    cands = [instrument.pitch_to_positions(p) for (_, _, p) in notes]

    # ilk nota
    prev_scores = []
    for (s, f) in cands[0]:
        start, end, _ = notes[0]
        sc = w_emission * emission_logprob(probs, start, end, s, f) \
             - high_fret_penalty * f
        prev_scores.append(sc)
    prev_scores = np.array(prev_scores, dtype=np.float64) if cands[0] else np.array([])
    backptr = []

    for i in range(1, n):
        start, end, _ = notes[i]
        prev_start, prev_end, _ = notes[i - 1]
        gap_sec = max(0.0, (start - prev_end) / frame_rate)

        cur = cands[i]
        if not cur or prev_scores.size == 0:
            # çalınamayan perde (enstrüman aralığı dışı) -> zinciri kır
            backptr.append([-1] * max(1, len(cur)))
            prev_scores = np.array([
                w_emission * emission_logprob(probs, start, end, s, f) - high_fret_penalty * f
                for (s, f) in cur
            ], dtype=np.float64) if cur else np.array([])
            continue

        scores = np.empty(len(cur), dtype=np.float64)
        ptr = np.empty(len(cur), dtype=np.int64)
        for j, (s, f) in enumerate(cur):
            emis = w_emission * emission_logprob(probs, start, end, s, f) \
                   - high_fret_penalty * f
            trans = np.array([
                transition_cost(prev_pos, (s, f), gap_sec, w_fret, w_string,
                                open_discount=open_discount, gap_relax=gap_relax)
                for prev_pos in cands[i - 1]
            ], dtype=np.float64)
            total = prev_scores - w_transition * trans
            k = int(total.argmax())
            scores[j] = total[k] + emis
            ptr[j] = k
        backptr.append(ptr.tolist())
        prev_scores = scores

    # geri izleme
    path = [0] * n
    if prev_scores.size:
        path[n - 1] = int(np.argmax(prev_scores))
    for i in range(n - 1, 0, -1):
        k = backptr[i - 1][path[i]] if path[i] < len(backptr[i - 1]) else -1
        path[i - 1] = k if k >= 0 else 0

    out = []
    for i, idx in enumerate(path):
        out.append(cands[i][idx] if cands[i] else (None, None))
    return out


# ---------------------------------------------------------------- uçtan uca
def viterbi_transcribe(probs, instrument: Instrument = STANDARD_6,
                      threshold=0.5, frame_rate: float = FRAME_RATE,
                      min_frames=2, **vit_kwargs) -> Transcription:
    """probs (T,S,ncls) -> Transcription (string/fret Viterbi ile atanmış)."""
    pm, lo = pitch_matrix(probs, instrument)
    segs = segment_notes(pm, lo, threshold, frame_rate, min_frames)
    assigns = viterbi_assign(segs, probs, instrument, frame_rate, **vit_kwargs)

    events = []
    for (start, end, pitch), (s, f) in zip(segs, assigns):
        ev = NoteEvent(start / frame_rate, end / frame_rate, pitch)
        ev.string, ev.fret = s, f
        events.append(ev)
    return Transcription(notes=events).sort()


def transcription_to_frames(tr: Transcription, n_frames: int,
                            instrument: Instrument = STANDARD_6,
                            frame_rate: float = FRAME_RATE) -> np.ndarray:
    """
    Transcription -> (T, S) sınıf dizisi (0=sessiz, fret+1).
    Mevcut tab F1 metriğiyle AYNI temelde karşılaştırabilmek için gerekli.
    """
    S = instrument.num_strings
    out = np.zeros((n_frames, S), dtype=np.int64)
    for n in tr.notes:
        if not n.has_position():
            continue
        a = max(0, int(round(n.onset * frame_rate)))
        b = min(n_frames, int(round(n.offset * frame_rate)))
        if b > a:
            out[a:b, n.string] = n.fret + 1
    return out


# ---------------------------------------------------------------- Katman 3.9: onset
def pitch_onset_matrix(probs, onsets, instrument: Instrument = STANDARD_6):
    """
    probs (T,S,ncls), onsets (T,S) tel onset olasılığı -> (T,P) perde-onset matrisi.
    Her perde için: o perdeyi çalabilen (tel, fret) pozisyonlarında
    max(onset_tel · P(tel, fret)) -- pitch_matrix ile aynı marjinalizasyon.
    """
    T, S, ncls = probs.shape
    lo, hi = instrument.pitch_range()
    out = np.zeros((T, hi - lo + 1), dtype=np.float32)
    for p in range(hi - lo + 1):
        for (s, f) in instrument.pitch_to_positions(lo + p):
            if f + 1 < ncls:
                np.maximum(out[:, p], onsets[:, s] * probs[:, s, f + 1], out=out[:, p])
    return out


def segment_notes_onset(pitch_mat, onset_mat, pitch_lo, threshold=0.5, onset_threshold=0.5,
                        min_frames=2, lookahead=2, off_ratio=1.0, refractory=0, fallback=0,
                        peak=False, reattack=0.0, rise=None, rise_keep=0.0, rise_split=0.0,
                        offset_mat=None, offset_threshold=0.0):
    """
    Onsets & Frames kuralı:
      - Bir nota YALNIZCA onset ile başlar (onset_mat > onset_threshold, yükselen kenar).
        Onset'siz aktif kareler (hayalet nota kaynağı) atılır.
      - Onset, perde 'lookahead' kare içinde aktifleşirse kabul edilir.
      - Nota, perde aktif kaldıkça sürer; AYNI perdede yeni onset gelirse biter ve
        yeni nota başlar (tekrar eden notalar artık birleşmez).
    Katman 3.10 eklentileri (varsayılanlar = 3.9 davranışı, birebir):
      - off_ratio  < 1: HİSTEREZİS. Nota 'threshold' ile başlar, threshold·off_ratio
                       üstünde kaldıkça sürer -> sönümlenen kuyruk kesilmez.
      - refractory > 0: aynı perdede önceki nota hâlâ sürerken 'refractory' kareden
                       kısa sürede gelen yeni onset yok sayılır -> çift tetik hayaletleri.
      - fallback   > 0: hiçbir notanın kapsamadığı, 'fallback' kare boyunca 'threshold'
                       üstünde kalan perde onset'siz de nota sayılır -> onset kaçınca
                       notanın tamamen düşmesi.
      - peak=True    : nota, onset eşiği üstündeki koşunun TEPE karesinden başlar
                       (yükselen kenardan değil) -> başlangıç zamanlaması (Adım 2).
      - reattack > 0 : aynı perde HÂLÂ ÇALARKEN gelen yeni onset, ancak tepe değeri
                       >= reattack ise yeni nota başlatır; değilse nota sürer
                       -> nota parçalanması (Adım 3a: akor hayaletlerinin ~%78'i).
      - rise (T,P) dB: perde başına CQT enerji yükselişi (energy_rise). Gerçek bir yeniden
        vuruş, önceki nota sönmeden de enerji sıçraması üretir; sahte bölünme üretmez.
      - rise_keep  > 0 : aynı perde çalarken gelen onset, yükseliş >= rise_keep dB ise kabul
                         (reattack ile birlikteyse ikisinden biri yeter).
      - rise_split > 0 : süren notanın içinde onset eşiği AŞILMASA BİLE (onset >= eşik/2)
                         yükseliş >= rise_split dB olan yerel tepe yeni nota başlatır
                         -> önceki notaya gömülen hızlı tekrarlar (Adım 3 revizyonu).
      - offset_mat (T,P) + offset_threshold > 0 (Adım 3c): nota, offset olasılığı eşiği aştığı
                         karede biter (perde hâlâ aktif olsa bile) -> kuyruk/sonraki vuruş ayrımı.
    -> [(start, end, pitch)] zaman sırasına göre (segment_notes ile aynı biçim).
    """
    active = pitch_mat > threshold
    cont = pitch_mat > threshold * off_ratio if off_ratio < 1.0 else active
    on = onset_mat > onset_threshold
    T, P = active.shape
    starts = on & ~np.vstack([np.zeros((1, P), bool), on[:-1]])      # yükselen kenarlar
    use_off = offset_mat is not None and offset_threshold > 0
    notes = []
    for p in range(P):
        st = list(np.nonzero(starts[:, p])[0])
        use_rise = rise is not None and (rise_keep > 0 or rise_split > 0)
        rmax = (lambda t: float(rise[max(0, t - 1):t + 3, p].max())) if use_rise else None
        if (peak or reattack > 0 or (use_rise and rise_keep > 0)) and st:
            adj, pkv = [], {}
            for s_ in st:
                e_ = s_
                while e_ < T and on[e_, p]:
                    e_ += 1
                k_ = s_ + int(np.argmax(onset_mat[s_:e_, p])) if peak else s_
                adj.append(k_); pkv[k_] = float(onset_mat[s_:e_, p].max())
            st = adj
            keep_r = rise_keep if use_rise else 0.0
            if reattack > 0 or keep_r > 0:        # nota sürerken zayıf yeniden vuruşu yok say
                st = [s_ for i_, s_ in enumerate(st)
                      if i_ == 0 or s_ == 0 or not cont[s_ - 1, p]
                      or (reattack > 0 and pkv[s_] >= reattack)
                      or (keep_r > 0 and rmax(s_) >= keep_r)]
        if refractory > 0 and len(st) > 1:
            kept = [st[0]]
            for s_ in st[1:]:
                if s_ - kept[-1] < refractory and cont[kept[-1]:s_ + 1, p].all():
                    continue                      # önceki nota sürüyor: çift tetik
                kept.append(s_)
            st = kept
        if use_rise and rise_split > 0:           # gömülü hızlı tekrarları enerji kanıtıyla ayır
            r = rise[:, p]
            cand = np.nonzero((r >= rise_split) & cont[:, p]
                              & np.concatenate([[False], cont[:-1, p]])
                              & (onset_mat[:, p] >= onset_threshold * 0.5)
                              & (r >= np.concatenate([[0], r[:-1]]))
                              & (r >= np.concatenate([r[1:], [0]])))[0]
            have = np.array(st, int)
            extra = [int(t) for t in cand if not len(have) or np.abs(have - t).min() > 3]
            if extra:
                st = sorted(st + extra)
        covered = np.zeros(T, bool) if fallback > 0 else None
        for k, s0 in enumerate(st):
            nxt = st[k + 1] if k + 1 < len(st) else T
            a = s0
            while a < min(s0 + lookahead + 1, nxt) and not active[a, p]:
                a += 1
            if a >= nxt or a >= T or not active[a, p]:
                continue
            b = a
            while b < nxt and cont[b, p]:
                b += 1
                if use_off and b - s0 >= min_frames and offset_mat[b - 1, p] >= offset_threshold:
                    break                         # offset kafası: nota bu karede bitiyor
            if b - s0 >= min_frames:
                notes.append((int(s0), int(b), pitch_lo + p))
                if covered is not None:
                    covered[s0:b] = True
        if fallback > 0:
            t = 0
            while t < T:
                if active[t, p] and not covered[t]:
                    s_ = t
                    while t < T and cont[t, p] and not covered[t]:
                        t += 1
                    if t - s_ >= fallback:
                        notes.append((int(s_), int(t), pitch_lo + p))
                else:
                    t += 1
    notes.sort(key=lambda n: (n[0], n[2]))
    return notes


def energy_rise(cqt_db, instrument: Instrument = STANDARD_6, k: int = 3,
                harmonics=(0, 24, 38), bins_per_semitone: int = 2, fmin_midi: int = 24):
    """
    Katman 3.10 Adım 3 revizyonu — CQT (T, n_bins, dB) -> (T, P) perde başına enerji yükselişi (dB).
    E_p(t) = perdenin temel + 2. + 3. harmonik binlerindeki (±1 bin) en yüksek enerjinin ortalaması;
    rise(t) = E_p(t) - min(E_p(t-k .. t-1)), negatifler 0. Model değil, doğrudan sesin kanıtı.
    """
    c = np.asarray(cqt_db, dtype=np.float32)
    T, F = c.shape
    lo, hi = instrument.pitch_range()
    E = np.zeros((T, hi - lo + 1), np.float32)
    for i, p in enumerate(range(lo, hi + 1)):
        b = bins_per_semitone * (p - fmin_midi)
        vals = [c[:, max(0, b + h - 1):min(F, b + h + 2)].max(1) for h in harmonics if b + h < F]
        E[:, i] = np.mean(vals, axis=0)
    prev = np.full_like(E, np.inf)
    for j in range(1, k + 1):
        sh = np.full_like(E, np.inf); sh[j:] = E[:-j]
        prev = np.minimum(prev, sh)
    return np.maximum(E - np.where(np.isinf(prev), E, prev), 0.0)


def segment(probs, onsets=None, threshold=0.5, onset_threshold=0.5,
            instrument: Instrument = STANDARD_6, min_frames: int = 2, rise=None, offsets=None, **dec):
    """
    Tek giriş noktası: model çıktısı -> [(start, end, pitch)].
    onsets None ise eski kare-eşik segmentasyonu (Katman 3.6), değilse onset tabanlı (3.9).
    dec: segment_notes_onset'in Katman 3.10 parametreleri (off_ratio, refractory, fallback).
    """
    pm, lo = pitch_matrix(probs, instrument)
    if onsets is None or onset_threshold is None:
        return segment_notes(pm, lo, threshold, min_frames=min_frames)
    om = pitch_onset_matrix(probs, onsets, instrument)
    if offsets is not None and dec.get("offset_threshold", 0) > 0:     # Adım 3c: offset kafası
        dec = dict(dec, offset_mat=pitch_onset_matrix(probs, offsets, instrument))
    else:
        dec = {k: v for k, v in dec.items() if k != "offset_threshold"}
    return segment_notes_onset(pm, om, lo, threshold, onset_threshold, min_frames=min_frames,
                               rise=rise, **dec)
