"""
Model yükleme ve çıkarım (tüm değerlendirme script'lerinin ortak noktası).

- load_model   : 'cnn' | 'crnn' -> (model, checkpoint sözlüğü, tür)
- predict_probs: normalize CQT (T, n_bins) -> (T, S, ncls) softmax
                 CNN: batch'li pencere çıkarımı; CRNN: uzun kayıtlarda parça parça
- predict_with_onsets: onset kafalı CRNN -> (tab olasılıkları, (T, S) onset olasılıkları)
- predict_heads      : + offset kafası (Katman 3.10 Adım 3c) -> (tab, onset | None, offset | None)

Checkpoint'te "onset": True varsa 'crnn' türü otomatik olarak TabCRNNOnset kurar
(forward() yine tab döndürür -> eski değerlendirme kodu değişmeden çalışır).
"""

import numpy as np
import torch

from gtab.core.instrument import STANDARD_6
from gtab.data.torch_dataset import _window
from gtab.models.nets import TabCNN, TabCRNN, TabCRNNOnset
from gtab.paths import ckpt_path

DEFAULT_CKPT = {"cnn": "tabcnn.pt", "crnn": "tabcrnn.pt"}


def load_model(kind, device, ckpt=None, instrument=STANDARD_6):
    """ckpt=None -> checkpoints/tabcnn.pt ya da checkpoints/tabcrnn.pt."""
    path = ckpt_path(ckpt or DEFAULT_CKPT[kind])
    ck = torch.load(path, map_location=device)
    if kind == "crnn":
        if ck.get("onset"):
            m = TabCRNNOnset(instrument.num_strings, ck["n_classes"], harmonics=ck.get("harmonics"),
                             offset=bool(ck.get("offset")), deep=int(ck.get("deep") or 0),
                             pitch_onset=bool(ck.get("pitch_onset"))).to(device)
        else:
            m = TabCRNN(instrument.num_strings, ck["n_classes"], harmonics=ck.get("harmonics")).to(device)
    else:
        m = TabCNN(instrument.num_strings, ck["n_classes"], ck["context"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck, kind


def _chunked(fn, cqt, device, chunk=2000, ctx=100):
    """fn(x) -> tensör(lar) (1, L, ...). Örtüşmeli pencereler, her pencerenin ortası alınır."""
    T = cqt.shape[0]
    outs = None
    with torch.no_grad():
        for s in range(0, T, chunk):
            a, b = max(0, s - ctx), min(T, s + chunk + ctx)
            x = torch.from_numpy(cqt[a:b]).unsqueeze(0).unsqueeze(0).to(device)
            res = fn(x)
            res = res if isinstance(res, tuple) else (res,)
            if outs is None:
                outs = [np.zeros((T,) + r.shape[2:], np.float32) for r in res]
            e = min(T, s + chunk)
            for o, r in zip(outs, res):
                o[s:e] = r[0].cpu().numpy()[s - a:e - a]
    return outs


def predict_with_onsets(model, cqt, device):
    """Onset kafalı CRNN -> (tab softmax (T,S,C), onset sigmoid (T,S)); kafa yoksa onset=None."""
    if not hasattr(model, "onset_head"):
        return _chunked(lambda x: torch.softmax(model(x), -1), cqt, device)[0], None
    def both(x):
        tab, on = model.forward_both(x)
        return torch.softmax(tab, -1), torch.sigmoid(on)
    tab, on = _chunked(both, cqt, device)
    return tab, on


def predict_heads(model, cqt, device, with_pitch_onset=False):
    """
    Katman 3.10 Adım 3c — tüm kafalar: (tab softmax (T,S,C), onset (T,S) | None, offset (T,S) | None).
    Offset kafası yoksa predict_with_onsets ile birebir aynı.
    with_pitch_onset (Katman 3.12): 4. öğe perde-onset kafası (T,P) | None.
    """
    if with_pitch_onset:
        if getattr(model, "pitch_onset_head", None) is None:
            return tuple(predict_heads(model, cqt, device)) + (None,)
        def fullh(x):
            tab, on, off, pon = model.forward_full(x)
            o = torch.sigmoid(off) if off is not None else torch.zeros_like(on)
            return torch.softmax(tab, -1), torch.sigmoid(on), o, torch.sigmoid(pon)
        tab, on, off, pon = _chunked(fullh, cqt, device)
        return tab, on, (off if model.offset_head is not None else None), pon
    if getattr(model, "offset_head", None) is None:
        tab, on = predict_with_onsets(model, cqt, device)
        return tab, on, None
    def allh(x):
        tab, on, off = model.forward_all(x)
        return torch.softmax(tab, -1), torch.sigmoid(on), torch.sigmoid(off)
    return tuple(_chunked(allh, cqt, device))


def predict_probs(model, ck, kind, cqt, device, batch=1024, chunk=2000, ctx=100):
    """
    CNN : her kare için bağlam penceresi, batch'li.
    CRNN: tüm kayıt; 'chunk' kareden uzunsa örtüşmeli pencerelerle (GPU belleği sabit,
          her pencerenin ortası alınır). Kısa kayıtlarda tek forward ile aynı sonucu verir.
    """
    with torch.no_grad():
        if kind == "crnn":
            T = cqt.shape[0]
            out = None
            for s in range(0, T, chunk):
                a, b = max(0, s - ctx), min(T, s + chunk + ctx)
                x = torch.from_numpy(cqt[a:b]).unsqueeze(0).unsqueeze(0).to(device)
                p = torch.softmax(model(x)[0], dim=-1).cpu().numpy()
                if out is None:
                    out = np.zeros((T,) + p.shape[1:], np.float32)
                e = min(T, s + chunk)
                out[s:e] = p[s - a:e - a]
            return out
        T = cqt.shape[0]; context = ck["context"]
        wins = np.stack([_window(cqt, t, context) for t in range(T)])
        out = None
        for i in range(0, T, batch):
            xb = torch.from_numpy(wins[i:i + batch]).unsqueeze(1).to(device)
            p = torch.softmax(model(xb), dim=-1).cpu().numpy()
            if out is None:
                out = np.zeros((T, p.shape[1], p.shape[2]), np.float32)
            out[i:i + batch] = p
        return out
