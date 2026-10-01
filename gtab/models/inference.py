"""
Model yükleme ve çıkarım (tüm değerlendirme script'lerinin ortak noktası).

- load_model   : 'cnn' | 'crnn' -> (model, checkpoint sözlüğü, tür)
- predict_probs: normalize CQT (T, n_bins) -> (T, S, ncls) softmax
                 CNN: batch'li pencere çıkarımı; CRNN: uzun kayıtlarda parça parça
"""

import numpy as np
import torch

from gtab.core.instrument import STANDARD_6
from gtab.data.torch_dataset import _window
from gtab.models.nets import TabCNN, TabCRNN
from gtab.paths import ckpt_path

DEFAULT_CKPT = {"cnn": "tabcnn.pt", "crnn": "tabcrnn.pt"}


def load_model(kind, device, ckpt=None, instrument=STANDARD_6):
    """ckpt=None -> checkpoints/tabcnn.pt ya da checkpoints/tabcrnn.pt."""
    path = ckpt_path(ckpt or DEFAULT_CKPT[kind])
    ck = torch.load(path, map_location=device)
    if kind == "crnn":
        m = TabCRNN(instrument.num_strings, ck["n_classes"]).to(device)
    else:
        m = TabCNN(instrument.num_strings, ck["n_classes"], ck["context"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck, kind


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
