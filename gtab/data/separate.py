"""
Katman 6 Adım 2b — Kaynak ayırma (ölçüm aracı; README18). Demucs `htdemucs_6s` (davul, bas, diğer, vokal, gitar, piyano)
ile karışık / şarkılı kayıttan gitarı ayırır. Ürün varsayılanı değil; `eval_robust --separate` ile ölçülür.

Kipler:
    "guitar"      yalnız gitar kanalı
    "mix-vocals"  girdi karışımı eksi vokal kanalı (insan sesi sorununa en doğrudan yol; naylon gitarın "guitar"
                  yerine "other" kanalına düşme riskini taşımaz)

Girdi / çıktı 22050 Hz (proje hızı), uzunluk girdiyle AYNI. Demucs 44100 Hz çalışır (iç yeniden örnekleme).
demucs isteğe bağlı bağımlılıktır (requirements.txt); ilk çağrıda tembel yüklenir, ağırlıklar ilk seferde indirilir.
"""

import librosa
import numpy as np

MODES = ("guitar", "mix-vocals")
_DEMUCS_SR = 44100
_model = {}


def _load(device):
    try:
        import torch  # noqa: F401
        from demucs.pretrained import get_model
    except ImportError as e:
        raise ImportError("demucs kurulu degil: python -m pip install demucs (requirements.txt, istege bagli)") from e
    if "m" not in _model:
        _model["m"] = get_model("htdemucs_6s").to(device).eval()
    return _model["m"]


def separate(y, sr=22050, mode="mix-vocals"):
    """y: tek kanal float dizi (sr Hz) -> aynı uzunlukta, aynı sr'de tek kanal dizi."""
    if mode not in MODES:
        raise ValueError(f"mode {MODES} icinden olmali: {mode!r}")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load(device)
    from demucs.apply import apply_model
    n = len(y)
    x = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=_DEMUCS_SR)
    wav = torch.from_numpy(x)[None, None].repeat(1, 2, 1).to(device)  # (1, 2, T) sahte stereo
    ref = wav.mean(1, keepdim=True)
    std = ref.std() + 1e-8
    with torch.no_grad():
        src = apply_model(model, (wav - ref.mean()) / std, device=device, progress=False)[0] * std + ref.mean()
    names = model.sources
    out = src[names.index("guitar")] if mode == "guitar" else wav[0] - src[names.index("vocals")]
    out = out.mean(0).cpu().numpy()
    out = librosa.resample(out, orig_sr=_DEMUCS_SR, target_sr=sr)
    out = out[:n] if len(out) >= n else np.pad(out, (0, n - len(out)))
    return out.astype(np.float32)
