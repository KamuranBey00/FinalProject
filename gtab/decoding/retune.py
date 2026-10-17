"""
Katman 6 Adım 3 — Akort / kapo: tel olasılıklarını kullanıcı akortuna taşıma (saf fonksiyon).

Akustik model standart akortta eğitildi: tab kafasında "tel s, sınıf f+1" = standart akortta std[s]+f perdesi. Akustik
çözümleme (segment / pitch_matrix / energy_rise) bu yüzden HER ZAMAN STANDARD_6 ile çalışır; kullanıcı akortu yalnız konum
aşamasına girer: kullanıcı akortunda (s, f') aynı perdeyi çalar = standart akortta (s, f' + user[s] − std[s]).
Kapo k: perdeler kapodan sayılır (gtab.core.instrument.with_capo), yani nutten f' + k.
"""

import numpy as np


def remap_probs(probs, std, user, capo=0):
    """
    probs (T, S, C) standart akort tel olasılıkları (sınıf 0 = sessiz, f+1 = perde f) -> aynı şekilde, kullanıcı akortunun
    (kapodan sayılan) perdelerinde: out[:, s, f'+1] = probs[:, s, f' + capo + user[s] − std[s] + 1]; model aralığı ya da
    kullanıcı perde sayısı dışı -> 0. Sessiz sınıfı aynen kalır. Standart akort + kapo 0 -> girdinin kendisi (birebir).
    """
    off = np.array(user.tuning) + capo - np.array(std.tuning)
    if capo == 0 and not off.any():
        return probs
    T, S, C = probs.shape
    out = np.zeros_like(probs)
    out[:, :, 0] = probs[:, :, 0]
    n_fret = min(C - 1, user.num_frets - capo + 1)                  # kullanıcı perdeleri 0 .. num_frets − capo
    for s in range(S):
        f = np.arange(n_fret)
        src = f + off[s] + 1
        ok = (src >= 1) & (src < C)
        out[:, s, f[ok] + 1] = probs[:, s, src[ok]]
    return out
