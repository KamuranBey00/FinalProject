"""
gtab — gitar tab transkripsiyonu çekirdek kütüphanesi.

Yalnızca import edilir; çalıştırılan her şey scripts/ altındadır.

    gtab.config          ses/CQT sabitleri (proje ömrü boyunca sabit)
    gtab.paths           proje yolları (data/raw, data/cache, checkpoints)
    gtab.utils           seed, cihaz
    gtab.core            instrument (akort, tel/perde), note_event (ortak nota şeması)
    gtab.data            features (CQT), labels/tab_labels (etiket), torch_dataset,
                         guitarset / gaps / synthtab (veri seti okuyucuları)
    gtab.models          nets (PitchCNN/TabCNN/TabCRNN), losses, inference
    gtab.decoding        decode (çözümleme + ASCII tab), viterbi, transitions
    gtab.evaluation      metrics
"""
