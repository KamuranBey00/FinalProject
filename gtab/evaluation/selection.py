"""
Katman 4 Faz 0 — Model seçimi / early stopping / aşırı öğrenme takibi (eğitim YOK, yalnız kural).

Katman 4'ün eğitim script'leri epoch döngüsünde bu sınıfı kullanır; böylece şu kurallar
koda bağlanmış olur:
  1. Seçim YALNIZ kalibrasyon split'inde yapılır (protokol kapısı: `assert_may_select`).
     Bağımsız test (gaps_test) seçimde kullanılamaz -> ProtocolError.
  2. Seçim ölçütü teknik tanımada **macro-F1**'dir (NORMAL baskınlığı; bkz. technique_eval).
  3. Checkpoint yalnız seçim ölçütü iyileşince yazılır; hiçbir epoch başlangıç modelini
     (epoch 0) geçmezse başlangıç modeli korunur (Katman 3.13 Adım 1'deki davranış).
  4. Eğitim kaybı düşerken doğrulama ölçütü iyileşmiyorsa aşırı öğrenme işaretlenir
     (README14 Adım 2b'de parmak kaybının 10. epoch'tan sonra yaptığı şey).
  5. `patience` epoch boyunca iyileşme yoksa eğitim durur.

Sınıf dengesizliği için YENİ kod yazılmaz: mevcut `GuitarSetSeq/PitchSeq.class_weights`
(şema `inv` / `sqrt`, `cap`) ve `train_onset --oversample` yeniden kullanılır.
"""

from dataclasses import dataclass
from typing import List, Optional

from gtab.evaluation.protocol import assert_may_select

DEFAULT_PATIENCE = 5
OVERFIT_WINDOW = 3


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    val_score: float                      # seçim ölçütü (teknik: macro-F1)


class EpochSelector:
    """Epoch günlüğü + en iyi epoch seçimi + early stopping + aşırı öğrenme işareti."""

    def __init__(self, selection_split: str, patience: int = DEFAULT_PATIENCE,
                 min_delta: float = 0.0, metric_name: str = "macro-F1"):
        assert_may_select(selection_split)               # protokol kapisi
        self.selection_split = selection_split
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.history: List[EpochRecord] = []
        self._best: Optional[EpochRecord] = None

    def update(self, epoch: int, train_loss: float, val_score: float,
               val_loss: Optional[float] = None) -> bool:
        """-> bu epoch yeni en iyi mi (checkpoint yazılmalı mı)."""
        rec = EpochRecord(epoch, float(train_loss), val_loss, float(val_score))
        self.history.append(rec)
        if self._best is None or rec.val_score > self._best.val_score + self.min_delta:
            self._best = rec
            return True
        return False

    @property
    def best(self) -> Optional[EpochRecord]:
        return self._best

    @property
    def epochs_since_best(self) -> int:
        if self._best is None:
            return 0
        return self.history[-1].epoch - self._best.epoch

    def should_stop(self) -> bool:
        return self._best is not None and self.epochs_since_best >= self.patience

    def kept_initial(self) -> bool:
        """En iyi epoch 0 ise: hiçbir epoch başlangıç modelini geçmedi, taban korunur."""
        return self._best is not None and self._best.epoch == 0

    def overfitting(self, window: int = OVERFIT_WINDOW) -> bool:
        """
        Son `window` epoch'ta eğitim kaybı sürekli düştü ama doğrulama ölçütü iyileşmedi mi?
        (Kayıp düşüşü = öğrenme, ölçüt durgun = ezberleme.)
        """
        if len(self.history) < window + 1:
            return False
        tail = self.history[-(window + 1):]
        losses = [r.train_loss for r in tail]
        falling = all(b < a for a, b in zip(losses, losses[1:]))
        return falling and max(r.val_score for r in tail[1:]) <= tail[0].val_score + self.min_delta

    def format_log(self) -> str:
        head = (f"secim split'i: {self.selection_split} | olcut: {self.metric_name} | "
                f"patience: {self.patience}")
        rows = [f"  ep {r.epoch:3d} | egitim kaybi {r.train_loss:.4f}"
                + (f" | dogrulama kaybi {r.val_loss:.4f}" if r.val_loss is not None else "")
                + f" | {self.metric_name} {r.val_score:.4f}"
                + ("  <- en iyi" if self._best is not None and r.epoch == self._best.epoch else "")
                for r in self.history]
        tail = []
        if self._best is not None:
            tail.append(f"en iyi: ep {self._best.epoch} ({self.metric_name} {self._best.val_score:.4f})"
                        + ("  [baslangic modeli korundu]" if self.kept_initial() else ""))
        if self.overfitting():
            tail.append("UYARI: egitim kaybi dusuyor ama dogrulama olcutu iyilesmiyor (asiri ogrenme)")
        if self.should_stop():
            tail.append(f"early stopping: {self.epochs_since_best} epoch iyilesme yok")
        return "\n".join([head] + rows + tail)
