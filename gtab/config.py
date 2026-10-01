"""
Katman 0 — Ön-işleme sabitleri.

Bu değerler PROJENİN ÖMRÜ BOYUNCA SABİT kalmalı.
Sebep: modelini bir kez bu ayarlarla eğitirsen, sonradan sample rate ya da
CQT parametresini değiştirmek "her şeyi baştan eğit" demektir. O yüzden
en baştan, tüm katmanların paylaşacağı tek bir kaynak burada tanımlı.
Değiştirmek istersen: yeni bir sürüm numarası ver, eski modelleri koru.
"""

import os

# --- GuitarSet veri yolu ---
# mirdata.initialize("guitarset") data_home vermezsen os.getenv("HOME", "/tmp")
# kullanır. Windows'ta "HOME" kabuğa göre değişir: PowerShell/cmd'de genelde
# TANIMSIZ (-> "/tmp" -> o an çalışılan sürücüde \tmp\mir_datasets), Git Bash'te
# TANIMLI (-> \Users\<kullanıcı>\mir_datasets). Aynı makinede hangi terminalden
# çalıştırdığına göre mirdata SESSİZCE FARKLI (ve boş) bir dizine bakabilir.
# Veriyi bir kez indirdiğimiz gerçek konumu burada SABİTLEYİP tüm mirdata.initialize
# çağrılarının bunu kullanmasını sağlıyoruz -> hangi terminalden çalıştırılırsa
# çalıştırılsın aynı sonuç.
GUITARSET_DATA_HOME = os.environ.get("GUITARSET_DATA_HOME", r"C:\tmp\mir_datasets\guitarset")

# --- Ses ---
SAMPLE_RATE = 22050          # Hz. Gitar için fazlasıyla yeterli (Nyquist ~11 kHz).
HOP_LENGTH = 512             # ~23 ms'lik zaman çözünürlüğü (512 / 22050).
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # saniyedeki analiz karesi sayısı (~43)

# --- CQT (Constant-Q Transform): gitar transkripsiyonunda standart girdi ---
# Neden CQT? Frekans eksenini nota (yarım ses) mantığına göre böler; bir
# spektrogramın aksine oktavlar eşit aralıklı çıkar -> model için çok daha kolay.
CQT_FMIN_HZ = 32.703         # C1. Gitarın en pesinden (E2, ~82 Hz) daha altta başlıyoruz ki güvende olalım.
CQT_BINS_PER_OCTAVE = 24     # oktav başına 24 bin = yarım ses başına 2 bin (bend gibi mikroton geçişler için önemli).
CQT_N_BINS = 192             # 192 / 24 = 8 oktav kapsama.

# --- Etiketleme çözünürlüğü ---
# Teknik tespiti için sürekli F0 (pitch) eğrisini bu kare hızında saklayacağız.
# Katman 4'te (bend/slide) hayati; şimdiden sabitliyoruz.
PITCH_CONTOUR_FRAME_RATE = FRAME_RATE
