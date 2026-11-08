"""
config.py
=========
Pusat konfigurasi untuk pipeline versi Advanced.

Perubahan dari versi sebelumnya (MobileNetV2):
  - Backbone diganti EfficientNetB0
  - Ditambahkan SE Block (Squeeze-and-Excitation attention)
  - Ditambahkan SpecAugment (augmentasi frekuensi dan waktu)

Kalau mau eksperimen, ubah nilai di sini saja — tidak perlu
buka file step1 sampai step6 satu per satu.
"""

import os


# ---------------------------------------------------------------------------
# PATH — sesuaikan BASE_DIR dengan lokasi Google Drive kamu
# ---------------------------------------------------------------------------

BASE_DIR     = "/content/drive/MyDrive/SER_Advanced"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")
SPEC_DIR     = os.path.join(BASE_DIR, "data/spectrograms")
ENHANCED_DIR = os.path.join(BASE_DIR, "data/enhanced")
FUSED_DIR    = os.path.join(BASE_DIR, "data/fused")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RESULT_DIR   = os.path.join(BASE_DIR, "results")
GRADCAM_DIR  = os.path.join(BASE_DIR, "gradcam")


# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050   # Hz, standard librosa


# ---------------------------------------------------------------------------
# SPEKTROGRAM
# ---------------------------------------------------------------------------

IMG_SIZE        = (128, 128)
N_MELS          = 128
N_FFT           = 2048
HOP_LENGTH      = 512
N_MFCC          = 40
N_BINS          = 84
BINS_PER_OCTAVE = 12
N_CHROMA        = 12


# ---------------------------------------------------------------------------
# DIP — sama seperti versi sebelumnya, sudah terbukti efektif
# ---------------------------------------------------------------------------

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)
UM_SIGMA         = 1.0
UM_STRENGTH      = 1.5


# ---------------------------------------------------------------------------
# MULTI-CHANNEL
# ---------------------------------------------------------------------------

N_CHANNELS    = 4
CHANNEL_ORDER = ['mel', 'mfcc', 'cqt', 'chroma']


# ---------------------------------------------------------------------------
# SPLIT
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15


# ---------------------------------------------------------------------------
# HYPERPARAMETER TRAINING
# ---------------------------------------------------------------------------

BATCH_SIZE      = 32
EPOCHS_FROZEN   = 10    # Phase 1: hanya head yang dilatih
EPOCHS_UNFROZEN = 100   # Phase 2: fine-tuning penuh + early stopping

# EfficientNetB0 sedikit lebih sensitif dari MobileNetV2 —
# LR phase 1 tetap 1e-4, phase 2 tetap LR/10 = 1e-5
LEARNING_RATE = 1e-4
DROPOUT_RATE  = 0.3


# ---------------------------------------------------------------------------
# SE BLOCK (Squeeze-and-Excitation)
# ---------------------------------------------------------------------------

# Reduction ratio untuk SE Block: jumlah unit di Dense squeeze = channels / SE_RATIO
# Nilai 16 adalah standar dari paper Hu et al. (CVPR 2018)
# Terlalu kecil (mis. 4) → tidak cukup kompresi
# Terlalu besar (mis. 32) → bottleneck terlalu sempit, kehilangan informasi
SE_RATIO = 16


# ---------------------------------------------------------------------------
# SPECAUGMENT
# ---------------------------------------------------------------------------

# SpecAugment — Park et al., Interspeech 2019
# Mask frekuensi: pilih F bin secara random dan set ke 0
# Mask waktu    : pilih T frame secara random dan set ke 0
# Nilai ini dipakai di step5 saat augmentasi training data

# Max lebar mask frekuensi (dalam piksel dari 128 baris frekuensi)
# 20/128 ≈ 15% dari dimensi frekuensi — cukup challenging tapi tidak terlalu ekstrem
SPEC_AUG_FREQ_MASK_MAX = 20

# Max lebar mask waktu (dalam piksel dari 128 kolom waktu)
# 25/128 ≈ 20% dari dimensi waktu
SPEC_AUG_TIME_MASK_MAX = 25

# Jumlah mask yang diterapkan per dimensi
SPEC_AUG_NUM_FREQ_MASKS = 2
SPEC_AUG_NUM_TIME_MASKS = 2


# ---------------------------------------------------------------------------
# LABEL EMOSI RAVDESS
# ---------------------------------------------------------------------------

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
EMOTION_LABELS = list(EMOTION_MAP.values())
N_CLASSES      = len(EMOTION_LABELS)   # = 8


# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------

def create_dirs():
    """Buat semua folder pipeline jika belum ada."""
    for d in [RAW_DATA_DIR, SPEC_DIR, ENHANCED_DIR,
              FUSED_DIR, MODEL_DIR, RESULT_DIR, GRADCAM_DIR]:
        os.makedirs(d, exist_ok=True)
    print("Semua folder pipeline sudah siap.")


if __name__ == "__main__":
    create_dirs()
