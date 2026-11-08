"""
config_local.py
===============
Konfigurasi untuk menjalankan pipeline di environment lokal (bukan Google Colab).
Path disesuaikan dengan direktori kerja saat ini.

Semua parameter model dan DIP SAMA dengan config.py yang akan digunakan di Colab/Drive.
Hanya path direktori yang berbeda.
"""

import os

# ── PATH LOKAL ────────────────────────────────────────────────────────────────
BASE_DIR = "/home/claude/ser_advanced_local"

RAW_DATA_DIR  = "/home/claude/ravdess_data"       # sudah ada dari extract
SPEC_DIR      = os.path.join(BASE_DIR, "spectrograms")
ENHANCED_DIR  = os.path.join(BASE_DIR, "enhanced_spectrograms")
FUSED_DIR     = os.path.join(BASE_DIR, "fused_dataset")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
RESULT_DIR    = os.path.join(BASE_DIR, "results")

def create_dirs():
    for d in [SPEC_DIR, ENHANCED_DIR, FUSED_DIR, MODEL_DIR, RESULT_DIR]:
        os.makedirs(d, exist_ok=True)

# ── LABEL EMOSI ───────────────────────────────────────────────────────────────
EMOTION_MAP = {
    '01': 'neutral', '02': 'calm',    '03': 'happy',   '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust',  '08': 'surprised'
}
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
N_CLASSES      = 8

# ── PARAMETER AUDIO DAN SPEKTROGRAM ───────────────────────────────────────────
SAMPLE_RATE      = 22050
IMG_SIZE         = (128, 128)
N_MELS           = 128
N_FFT            = 2048
HOP_LENGTH       = 512
N_MFCC           = 128
N_BINS           = 128
BINS_PER_OCTAVE  = 24
CHANNEL_ORDER    = ['mel', 'mfcc', 'cqt', 'chroma']

# ── PARAMETER DIP ─────────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)
UM_SIGMA         = 1.0
UM_STRENGTH      = 1.5

# ── PARAMETER TRAINING ────────────────────────────────────────────────────────
BATCH_SIZE       = 32
EPOCHS_FROZEN    = 10
EPOCHS_UNFROZEN  = 100
LEARNING_RATE    = 1e-4
DROPOUT_RATE     = 0.3
RANDOM_SEED      = 42
TEST_SIZE        = 0.15
VAL_SIZE         = 0.15

# ── SE BLOCK ──────────────────────────────────────────────────────────────────
SE_RATIO = 16

# ── SPECAUGMENT ───────────────────────────────────────────────────────────────
SPEC_AUG_FREQ_MASK_MAX   = 20
SPEC_AUG_TIME_MASK_MAX   = 25
SPEC_AUG_NUM_FREQ_MASKS  = 2
SPEC_AUG_NUM_TIME_MASKS  = 2
