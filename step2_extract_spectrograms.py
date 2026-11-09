"""
step2_extract_spectrograms.py
==============================
STEP KEDUA — ekstraksi 4 spektrogram dari setiap file .wav.

Sama persis dengan versi sebelumnya:
  Mel Spectrogram → energi per frekuensi-mel vs waktu
  MFCC            → koefisien cepstral, identitas vokal
  CQT             → log-frekuensi, kontur nada
  Chroma          → distribusi energi per kelas nada (12 nada)

Setiap file .wav menghasilkan 4 PNG grayscale 128×128.
Output: SPEC_DIR
"""

import os
import sys
import glob
import numpy as np
import librosa
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = '/content/drive/MyDrive/SER_Advanced'
sys.path.insert(0, BASE_DIR)
import config as _cfg

# Auto-patch paths agar selalu mengarah ke Drive
_cfg.BASE_DIR     = BASE_DIR
_cfg.DATA_DIR     = os.path.join(BASE_DIR, 'ravdess_data')
_cfg.RAW_DATA_DIR = os.path.join(BASE_DIR, 'ravdess_data')
_cfg.SPEC_DIR     = os.path.join(BASE_DIR, 'spectrograms')
_cfg.ENHANCED_DIR = os.path.join(BASE_DIR, 'enhanced_spectrograms')
_cfg.FUSED_DIR    = os.path.join(BASE_DIR, 'fused_dataset')
_cfg.MODEL_DIR    = os.path.join(BASE_DIR, 'models')
_cfg.RESULT_DIR   = os.path.join(BASE_DIR, 'results')
_cfg.create_dirs()

RAW_DATA_DIR   = _cfg.RAW_DATA_DIR
SPEC_DIR       = _cfg.SPEC_DIR
SAMPLE_RATE    = _cfg.SAMPLE_RATE
IMG_SIZE       = _cfg.IMG_SIZE
N_MELS         = _cfg.N_MELS
N_FFT          = _cfg.N_FFT
HOP_LENGTH     = _cfg.HOP_LENGTH
N_MFCC         = _cfg.N_MFCC
N_BINS         = _cfg.N_BINS
BINS_PER_OCTAVE = _cfg.BINS_PER_OCTAVE


def load_audio(filepath, target_duration=3.0):
    """
    Load audio dan normalisasi ke 3 detik.
    Pad dengan reflect (bukan zero) supaya tidak ada artefak dinding diam.
    Kalau terlalu panjang, ambil dari tengah.
    """
    target_samples = int(SAMPLE_RATE * target_duration)
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)

    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='reflect')
    elif len(y) > target_samples:
        start = (len(y) - target_samples) // 2
        y = y[start: start + target_samples]

    return y, sr


def extract_mel(y, sr):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(S, ref=np.max)


def extract_mfcc(y, sr):
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )


def extract_cqt(y, sr):
    C = np.abs(librosa.cqt(
        y=y, sr=sr, n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP_LENGTH
    ))
    return librosa.amplitude_to_db(C, ref=np.max)


def extract_chroma(y, sr):
    return librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )


def spectrogram_to_image(spec_data):
    """
    Normalisasi ke [0,255], resize ke 128×128, simpan sebagai uint8 grayscale.
    Disimpan grayscale karena akan di-stack jadi 4-channel tensor di step4.
    """
    spec_min, spec_max = spec_data.min(), spec_data.max()
    if spec_max - spec_min > 1e-8:
        normalized = (spec_data - spec_min) / (spec_max - spec_min) * 255
    else:
        normalized = np.zeros_like(spec_data)
    img = cv2.resize(normalized.astype(np.uint8), IMG_SIZE,
                     interpolation=cv2.INTER_AREA)
    return img


def process_one_file(wav_path, output_dir):
    """Proses satu .wav → 4 PNG. Skip kalau sudah ada."""
    try:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        expected = [f"{basename}_{t}.png" for t in ['mel','mfcc','cqt','chroma']]
        if all(os.path.exists(os.path.join(output_dir, f)) for f in expected):
            return True

        y, sr = load_audio(wav_path)
        specs = {
            'mel':    extract_mel(y, sr),
            'mfcc':   extract_mfcc(y, sr),
            'cqt':    extract_cqt(y, sr),
            'chroma': extract_chroma(y, sr),
        }
        for spec_type, spec_data in specs.items():
            img = spectrogram_to_image(spec_data)
            cv2.imwrite(os.path.join(output_dir, f"{basename}_{spec_type}.png"), img)
        return True
    except Exception as e:
        print(f"\nError: {os.path.basename(wav_path)}: {e}")
        return False


def visualize_sample(wav_path):
    """Tampilkan 4 spektrogram dari satu file sebagai sanity check."""
    y, sr = load_audio(wav_path)
    specs = {
        'Mel': extract_mel(y, sr),
        'MFCC': extract_mfcc(y, sr),
        'CQT': extract_cqt(y, sr),
        'Chroma': extract_chroma(y, sr),
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Spektrogram: {os.path.basename(wav_path)}", fontsize=12)
    for ax, (title, S) in zip(axes, specs.items()):
        im = ax.imshow(S, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("STEP 2: Ekstraksi Spektrogram")
    print("=" * 45)
    os.makedirs(SPEC_DIR, exist_ok=True)

    wav_files = sorted(glob.glob(f"{RAW_DATA_DIR}/**/*.wav", recursive=True))
    if not wav_files:
        wav_files = sorted(glob.glob(f"{RAW_DATA_DIR}/*.wav"))
    if not wav_files:
        print("Tidak ada file .wav. Jalankan step1 dulu.")
        sys.exit(1)

    print(f"Ditemukan {len(wav_files)} file .wav")
    visualize_sample(wav_files[0])

    success = sum(process_one_file(f, SPEC_DIR)
                  for f in tqdm(wav_files, desc="Ekstraksi", unit="file"))
    fail = len(wav_files) - success
    print(f"\nSelesai: {success} berhasil, {fail} gagal")
    print(f"PNG dihasilkan: {len(glob.glob(f'{SPEC_DIR}/*.png'))}")
    print("Step 2 selesai. Lanjut ke step3_dip_enhancement.py")
