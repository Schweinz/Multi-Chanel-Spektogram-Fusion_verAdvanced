"""
step1_prepare_data.py
=====================
STEP PERTAMA — persiapan dataset RAVDESS.

Sama persis dengan versi sebelumnya karena dataset tidak berubah.
RAVDESS speech-only: 24 aktor, 8 emosi, 1440 file .wav.

Jalankan ini sekali di awal. Kalau data sudah ada di Drive,
fungsi download_ravdess() akan skip otomatis.
"""

import os
import sys
import glob
import shutil
from collections import Counter

sys.path.append('/content/drive/MyDrive/SER_Advanced')
from config import RAW_DATA_DIR, EMOTION_MAP, create_dirs


def mount_drive():
    """
    Mount Google Drive supaya data tersimpan permanen lintas sesi Colab.
    Aman dipanggil berkali-kali.
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive sudah ter-mount.")
    except Exception as e:
        print(f"Tidak bisa mount Drive (normal kalau di luar Colab): {e}")


def install_libraries():
    """
    Install semua library yang dibutuhkan pipeline.

    Tambahan dibanding versi sebelumnya: tidak ada.
    EfficientNetB0 sudah ada di tensorflow.keras.applications.
    """
    os.system(
        "pip install -q kaggle librosa opencv-python-headless "
        "tensorflow matplotlib seaborn scikit-learn tf-keras-vis tqdm"
    )
    print("Library sudah terinstall.")


def setup_kaggle():
    """
    Salin kaggle.json ke ~/.kaggle/ supaya Kaggle CLI bisa dipakai.

    Cara mendapatkan kaggle.json:
    1. Buka kaggle.com/account → Create New Token
    2. Upload ke Colab via panel kiri → Files → Upload
    """
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    src = "/content/kaggle.json"
    dst = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.exists(src):
        shutil.copy(src, dst)
        os.chmod(dst, 0o600)
        print("Kaggle API key sudah dikonfigurasi.")
    else:
        print("File kaggle.json tidak ditemukan. Upload dulu ke /content/.")


def download_ravdess():
    """Download dan flatten RAVDESS ke RAW_DATA_DIR."""
    existing = glob.glob(f"{RAW_DATA_DIR}/**/*.wav", recursive=True)
    if existing:
        print(f"Data sudah ada ({len(existing)} file), skip download.")
        return

    print("Mendownload RAVDESS dari Kaggle...")
    ret = os.system(
        "kaggle datasets download "
        "-d uwrfkaggler/ravdess-emotional-speech-audio "
        "-p /content/ --unzip"
    )
    if ret != 0:
        print("Download gagal. Cek koneksi atau kaggle.json.")
        return

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    wav_files = glob.glob(
        "/content/audio_speech_actors_01-24/**/*.wav", recursive=True
    )
    for f in wav_files:
        shutil.copy(f, os.path.join(RAW_DATA_DIR, os.path.basename(f)))
    print(f"{len(wav_files)} file .wav dipindahkan ke {RAW_DATA_DIR}")


def verify_dataset():
    """Verifikasi 1440 file ada dan label terbaca benar."""
    wav_files = glob.glob(f"{RAW_DATA_DIR}/**/*.wav", recursive=True)
    if not wav_files:
        wav_files = glob.glob(f"{RAW_DATA_DIR}/*.wav")

    print(f"\nVerifikasi dataset RAVDESS")
    print(f"{'─'*40}")
    print(f"Total file .wav: {len(wav_files)}")

    if not wav_files:
        print("Tidak ada file .wav ditemukan.")
        return False

    counts = Counter()
    for f in wav_files:
        parts = os.path.basename(f).split('-')
        if len(parts) >= 3:
            name = EMOTION_MAP.get(parts[2], f"?({parts[2]})")
            counts[name] += 1

    print("\nDistribusi per emosi:")
    for emo in EMOTION_MAP.values():
        n = counts.get(emo, 0)
        print(f"  {emo:<12}: {n:>4}  {'█' * (n // 10)}")

    ok = len(wav_files) >= 1440
    print(f"\n{'Dataset lengkap' if ok else 'Dataset TIDAK lengkap'} ({len(wav_files)} file).")
    return len(wav_files) > 0


if __name__ == "__main__":
    print("STEP 1: Persiapan Dataset RAVDESS")
    print("=" * 45)
    mount_drive()
    create_dirs()
    install_libraries()
    setup_kaggle()
    download_ravdess()
    ok = verify_dataset()
    if ok:
        print("\nStep 1 selesai. Lanjut ke step2_extract_spectrograms.py")
    else:
        print("\nStep 1 gagal. Periksa pesan error di atas.")
