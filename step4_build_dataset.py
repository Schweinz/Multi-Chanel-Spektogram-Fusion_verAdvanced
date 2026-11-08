"""
step4_build_dataset.py
=======================
STEP KEEMPAT — fusion 4 channel + split dataset.

Sama seperti versi sebelumnya:
  4 PNG grayscale per sampel → tensor float32 128×128×4
  Stratified split 70/15/15 (train/val/test), seed 42
  Disimpan sebagai dataset.npz

Perubahan: BASE_DIR sekarang SER_Advanced (dari config.py).
"""

import os
import sys
import glob
import numpy as np
import cv2
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append('/content/drive/MyDrive/SER_Advanced')
from config import (
    ENHANCED_DIR, SPEC_DIR, FUSED_DIR,
    EMOTION_MAP, EMOTION_LABELS, N_CLASSES,
    IMG_SIZE, CHANNEL_ORDER,
    RANDOM_SEED, TEST_SIZE, VAL_SIZE
)


def parse_ravdess_label(filename):
    """
    Ekstrak label emosi dari nama file RAVDESS.
    Format: 03-01-[EMOSI]-01-01-01-01_mel.png
    Index 2 setelah split '-' adalah kode emosi.
    Return: integer [0,7] atau None kalau tidak dikenali.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    ravdess_part = base.rsplit('_', 1)[0]
    parts = ravdess_part.split('-')
    if len(parts) < 3:
        return None
    emotion_name = EMOTION_MAP.get(parts[2])
    if emotion_name is None:
        return None
    return EMOTION_LABELS.index(emotion_name)


def fuse_channels(file_dict):
    """
    Gabungkan 4 PNG grayscale → tensor float32 (128,128,4) range [0,1].
    Urutan channel mengikuti CHANNEL_ORDER: mel, mfcc, cqt, chroma.
    """
    channels = []
    for ch_type in CHANNEL_ORDER:
        path = file_dict.get(ch_type)
        if path is None or not os.path.exists(path):
            channels.append(np.zeros(IMG_SIZE, dtype=np.float32))
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            channels.append(np.zeros(IMG_SIZE, dtype=np.float32))
            continue
        if img.shape[:2] != IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        channels.append(img.astype(np.float32) / 255.0)
    return np.stack(channels, axis=-1)   # (128, 128, 4)


def build_full_dataset(enhanced_dir=None):
    """
    Scan semua PNG, kelompokkan per sampel, fusion 4 channel.
    enhanced_dir: gunakan SPEC_DIR untuk kondisi tanpa DIP (B1/B3).
    """
    if enhanced_dir is None:
        enhanced_dir = ENHANCED_DIR

    all_png = glob.glob(f"{enhanced_dir}/*.png")
    if not all_png:
        raise FileNotFoundError(
            f"Tidak ada PNG di {enhanced_dir}. Pastikan step2/step3 sudah jalan."
        )

    sample_files = defaultdict(dict)
    for png_path in all_png:
        name_no_ext = os.path.splitext(os.path.basename(png_path))[0]
        parts = name_no_ext.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in CHANNEL_ORDER:
            sample_files[parts[0]][parts[1]] = png_path

    X_list, y_list, skipped = [], [], 0
    for basename, file_dict in tqdm(sample_files.items(),
                                     desc="Fusing channels", unit="sampel"):
        if not all(ch in file_dict for ch in CHANNEL_ORDER):
            skipped += 1
            continue
        label = parse_ravdess_label(f"{basename}_mel.png")
        if label is None:
            skipped += 1
            continue
        X_list.append(fuse_channels(file_dict))
        y_list.append(label)

    if skipped:
        print(f"Catatan: {skipped} sampel dilewati")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Dataset: {len(X)} sampel, shape={X.shape}")
    return X, y


def split_dataset(X, y):
    """
    Stratified split 70/15/15. Seed 42 untuk reproducibility.
    Semua kondisi ablasi dievaluasi pada test set yang sama.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio,
        random_state=RANDOM_SEED, stratify=y_temp
    )
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    print("STEP 4: Bangun Dataset (Fusion 4-Channel + Split)")
    print("=" * 50)
    os.makedirs(FUSED_DIR, exist_ok=True)

    out_path = os.path.join(FUSED_DIR, "dataset.npz")
    if os.path.exists(out_path):
        print(f"File sudah ada: {out_path}")
        data = np.load(out_path)
        print(f"  X_train: {data['X_train'].shape}")
        print(f"  X_val  : {data['X_val'].shape}")
        print(f"  X_test : {data['X_test'].shape}")
        sys.exit(0)

    X, y = build_full_dataset(enhanced_dir=ENHANCED_DIR)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    np.savez_compressed(out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Tersimpan ke {out_path} ({size_mb:.1f} MB)")
    print("Step 4 selesai. Lanjut ke step5_train.py")
