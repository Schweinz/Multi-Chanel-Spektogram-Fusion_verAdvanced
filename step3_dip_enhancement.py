"""
step3_dip_enhancement.py
=========================
STEP KETIGA — DIP enhancement: CLAHE → Unsharp Masking.

Sama persis dengan versi sebelumnya. Pipeline DIP ini sudah terbukti
efektif di paper pertama dan dipertahankan di versi Advanced.

CLAHE  : perbaiki kontras lokal per tile 16×16
         clipLimit=2.0, tileGridSize=(8×8)
Unsharp: tajamkan tepi setelah CLAHE
         sigma=1.0, strength=1.5

Urutan CLAHE → UM disengaja:
CLAHE angkat detail tersembunyi dulu, baru UM tajamkan detail itu.
Kalau dibalik, UM akan menajamkan noise sebelum CLAHE meredamnya.

Input : SPEC_DIR  (gambar sebelum DIP)
Output: ENHANCED_DIR (gambar setelah DIP)
"""

import os
import sys
import glob
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('/content/drive/MyDrive/SER_Advanced')
from config import (
    SPEC_DIR, ENHANCED_DIR,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
    UM_SIGMA, UM_STRENGTH
)


def apply_clahe(img):
    """
    CLAHE: perbaiki kontras per tile secara independen.
    clipLimit=2.0 mencegah noise ikut diperkuat.
    tileGridSize=(8,8) → 64 tile ukuran 16×16 pada gambar 128×128.
    """
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )
    return clahe.apply(img)


def apply_unsharp_masking(img, sigma=None, strength=None):
    """
    Unsharp Masking: sharpened = (1+α)*img - α*blur
    dengan α = strength = 1.5, sigma = 1.0.

    cv2.addWeighted(img, 1+strength, blurred, -strength, 0)
    = 2.5 * img - 1.5 * blur
    = img + 1.5 * (img - blur)   ← tepi ditambahkan kembali
    """
    if sigma is None:
        sigma = UM_SIGMA
    if strength is None:
        strength = UM_STRENGTH
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)


def apply_dip_pipeline(img):
    """Pipeline penuh: CLAHE → Unsharp Masking."""
    return apply_unsharp_masking(apply_clahe(img))


def enhance_one_file(png_path, output_dir):
    """Baca PNG grayscale, enhance, simpan. Skip kalau sudah ada."""
    try:
        out_path = os.path.join(output_dir, os.path.basename(png_path))
        if os.path.exists(out_path):
            return True
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        cv2.imwrite(out_path, apply_dip_pipeline(img))
        return True
    except Exception as e:
        print(f"\nError: {os.path.basename(png_path)}: {e}")
        return False


def visualize_dip_comparison(basename_prefix):
    """
    Tampilkan perbandingan 3 tahap untuk 4 channel:
    Asli | Setelah CLAHE | Setelah CLAHE+UM
    """
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle(f"DIP Comparison: {basename_prefix}", fontsize=13)
    col_labels = ['Asli', 'Setelah CLAHE', 'Setelah CLAHE+UM']

    for row, ch in enumerate(['mel', 'mfcc', 'cqt', 'chroma']):
        orig = cv2.imread(
            os.path.join(SPEC_DIR, f"{basename_prefix}_{ch}.png"),
            cv2.IMREAD_GRAYSCALE
        )
        if orig is None:
            continue
        for col, img in enumerate([
            orig, apply_clahe(orig),
            apply_unsharp_masking(apply_clahe(orig))
        ]):
            axes[row, col].imshow(img, cmap='magma', aspect='auto')
            axes[row, col].set_title(col_labels[col] if row == 0 else '')
            axes[row, col].set_ylabel(ch.upper() if col == 0 else '')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("STEP 3: DIP Enhancement (CLAHE + Unsharp Masking)")
    print("=" * 50)
    print(f"  CLAHE : clipLimit={CLAHE_CLIP_LIMIT}, tileGrid={CLAHE_TILE_GRID}")
    print(f"  Unsharp: sigma={UM_SIGMA}, strength={UM_STRENGTH}")

    os.makedirs(ENHANCED_DIR, exist_ok=True)
    png_files = sorted(glob.glob(f"{SPEC_DIR}/*.png"))
    if not png_files:
        print("Tidak ada PNG di SPEC_DIR. Jalankan step2 dulu.")
        sys.exit(1)

    print(f"\nDitemukan {len(png_files)} gambar")
    sample_prefix = '_'.join(os.path.basename(png_files[0]).split('_')[:-1])
    visualize_dip_comparison(sample_prefix)

    success = sum(enhance_one_file(f, ENHANCED_DIR)
                  for f in tqdm(png_files, desc="DIP Enhancement", unit="file"))
    print(f"\nSelesai: {success} berhasil, {len(png_files)-success} gagal")
    print(f"File di ENHANCED_DIR: {len(glob.glob(f'{ENHANCED_DIR}/*.png'))}")
    print("Step 3 selesai. Lanjut ke step4_build_dataset.py")
