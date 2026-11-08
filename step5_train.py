"""
step5_train.py
==============
STEP KELIMA — training EfficientNetB0 + SE Block + SpecAugment.

PERUBAHAN UTAMA dari versi sebelumnya (MobileNetV2):

1. EfficientNetB0 menggantikan MobileNetV2
   - Compound scaling: lebar, kedalaman, resolusi di-scale bersamaan
   - Parameter: ~5.3M vs MobileNetV2 ~3.4M (lebih ekspresif)
   - Output feature map sebelum GAP: 1280-d (sama dengan MobileNetV2)

2. SE Block (Squeeze-and-Excitation) setelah backbone
   - Belajar bobot kepentingan tiap channel feature map secara otomatis
   - Terinspirasi dari hasil Grad-CAM paper 1 yang menunjukkan channel
     importance berbeda per emosi — SE Block mengeksplisitkan ini
   - Overhead parameter sangat kecil (< 0.1% dari total parameter)

3. SpecAugment pada data training
   - Mask frekuensi: pilih F baris random, set ke nol
   - Mask waktu    : pilih T kolom random, set ke nol
   - Hanya diaplikasikan saat training, bukan saat validasi/test

Arsitektur lengkap:
  Input (128×128×4)
       ↓
  Conv2D(3, 1×1)              ← channel projection (sama paper 1)
       ↓
  EfficientNetB0              ← BARU
       ↓
  SE Block                    ← BARU: squeeze-and-excitation attention
       ↓
  GlobalAveragePooling2D
       ↓
  Dropout(0.3)
       ↓
  Dense(128, ReLU) + BN
       ↓
  Dropout(0.3)
       ↓
  Dense(8, softmax)
"""

import os
import sys
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('/content/drive/MyDrive/SER_Advanced')
from config import (
    FUSED_DIR, SPEC_DIR, MODEL_DIR, RESULT_DIR,
    N_CLASSES, BATCH_SIZE,
    EPOCHS_FROZEN, EPOCHS_UNFROZEN,
    LEARNING_RATE, DROPOUT_RATE,
    SE_RATIO,
    SPEC_AUG_FREQ_MASK_MAX, SPEC_AUG_TIME_MASK_MAX,
    SPEC_AUG_NUM_FREQ_MASKS, SPEC_AUG_NUM_TIME_MASKS,
    CHANNEL_ORDER
)

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Input, Conv2D, GlobalAveragePooling2D,
    Dropout, Dense, BatchNormalization,
    Reshape, Multiply, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.utils import to_categorical

print(f"TensorFlow versi: {tf.__version__}")
print(f"GPU tersedia: {tf.config.list_physical_devices('GPU')}")


# ── SPECAUGMENT ───────────────────────────────────────────────────────────────

def apply_specaugment(X):
    """
    SpecAugment — Park et al., Interspeech 2019.

    Dua jenis masking yang diterapkan ke tensor 4-channel sekaligus:

    1. Frequency masking: pilih F baris berturut-turut secara random,
       set semua nilainya ke nol. Ini mensimulasikan kehilangan informasi
       di rentang frekuensi tertentu — model dipaksa belajar dari frekuensi lain.

    2. Time masking: pilih T kolom berturut-turut secara random, set ke nol.
       Mensimulasikan bagian ucapan yang hilang — model tidak boleh terlalu
       bergantung pada momen waktu tertentu.

    Diterapkan ke semua 4 channel sekaligus supaya konsisten:
    kalau frekuensi 50-70 dimasking di Mel, channel lain juga dimasking
    di rentang yang sama — ini lebih realistis secara fisik.

    X: array (N, 128, 128, 4) float32
    Returns: X yang sudah diaugmentasi (copy, bukan in-place)
    """
    X_aug = X.copy()
    N, H, W, C = X_aug.shape   # H=frekuensi, W=waktu

    for i in range(N):
        # Frequency masking
        for _ in range(SPEC_AUG_NUM_FREQ_MASKS):
            f = np.random.randint(0, SPEC_AUG_FREQ_MASK_MAX + 1)
            f0 = np.random.randint(0, max(1, H - f))
            X_aug[i, f0:f0 + f, :, :] = 0.0

        # Time masking
        for _ in range(SPEC_AUG_NUM_TIME_MASKS):
            t = np.random.randint(0, SPEC_AUG_TIME_MASK_MAX + 1)
            t0 = np.random.randint(0, max(1, W - t))
            X_aug[i, :, t0:t0 + t, :] = 0.0

    return X_aug


def visualize_specaugment(X_sample, idx=0):
    """
    Tampilkan satu sampel sebelum dan sesudah SpecAugment untuk 4 channel.
    Berguna untuk verifikasi augmentasi bekerja benar.
    """
    original = X_sample[idx:idx+1]
    augmented = apply_specaugment(original.copy())

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("SpecAugment — Sebelum vs Sesudah", fontsize=13)

    for col, ch in enumerate(CHANNEL_ORDER):
        axes[0, col].imshow(
            original[0, :, :, col], cmap='viridis', aspect='auto', origin='lower'
        )
        axes[0, col].set_title(f'{ch.upper()}\nAsli')
        axes[0, col].axis('off')

        axes[1, col].imshow(
            augmented[0, :, :, col], cmap='viridis', aspect='auto', origin='lower'
        )
        axes[1, col].set_title(f'{ch.upper()}\nSetelah SpecAugment')
        axes[1, col].axis('off')

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "specaugment_demo.png")
    os.makedirs(RESULT_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"SpecAugment demo disimpan ke {save_path}")


# ── SE BLOCK ──────────────────────────────────────────────────────────────────

def se_block(x, ratio=SE_RATIO):
    """
    Squeeze-and-Excitation Block — Hu et al., CVPR 2018.

    Cara kerja:
    1. SQUEEZE: GlobalAveragePooling → vektor (1, 1, C)
       "Merangkum" seluruh spatial info menjadi satu nilai per channel.

    2. EXCITATION: Dense(C/ratio, relu) → Dense(C, sigmoid)
       Belajar bobot kepentingan per channel dari vektor terkompres.
       Dense pertama = bottleneck untuk memaksa generalisasi.
       Sigmoid menghasilkan nilai [0,1] per channel — ini bobot pentingnya.

    3. SCALE: kalikan bobot ke feature map asli.
       Channel yang "penting" (bobot ≈ 1) diperkuat.
       Channel yang "tidak penting" (bobot ≈ 0) ditekan.

    Kenapa ini cocok untuk paper ini?
    Paper 1 menunjukkan via Grad-CAM bahwa tiap emosi punya pola
    channel importance yang berbeda. SE Block mengeksplisitkan
    mekanisme itu langsung di dalam arsitektur — bukan hanya dianalisis
    setelah training, tapi aktif digunakan saat forward pass.

    x: tensor (batch, H, W, C) dari output EfficientNetB0
    Returns: tensor ukuran sama, channel-nya sudah di-reweight
    """
    channels = x.shape[-1]   # jumlah filter output EfficientNetB0

    # Squeeze: (batch, H, W, C) → (batch, C)
    squeeze = GlobalAveragePooling2D()(x)

    # Excitation: (batch, C) → (batch, C/ratio) → (batch, C)
    excitation = Dense(max(1, channels // ratio), activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)

    # Reshape supaya bisa di-broadcast: (batch, C) → (batch, 1, 1, C)
    excitation = Reshape((1, 1, channels))(excitation)

    # Scale: element-wise multiply
    return Multiply()([x, excitation])


# ── ARSITEKTUR MODEL ──────────────────────────────────────────────────────────

def build_model_4ch(n_classes=N_CLASSES):
    """
    Model 4-channel dengan EfficientNetB0 + SE Block.

    Alur:
      Input (128,128,4)
      → Conv2D(3, 1×1)        channel projection
      → EfficientNetB0         feature extraction
      → SE Block               channel attention
      → GAP → Dropout → Dense(128) → BN → Dropout → Dense(8)
    """
    inp = Input(shape=(128, 128, 4), name='input_4ch')

    # Channel projection: 4 → 3
    x = Conv2D(3, (1, 1), padding='same', name='channel_projection')(inp)

    # EfficientNetB0 backbone
    base = EfficientNetB0(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False   # Frozen di phase 1

    x = base(x, training=False)

    # SE Block — diterapkan pada output spatial EfficientNetB0
    x = se_block(x, ratio=SE_RATIO)

    # Classification head
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inp, out, name='SER_EfficientNetB0_SE_4ch')
    return model, base


def build_model_3ch(n_classes=N_CLASSES):
    """
    Model 3-channel (single Mel) dengan EfficientNetB0 + SE Block.
    Untuk kondisi ablasi B1 dan B2.
    """
    base = EfficientNetB0(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    inp = Input(shape=(128, 128, 3), name='input_3ch')
    x = base(inp, training=False)
    x = se_block(x, ratio=SE_RATIO)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(inp, out, name='SER_EfficientNetB0_SE_3ch')
    return model, base


# ── LOAD DATASET ──────────────────────────────────────────────────────────────

def load_dataset_npz():
    data_path = os.path.join(FUSED_DIR, "dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"File {data_path} tidak ditemukan. Jalankan step4 dulu."
        )
    data = np.load(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    print(f"Dataset dimuat: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_b1_b3_data():
    """Data tanpa DIP untuk kondisi B1/B3."""
    from step4_build_dataset import build_full_dataset, split_dataset
    print("Membangun dataset B1/B3 dari SPEC_DIR (tanpa DIP)...")
    X_raw, y_raw = build_full_dataset(enhanced_dir=SPEC_DIR)
    return split_dataset(X_raw, y_raw)


def prepare_ablation_data(X, condition):
    """
    B1/B2: duplikasi channel Mel → 3 channel untuk model 3ch.
    B3/B4: pakai 4 channel langsung.
    """
    if condition in ('B1', 'B2'):
        mel = X[:, :, :, 0:1]
        return np.concatenate([mel, mel, mel], axis=-1)
    return X


# ── TRAINING 2 FASE ───────────────────────────────────────────────────────────

def train_model(model, base_model, X_tr, y_tr, X_vl, y_vl,
                model_name, model_dir, use_specaugment=False):
    """
    Training 2 fase:
    Phase 1 — FROZEN (lr=1e-4, 10 epoch): stabilkan head
    Phase 2 — UNFROZEN (lr=1e-5, max 100 epoch): fine-tuning

    use_specaugment: kalau True, augmentasi X_tr sebelum tiap fase.
    SpecAugment hanya untuk training, bukan validasi.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Augmentasi data training kalau diminta
    X_train_input = apply_specaugment(X_tr) if use_specaugment else X_tr
    y_tr_oh = to_categorical(y_tr, N_CLASSES)
    y_vl_oh = to_categorical(y_vl, N_CLASSES)

    # ── Phase 1 ──────────────────────────────────────────────────
    print(f"\n[{model_name}] Phase 1: Training head (backbone frozen)...")
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    hist1 = model.fit(
        X_train_input, y_tr_oh,
        validation_data=(X_vl, y_vl_oh),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_FROZEN,
        callbacks=[
            CSVLogger(os.path.join(model_dir, f"{model_name}_phase1.csv")),
            ModelCheckpoint(os.path.join(model_dir, f"{model_name}_best.keras"),
                            monitor='val_accuracy', save_best_only=True, verbose=0),
        ],
        verbose=1
    )

    # ── Phase 2 ──────────────────────────────────────────────────
    print(f"\n[{model_name}] Phase 2: Fine-tuning semua layer...")
    base_model.trainable = True

    # Augmentasi ulang untuk phase 2 (seed berbeda karena random)
    X_train_input2 = apply_specaugment(X_tr) if use_specaugment else X_tr

    model.compile(
        optimizer=Adam(LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    hist2 = model.fit(
        X_train_input2, y_tr_oh,
        validation_data=(X_vl, y_vl_oh),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_UNFROZEN,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(os.path.join(model_dir, f"{model_name}_best.keras"),
                            monitor='val_accuracy', save_best_only=True, verbose=0),
            CSVLogger(os.path.join(model_dir, f"{model_name}_phase2.csv")),
        ],
        verbose=1
    )
    return hist1, hist2


def plot_training_history(hist1, hist2, model_name):
    """Plot kurva training gabungan phase 1 + phase 2."""
    acc      = hist1.history['accuracy']     + hist2.history['accuracy']
    val_acc  = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    loss     = hist1.history['loss']         + hist2.history['loss']
    val_loss = hist1.history['val_loss']     + hist2.history['val_loss']
    p1_end   = len(hist1.history['accuracy'])
    epochs   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History — {model_name}', fontsize=13)
    for ax, (tr, vl, lbl) in zip([ax1, ax2], [
        (acc, val_acc, 'Accuracy'), (loss, val_loss, 'Loss')
    ]):
        ax.plot(epochs, tr, label=f'Train {lbl}')
        ax.plot(epochs, vl, label=f'Val {lbl}')
        ax.axvline(x=p1_end, color='gray', linestyle='--',
                   label=f'Phase 2 (ep {p1_end+1})')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.set_title(lbl)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, f"training_{model_name}.png")
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Kurva training disimpan ke {out}")


# ── MAIN: ABLATION STUDY ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("STEP 5: Training EfficientNetB0 + SE Block + SpecAugment")
    print("=" * 60)
    print("Ablation Study:")
    print("  B1: Single Mel, No DIP, No SpecAugment")
    print("  B2: Single Mel, DIP, SpecAugment")
    print("  B3: 4-Channel, No DIP, No SpecAugment")
    print("  B4: 4-Channel, DIP, SpecAugment  ← PROPOSED METHOD")
    print()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_npz()
    X_tr_raw, y_tr_raw, X_vl_raw, y_vl_raw, X_ts_raw, y_ts_raw = load_b1_b3_data()

    # Demo SpecAugment
    visualize_specaugment(X_train, idx=0)

    ablation = {}

    # ── B1 ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("B1: Single-Mel, TANPA DIP, TANPA SpecAugment — baseline")
    X_b1 = prepare_ablation_data(X_tr_raw, 'B1')
    model_b1, base_b1 = build_model_3ch()
    h1a, h1b = train_model(model_b1, base_b1,
                            X_b1, y_tr_raw,
                            prepare_ablation_data(X_vl_raw, 'B1'), y_vl_raw,
                            'B1_mel_nodip_noaug', MODEL_DIR,
                            use_specaugment=False)
    plot_training_history(h1a, h1b, 'B1')
    ablation['B1'] = {'model': model_b1,
                      'X_test': prepare_ablation_data(X_ts_raw, 'B1'),
                      'y_test': y_ts_raw}

    # ── B2 ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("B2: Single-Mel, DIP, SpecAugment")
    X_b2 = prepare_ablation_data(X_train, 'B2')
    model_b2, base_b2 = build_model_3ch()
    h2a, h2b = train_model(model_b2, base_b2,
                            X_b2, y_train,
                            prepare_ablation_data(X_val, 'B2'), y_val,
                            'B2_mel_dip_aug', MODEL_DIR,
                            use_specaugment=True)
    plot_training_history(h2a, h2b, 'B2')
    ablation['B2'] = {'model': model_b2,
                      'X_test': prepare_ablation_data(X_test, 'B2'),
                      'y_test': y_test}

    # ── B3 ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("B3: 4-Channel, TANPA DIP, TANPA SpecAugment")
    model_b3, base_b3 = build_model_4ch()
    h3a, h3b = train_model(model_b3, base_b3,
                            X_tr_raw, y_tr_raw,
                            X_vl_raw, y_vl_raw,
                            'B3_4ch_nodip_noaug', MODEL_DIR,
                            use_specaugment=False)
    plot_training_history(h3a, h3b, 'B3')
    ablation['B3'] = {'model': model_b3,
                      'X_test': X_ts_raw,
                      'y_test': y_ts_raw}

    # ── B4 ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("B4: 4-Channel, DIP, SpecAugment — PROPOSED METHOD")
    model_b4, base_b4 = build_model_4ch()
    h4a, h4b = train_model(model_b4, base_b4,
                            X_train, y_train,
                            X_val, y_val,
                            'B4_4ch_dip_aug', MODEL_DIR,
                            use_specaugment=True)
    plot_training_history(h4a, h4b, 'B4')
    ablation['B4'] = {'model': model_b4,
                      'X_test': X_test,
                      'y_test': y_test}

    # ── Ringkasan ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RINGKASAN ABLATION STUDY")
    print(f"{'Kondisi':<12} {'Test Accuracy':>14}")
    print("-" * 28)
    summary = {}
    for cond, r in ablation.items():
        y_oh = to_categorical(r['y_test'], N_CLASSES)
        _, acc = r['model'].evaluate(
            prepare_ablation_data(r['X_test'], cond), y_oh, verbose=0
        )
        summary[cond] = round(float(acc), 4)
        print(f"{cond:<12} {acc*100:>13.2f}%")

    with open(os.path.join(RESULT_DIR, "ablation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nStep 5 selesai. Lanjut ke step6_evaluate_gradcam.py")
