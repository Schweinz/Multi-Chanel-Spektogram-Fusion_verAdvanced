"""
step5_train.py
==============
STEP KELIMA — training MobileNetV2 + SE Block + SpecAugment.

Novelty utama Paper 2 (Advanced) vs Paper 1:

1. SE Block (Squeeze-and-Excitation) setelah backbone
   - Belajar bobot kepentingan tiap channel feature map secara otomatis
   - Terinspirasi dari hasil Grad-CAM Paper 1 yang menunjukkan channel
     importance berbeda per emosi — SE Block mengeksplisitkan ini secara
     arsitektural, bukan hanya dianalisis post-hoc

2. SpecAugment pada data training (Park et al., Interspeech 2019)
   - Mask frekuensi: pilih F baris random, set ke nol
   - Mask waktu    : pilih T kolom random, set ke nol
   - Hanya diaplikasikan saat training, bukan saat validasi/test

3. Ablation study B1-B4
   - B1: Mel saja, tanpa DIP, tanpa SpecAugment (baseline)
   - B2: Mel+DIP+SpecAugment
   - B3: 4-channel, tanpa DIP, tanpa SpecAugment
   - B4: 4-channel+DIP+SpecAugment (proposed method)

Arsitektur:
  Input (128x128x4) -> Conv2D(3,1x1) -> MobileNetV2
  -> SE Block -> GAP -> Dropout -> Dense(128) -> BN -> Dropout -> Dense(8)

CATATAN: MobileNetV2 digunakan karena EfficientNetB0 menghasilkan
spatial output 4x4 untuk input 128x128 yang membuat model sulit belajar.
MobileNetV2 (alpha=1.0) menghasilkan 4x4x1280 — cukup representatif.
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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input, Conv2D, GlobalAveragePooling2D,
    Dropout, Dense, BatchNormalization,
    Reshape, Multiply
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
    Frequency masking + Time masking.
    X: array (N, 128, 128, C) float32
    """
    X_aug = X.copy()
    N, H, W, C = X_aug.shape
    for i in range(N):
        for _ in range(SPEC_AUG_NUM_FREQ_MASKS):
            f = np.random.randint(0, SPEC_AUG_FREQ_MASK_MAX + 1)
            f0 = np.random.randint(0, max(1, H - f))
            X_aug[i, f0:f0 + f, :, :] = 0.0
        for _ in range(SPEC_AUG_NUM_TIME_MASKS):
            t = np.random.randint(0, SPEC_AUG_TIME_MASK_MAX + 1)
            t0 = np.random.randint(0, max(1, W - t))
            X_aug[i, :, t0:t0 + t, :] = 0.0
    return X_aug


def visualize_specaugment(X_sample, idx=0):
    """Tampilkan satu sampel sebelum dan sesudah SpecAugment."""
    original  = X_sample[idx:idx+1]
    augmented = apply_specaugment(original.copy())
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("SpecAugment — Sebelum vs Sesudah", fontsize=13)
    for col, ch in enumerate(CHANNEL_ORDER):
        axes[0, col].imshow(original[0, :, :, col],  cmap='viridis', aspect='auto', origin='lower')
        axes[0, col].set_title(f'{ch.upper()}\nAsli')
        axes[0, col].axis('off')
        axes[1, col].imshow(augmented[0, :, :, col], cmap='viridis', aspect='auto', origin='lower')
        axes[1, col].set_title(f'{ch.upper()}\nSetelah SpecAugment')
        axes[1, col].axis('off')
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "specaugment_demo.png")
    os.makedirs(RESULT_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"SpecAugment demo disimpan ke {save_path}")


# ── SE BLOCK ──────────────────────────────────────────────────────────────────

def se_block(x, ratio=SE_RATIO):
    """
    Squeeze-and-Excitation Block — Hu et al., CVPR 2018.
    Belajar bobot kepentingan tiap channel feature map secara otomatis.
    """
    channels   = x.shape[-1]
    squeeze    = GlobalAveragePooling2D()(x)
    excitation = Dense(max(1, channels // ratio), activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, channels))(excitation)
    return Multiply()([x, excitation])


# ── ARSITEKTUR MODEL ──────────────────────────────────────────────────────────

def build_model_4ch(n_classes=N_CLASSES):
    """Model 4-channel: MobileNetV2 + SE Block."""
    inp  = Input(shape=(128, 128, 4), name='input_4ch')
    x    = Conv2D(3, (1, 1), padding='same', name='channel_projection')(inp)
    base = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )
    base.trainable = False
    x   = base(x, training=False)
    x   = se_block(x, ratio=SE_RATIO)
    x   = GlobalAveragePooling2D(name='gap')(x)
    x   = Dropout(DROPOUT_RATE)(x)
    x   = Dense(128, activation='relu', name='fc1')(x)
    x   = BatchNormalization()(x)
    x   = Dropout(DROPOUT_RATE)(x)
    out = Dense(n_classes, activation='softmax', name='output')(x)
    return Model(inp, out, name='SER_MobileNetV2_SE_4ch'), base


def build_model_3ch(n_classes=N_CLASSES):
    """Model 3-channel (Mel saja): MobileNetV2 + SE Block. Untuk B1/B2."""
    base = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )
    base.trainable = False
    inp = Input(shape=(128, 128, 3), name='input_3ch')
    x   = base(inp, training=False)
    x   = se_block(x, ratio=SE_RATIO)
    x   = GlobalAveragePooling2D(name='gap')(x)
    x   = Dropout(DROPOUT_RATE)(x)
    x   = Dense(128, activation='relu', name='fc1')(x)
    x   = BatchNormalization()(x)
    x   = Dropout(DROPOUT_RATE)(x)
    out = Dense(n_classes, activation='softmax', name='output')(x)
    return Model(inp, out, name='SER_MobileNetV2_SE_3ch'), base


# ── LOAD DATASET ──────────────────────────────────────────────────────────────

def load_dataset_npz():
    data_path = os.path.join(FUSED_DIR, "dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} tidak ditemukan. Jalankan step4 dulu.")
    data = np.load(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    print(f"Dataset dimuat: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"X range: [{X_train.min():.3f}, {X_train.max():.3f}], dtype={X_train.dtype}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_b1_b3_data():
    """Data tanpa DIP untuk kondisi B1/B3."""
    from step4_build_dataset import build_full_dataset, split_dataset
    print("Membangun dataset B1/B3 dari SPEC_DIR (tanpa DIP)...")
    X_raw, y_raw = build_full_dataset(enhanced_dir=SPEC_DIR)
    return split_dataset(X_raw, y_raw)


def prepare_ablation_data(X, condition):
    """B1/B2: duplikasi Mel -> 3ch. B3/B4: pakai 4ch langsung."""
    if condition in ('B1', 'B2'):
        mel = X[:, :, :, 0:1]
        return np.concatenate([mel, mel, mel], axis=-1)
    return X


# ── TRAINING 2 FASE ───────────────────────────────────────────────────────────

def train_model(model, base_model, X_tr, y_tr, X_vl, y_vl,
                model_name, model_dir, use_specaugment=False):
    """
    Training 2 fase:
    Phase 1 — FROZEN   (lr=1e-3, 30 epoch)
    Phase 2 — UNFROZEN (lr=1e-4, max 100 epoch, EarlyStopping patience=15)
    """
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, f"{model_name}_best.keras")

    X_input1 = apply_specaugment(X_tr) if use_specaugment else X_tr
    y_tr_oh  = to_categorical(y_tr, N_CLASSES)
    y_vl_oh  = to_categorical(y_vl, N_CLASSES)

    # Phase 1: frozen backbone
    print(f"\n[{model_name}] Phase 1: Training head (frozen, lr=1e-3, 30 ep)...")
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    hist1 = model.fit(
        X_input1, y_tr_oh,
        validation_data=(X_vl, y_vl_oh),
        batch_size=BATCH_SIZE, epochs=30,
        callbacks=[
            ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=0),
            CSVLogger(os.path.join(model_dir, f"{model_name}_phase1.csv")),
        ], verbose=1
    )

    # Phase 2: unfreeze backbone
    print(f"\n[{model_name}] Phase 2: Fine-tuning (unfrozen, lr=1e-4, max 100 ep)...")
    base_model.trainable = True
    X_input2 = apply_specaugment(X_tr) if use_specaugment else X_tr
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    hist2 = model.fit(
        X_input2, y_tr_oh,
        validation_data=(X_vl, y_vl_oh),
        batch_size=BATCH_SIZE, epochs=EPOCHS_UNFROZEN,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=0),
            CSVLogger(os.path.join(model_dir, f"{model_name}_phase2.csv")),
        ], verbose=1
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
        ax.axvline(x=p1_end, color='gray', linestyle='--', label=f'Phase 2 mulai (ep {p1_end+1})')
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
    print("STEP 5: Training MobileNetV2 + SE Block + SpecAugment")
    print("=" * 60)
    print("B1: Single Mel, No DIP, No SpecAugment  <- baseline")
    print("B2: Single Mel, DIP, SpecAugment")
    print("B3: 4-Channel, No DIP, No SpecAugment")
    print("B4: 4-Channel, DIP, SpecAugment          <- PROPOSED METHOD")
    print()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Dataset dengan DIP (B2, B4)
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_npz()
    # Dataset tanpa DIP (B1, B3)
    X_tr_raw, y_tr_raw, X_vl_raw, y_vl_raw, X_ts_raw, y_ts_raw = load_b1_b3_data()

    visualize_specaugment(X_train, idx=0)

    ablation = {}

    # B1: Mel, No DIP, No Aug
    print("\n" + "="*60)
    print("B1: Single-Mel, TANPA DIP, TANPA SpecAugment")
    model_b1, base_b1 = build_model_3ch()
    h1a, h1b = train_model(model_b1, base_b1,
        prepare_ablation_data(X_tr_raw, 'B1'), y_tr_raw,
        prepare_ablation_data(X_vl_raw, 'B1'), y_vl_raw,
        'B1_mel_nodip_noaug', MODEL_DIR, use_specaugment=False)
    plot_training_history(h1a, h1b, 'B1')
    ablation['B1'] = {'model': model_b1, 'X_test': prepare_ablation_data(X_ts_raw, 'B1'),
                      'y_test': y_ts_raw, 'description': 'Single Mel, No DIP, No SpecAugment'}

    # B2: Mel, DIP, Aug
    print("\n" + "="*60)
    print("B2: Single-Mel, DIP, SpecAugment")
    model_b2, base_b2 = build_model_3ch()
    h2a, h2b = train_model(model_b2, base_b2,
        prepare_ablation_data(X_train, 'B2'), y_train,
        prepare_ablation_data(X_val, 'B2'), y_val,
        'B2_mel_dip_aug', MODEL_DIR, use_specaugment=True)
    plot_training_history(h2a, h2b, 'B2')
    ablation['B2'] = {'model': model_b2, 'X_test': prepare_ablation_data(X_test, 'B2'),
                      'y_test': y_test, 'description': 'Single Mel, DIP, SpecAugment'}

    # B3: 4ch, No DIP, No Aug
    print("\n" + "="*60)
    print("B3: 4-Channel, TANPA DIP, TANPA SpecAugment")
    model_b3, base_b3 = build_model_4ch()
    h3a, h3b = train_model(model_b3, base_b3,
        X_tr_raw, y_tr_raw, X_vl_raw, y_vl_raw,
        'B3_4ch_nodip_noaug', MODEL_DIR, use_specaugment=False)
    plot_training_history(h3a, h3b, 'B3')
    ablation['B3'] = {'model': model_b3, 'X_test': X_ts_raw,
                      'y_test': y_ts_raw, 'description': '4-Channel, No DIP, No SpecAugment'}

    # B4: 4ch, DIP, Aug — PROPOSED
    print("\n" + "="*60)
    print("B4: 4-Channel, DIP, SpecAugment — PROPOSED METHOD")
    model_b4, base_b4 = build_model_4ch()
    h4a, h4b = train_model(model_b4, base_b4,
        X_train, y_train, X_val, y_val,
        'B4_4ch_dip_aug', MODEL_DIR, use_specaugment=True)
    plot_training_history(h4a, h4b, 'B4')
    ablation['B4'] = {'model': model_b4, 'X_test': X_test,
                      'y_test': y_test, 'description': '4-Channel, DIP, SpecAugment (Proposed)'}

    # Ringkasan
    print("\n" + "="*60)
    print("RINGKASAN ABLATION STUDY")
    print(f"{'Kondisi':<8} {'Test Accuracy':>14} {'F1 Weighted':>12}  Deskripsi")
    print("-" * 60)

    from sklearn.metrics import f1_score as sk_f1

    summary = []
    for cond, r in ablation.items():
        y_pred = np.argmax(r['model'].predict(r['X_test'], verbose=0), axis=1)
        acc    = float(np.mean(y_pred == r['y_test']))
        f1     = float(sk_f1(r['y_test'], y_pred, average='weighted'))
        summary.append({
            'name': cond, 'accuracy': round(acc, 4),
            'f1_weighted': round(f1, 4), 'description': r['description']
        })
        print(f"{cond:<8} {acc*100:>13.2f}% {f1*100:>11.2f}%  {r['description']}")

    with open(os.path.join(RESULT_DIR, "ablation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    best = max(summary, key=lambda x: x['accuracy'])
    print(f"\nTerbaik: {best['name']} — Accuracy {best['accuracy']*100:.2f}%, F1 {best['f1_weighted']*100:.2f}%")
    print("\nStep 5 selesai. Lanjut ke step6_evaluate_gradcam.py")
