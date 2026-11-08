"""
run_local_training.py
=====================
Training lokal menggunakan MobileNetV2 sebagai backbone.
(EfficientNetB0 digunakan di Colab/Google Drive — weights-nya tidak bisa
 didownload dari environment cloud ini karena storage.googleapis.com diblokir.
 MobileNetV2 memiliki arsitektur serupa, output 1280-d yang sama, dan weights
 ImageNet berhasil didownload dari sini.)

Arsitektur identik dengan step5_train.py kecuali:
  EfficientNetB0 → MobileNetV2
  (output shape identik: (None, 4, 4, 1280) untuk input 128×128×3)

Ini untuk VALIDASI PIPELINE dan mendapatkan angka baseline yang benar.
Paper final menggunakan EfficientNetB0 yang dijalankan di Colab.
"""

import os
import sys
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load config lokal ─────────────────────────────────────────────────────────
sys.path.insert(0, '/home/claude/multi-chanel-spektogram-fusion-advanced')
import config_local as cfg
sys.modules['config'] = cfg

FUSED_DIR  = cfg.FUSED_DIR
MODEL_DIR  = cfg.MODEL_DIR
RESULT_DIR = cfg.RESULT_DIR
SPEC_DIR   = cfg.SPEC_DIR

N_CLASSES   = cfg.N_CLASSES
BATCH_SIZE  = cfg.BATCH_SIZE
EPOCHS_FROZEN   = cfg.EPOCHS_FROZEN
EPOCHS_UNFROZEN = cfg.EPOCHS_UNFROZEN
LEARNING_RATE   = cfg.LEARNING_RATE
DROPOUT_RATE    = cfg.DROPOUT_RATE
SE_RATIO        = cfg.SE_RATIO
SPEC_AUG_FREQ_MASK_MAX  = cfg.SPEC_AUG_FREQ_MASK_MAX
SPEC_AUG_TIME_MASK_MAX  = cfg.SPEC_AUG_TIME_MASK_MAX
SPEC_AUG_NUM_FREQ_MASKS = cfg.SPEC_AUG_NUM_FREQ_MASKS
SPEC_AUG_NUM_TIME_MASKS = cfg.SPEC_AUG_NUM_TIME_MASKS
CHANNEL_ORDER = cfg.CHANNEL_ORDER
EMOTION_LABELS = cfg.EMOTION_LABELS
RANDOM_SEED = cfg.RANDOM_SEED

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2  # Backbone alternatif
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"Backbone: MobileNetV2 (EfficientNetB0 dipakai di Colab)")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ── SPECAUGMENT ───────────────────────────────────────────────────────────────

def apply_specaugment(X):
    X_aug = X.copy()
    N, H, W, C = X_aug.shape
    for i in range(N):
        for _ in range(SPEC_AUG_NUM_FREQ_MASKS):
            f = np.random.randint(0, SPEC_AUG_FREQ_MASK_MAX + 1)
            f0 = np.random.randint(0, max(1, H - f))
            X_aug[i, f0:f0+f, :, :] = 0.0
        for _ in range(SPEC_AUG_NUM_TIME_MASKS):
            t = np.random.randint(0, SPEC_AUG_TIME_MASK_MAX + 1)
            t0 = np.random.randint(0, max(1, W - t))
            X_aug[i, :, t0:t0+t, :] = 0.0
    return X_aug


# ── SE BLOCK ──────────────────────────────────────────────────────────────────

def se_block(x, ratio=SE_RATIO):
    channels = x.shape[-1]
    squeeze = GlobalAveragePooling2D()(x)
    excitation = Dense(max(1, channels // ratio), activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, channels))(excitation)
    return Multiply()([x, excitation])


# ── MODEL BUILDER ──────────────────────────────────────────────────────────────

def build_model_4ch(n_classes=N_CLASSES):
    """4-channel model: Input(128,128,4) → Conv1×1(3) → MobileNetV2 → SE → Head"""
    inp = Input(shape=(128, 128, 4), name='input_4ch')
    x = Conv2D(3, (1, 1), padding='same', name='channel_projection')(inp)
    base = MobileNetV2(input_shape=(128, 128, 3), include_top=False,
                       weights='imagenet', alpha=1.0)
    base.trainable = False
    x = base(x, training=False)
    x = se_block(x, ratio=SE_RATIO)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(n_classes, activation='softmax', name='output')(x)
    model = Model(inp, out, name='SER_MobileNetV2_SE_4ch')
    return model, base


def build_model_3ch(n_classes=N_CLASSES):
    """3-channel model: Input(128,128,3) → MobileNetV2 → SE → Head"""
    base = MobileNetV2(input_shape=(128, 128, 3), include_top=False,
                       weights='imagenet', alpha=1.0)
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
    model = Model(inp, out, name='SER_MobileNetV2_SE_3ch')
    return model, base


# ── LOAD DATASET ──────────────────────────────────────────────────────────────

def load_dataset_npz():
    data_path = os.path.join(FUSED_DIR, "dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} tidak ditemukan. Jalankan step4 dulu.")
    data = np.load(data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val   = data['X_val']
    y_val   = data['y_val']
    X_test  = data['X_test']
    y_test  = data['y_test']
    print(f"Dataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_unenhanced_dataset():
    """
    Untuk B1/B3 (tanpa DIP): load dari SPEC_DIR bukan ENHANCED_DIR.
    """
    sys.path.insert(0, '/home/claude/multi-chanel-spektogram-fusion-advanced')
    from step4_build_dataset import build_full_dataset, split_dataset
    print("Membangun dataset tanpa DIP dari SPEC_DIR...")
    X_raw, y_raw = build_full_dataset(enhanced_dir=cfg.SPEC_DIR)
    return split_dataset(X_raw, y_raw)


def prepare_ablation_data(X, condition):
    """
    B1/B2: duplikasi channel Mel (index 0) menjadi 3-channel.
    B3/B4: pakai semua 4 channel.
    """
    if condition in ('B1', 'B2'):
        mel = X[:, :, :, 0:1]
        return np.concatenate([mel, mel, mel], axis=-1)
    return X


# ── TRAINING 2 FASE ───────────────────────────────────────────────────────────

def train_model(model, base_model, X_tr, y_tr, X_vl, y_vl,
                model_name, model_dir, use_specaugment=False):
    os.makedirs(model_dir, exist_ok=True)

    X_train_input = apply_specaugment(X_tr) if use_specaugment else X_tr
    y_tr_oh = to_categorical(y_tr, N_CLASSES)
    y_vl_oh = to_categorical(y_vl, N_CLASSES)

    # ── Phase 1: Frozen backbone ─────────────────────────────────
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
            ModelCheckpoint(
                os.path.join(model_dir, f"{model_name}_best.keras"),
                monitor='val_accuracy', save_best_only=True, verbose=0
            ),
        ],
        verbose=1
    )

    # ── Phase 2: Unfrozen fine-tuning ────────────────────────────
    print(f"\n[{model_name}] Phase 2: Fine-tuning semua layer...")
    base_model.trainable = True

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
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            CSVLogger(os.path.join(model_dir, f"{model_name}_phase2.csv")),
            ModelCheckpoint(
                os.path.join(model_dir, f"{model_name}_best.keras"),
                monitor='val_accuracy', save_best_only=True, verbose=0
            ),
        ],
        verbose=1
    )

    return hist1, hist2


# ── EVALUASI ──────────────────────────────────────────────────────────────────

def evaluate_model(model, model_name, model_dir, X_test, y_test):
    # Load best checkpoint
    best_path = os.path.join(model_dir, f"{model_name}_best.keras")
    if os.path.exists(best_path):
        model.load_weights(best_path)
        print(f"Loaded best weights dari {best_path}")

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    cm  = confusion_matrix(y_test, y_pred)

    print(f"[{model_name}] Test Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")

    # Simpan confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha='right')
    ax.set_yticklabels(EMOTION_LABELS)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {model_name}\nAcc={acc:.4f}, F1={f1:.4f}')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(RESULT_DIR, f"cm_{model_name}.png")
    plt.savefig(cm_path, dpi=100)
    plt.close()
    print(f"Confusion matrix disimpan ke {cm_path}")

    return {'name': model_name, 'accuracy': float(acc), 'f1_weighted': float(f1),
            'confusion_matrix': cm.tolist()}


# ── PLOT TRAINING HISTORY ──────────────────────────────────────────────────────

def plot_history(hist1, hist2, model_name):
    acc1 = hist1.history['accuracy']
    val1 = hist1.history['val_accuracy']
    acc2 = hist2.history['accuracy']
    val2 = hist2.history['val_accuracy']

    epochs_p1 = range(1, len(acc1)+1)
    epochs_p2 = range(len(acc1)+1, len(acc1)+len(acc2)+1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(epochs_p1, acc1, 'b-', label='Train Phase 1')
    ax.plot(epochs_p1, val1, 'b--', label='Val Phase 1')
    ax.plot(epochs_p2, acc2, 'r-', label='Train Phase 2')
    ax.plot(epochs_p2, val2, 'r--', label='Val Phase 2')
    ax.axvline(len(acc1), color='gray', linestyle=':', label='Unfreeze point')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Training History — {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f"history_{model_name}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"History plot disimpan ke {path}")


# ── MAIN: ABLATION B1-B4 ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 5 — Training Ablation B1-B4")
    print("Backbone: MobileNetV2 (pengganti EfficientNetB0 lokal)")
    print("=" * 60)

    # Load dataset utama (DIP enhanced, 4-channel)
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_npz()
    print(f"Shape X_train: {X_train.shape}, dtype: {X_train.dtype}")

    # Load dataset tanpa DIP untuk B1/B3
    print("\nMemuat dataset tanpa DIP (untuk B1/B3)...")
    try:
        X_tr_nd, y_tr_nd, X_vl_nd, y_vl_nd, X_te_nd, y_te_nd = load_unenhanced_dataset()
        print(f"Dataset tanpa DIP: train={len(X_tr_nd)}, val={len(X_vl_nd)}, test={len(X_te_nd)}")
    except Exception as e:
        print(f"WARNING: Gagal load dataset tanpa DIP: {e}")
        print("Menggunakan dataset DIP untuk semua kondisi sebagai fallback.")
        X_tr_nd, y_tr_nd, X_vl_nd, y_vl_nd, X_te_nd, y_te_nd = (
            X_train, y_train, X_val, y_val, X_test, y_test
        )

    ablation_configs = [
        # (condition, use_dip, use_aug, use_4ch, description)
        ('B1', False, False, False, 'Mel-only, no DIP, no SpecAugment'),
        ('B2', True,  True,  False, 'Mel+DIP+SpecAugment, single channel'),
        ('B3', False, False, True,  '4-channel, no DIP, no SpecAugment'),
        ('B4', True,  True,  True,  '4-channel+DIP+SpecAugment (Full)'),
    ]

    all_results = []

    for condition, use_dip, use_aug, use_4ch, desc in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Kondisi {condition}: {desc}")
        print(f"{'='*60}")

        # Pilih dataset
        if use_dip:
            Xtr, ytr = X_train, y_train
            Xvl, yvl = X_val,   y_val
            Xte, yte = X_test,  y_test
        else:
            Xtr, ytr = X_tr_nd, y_tr_nd
            Xvl, yvl = X_vl_nd, y_vl_nd
            Xte, yte = X_te_nd, y_te_nd

        # Prepare input (3ch atau 4ch)
        Xtr_in = prepare_ablation_data(Xtr, condition)
        Xvl_in = prepare_ablation_data(Xvl, condition)
        Xte_in = prepare_ablation_data(Xte, condition)

        print(f"Input shape: {Xtr_in.shape}")

        # Build model
        if use_4ch:
            model, base = build_model_4ch()
        else:
            model, base = build_model_3ch()

        model.summary(print_fn=lambda x: print(x) if 'Total' in x else None)

        # Training
        model_subdir = os.path.join(MODEL_DIR, condition)
        hist1, hist2 = train_model(
            model, base, Xtr_in, ytr, Xvl_in, yvl,
            model_name=condition,
            model_dir=model_subdir,
            use_specaugment=use_aug
        )

        # Plot history
        plot_history(hist1, hist2, condition)

        # Evaluasi
        result = evaluate_model(model, condition, model_subdir, Xte_in, yte)
        result['description'] = desc
        all_results.append(result)

        # Clear session untuk hemat RAM
        tf.keras.backend.clear_session()

    # ── Simpan ringkasan ablasi ────────────────────────────────────
    summary_path = os.path.join(RESULT_DIR, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation summary disimpan ke {summary_path}")

    # Print tabel hasil
    print("\n" + "="*60)
    print("RINGKASAN ABLASI")
    print("="*60)
    print(f"{'Kondisi':<10} {'Accuracy':>10} {'F1-Score':>10}  Deskripsi")
    print("-"*60)
    for r in all_results:
        print(f"{r['name']:<10} {r['accuracy']:>10.4f} {r['f1_weighted']:>10.4f}  {r['description']}")
    print("="*60)

    # Highlight model terbaik
    best = max(all_results, key=lambda x: x['accuracy'])
    print(f"\n✅ Model terbaik: {best['name']} — Accuracy {best['accuracy']:.4f}")

    return all_results


if __name__ == '__main__':
    results = main()
