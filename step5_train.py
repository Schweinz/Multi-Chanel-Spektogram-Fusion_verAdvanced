"""
step5_train.py  --  Opsi C (RAM-efficient): ResNet50V2 + SE Block + BiGRU
==========================================================================
SOLUSI RAM CRASH:
  - Load PNG on-the-fly dari disk (TIDAK ada dataset.npz di RAM)
  - Mixed precision float16 (RAM GPU turun ~40%)
  - BATCH_SIZE=8 (aman untuk Colab free tier)
  - SSD local copy: copy enhanced_spectrograms ke /tmp/specs/ sekali diawal

RAM usage estimasi:
  Metode lama (npz ke RAM): ~4-6 GB RAM
  Metode ini (on-the-fly)  : ~400-600 MB RAM

Arsitektur (Opsi C):
  Input  : 224x224x4
  Proj   : Conv2D(3, 1x1) -- 4ch -> 3ch
  Backbone: ResNet50V2 pretrained ImageNet (7x7x2048)
  SE     : Squeeze-and-Excitation (channel attention, ratio=16)
  BiGRU  : Reshape (49,2048) -> Bidirectional GRU(64)
  Head   : Dense(128) + BN + Dropout(0.5) -> Softmax(8)

Ablasi:
  B1: Mel only, tanpa DIP, tanpa augmentasi
  B2: Mel only, dengan DIP, dengan augmentasi
  B3: 4ch, tanpa DIP, tanpa augmentasi
  B4: 4ch, dengan DIP, dengan augmentasi (kondisi utama)
"""

import os
import sys
import json
import time
import glob
import shutil
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Self-patching config
# ---------------------------------------------------------------------------
BASE_DIR = '/content/drive/MyDrive/SER_Advanced'
sys.path.insert(0, BASE_DIR)
import config as _cfg
_cfg.BASE_DIR     = BASE_DIR
_cfg.DATA_DIR     = os.path.join(BASE_DIR, 'ravdess_data')
_cfg.RAW_DATA_DIR = os.path.join(BASE_DIR, 'ravdess_data')
_cfg.SPEC_DIR     = os.path.join(BASE_DIR, 'spectrograms')
_cfg.ENHANCED_DIR = os.path.join(BASE_DIR, 'enhanced_spectrograms')
_cfg.FUSED_DIR    = os.path.join(BASE_DIR, 'fused_dataset')
_cfg.MODEL_DIR    = os.path.join(BASE_DIR, 'models')
_cfg.RESULT_DIR   = os.path.join(BASE_DIR, 'results')
_cfg.IMG_SIZE     = (224, 224)
_cfg.create_dirs()

SPEC_DIR_DRIVE    = _cfg.SPEC_DIR        # spectrograms/ (tanpa DIP)
ENHANCED_DIR_DRIVE = _cfg.ENHANCED_DIR   # enhanced_spectrograms/ (dengan DIP)
FUSED_DIR         = _cfg.FUSED_DIR
MODEL_DIR         = _cfg.MODEL_DIR
RESULT_DIR        = _cfg.RESULT_DIR
N_CLASSES         = _cfg.N_CLASSES
EMOTION_LABELS    = _cfg.EMOTION_LABELS
EMOTION_MAP       = _cfg.EMOTION_MAP
IMG_SIZE          = _cfg.IMG_SIZE         # (224, 224)
CHANNEL_ORDER     = _cfg.CHANNEL_ORDER    # ['mel','mfcc','cqt','chroma']
RANDOM_SEED       = _cfg.RANDOM_SEED
TEST_SIZE         = _cfg.TEST_SIZE
VAL_SIZE          = _cfg.VAL_SIZE

SE_RATIO               = _cfg.SE_RATIO
SPEC_AUG_FREQ_MASK_MAX = _cfg.SPEC_AUG_FREQ_MASK_MAX
SPEC_AUG_TIME_MASK_MAX = _cfg.SPEC_AUG_TIME_MASK_MAX
SPEC_AUG_NUM_FREQ_MASKS = _cfg.SPEC_AUG_NUM_FREQ_MASKS
SPEC_AUG_NUM_TIME_MASKS = _cfg.SPEC_AUG_NUM_TIME_MASKS

# ---------------------------------------------------------------------------
# Hyperparameter
# ---------------------------------------------------------------------------
BATCH_SIZE      = 8          # kecil agar RAM aman di Colab free
LR_PHASE1       = 5e-4
LR_PHASE2       = 5e-5
DROPOUT_RATE    = 0.5
L2_REG          = 1e-4
LABEL_SMOOTHING = 0.1
EPOCHS_PHASE1   = 30
EPOCHS_PHASE2   = 100
PATIENCE        = 20
UNFREEZE_LAST_N = 100

# Local SSD cache (jauh lebih cepat dibanding baca dari Drive tiap epoch)
LOCAL_SPEC_DIR     = '/tmp/specs_nodip'   # tanpa DIP (B1/B3)
LOCAL_ENHANCED_DIR = '/tmp/specs_dip'     # dengan DIP (B2/B4)

import tensorflow as tf

# ---------------------------------------------------------------------------
# MIXED PRECISION: GPU RAM turun ~40%, training lebih cepat di T4/V100
# ---------------------------------------------------------------------------
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"Precision: {tf.keras.mixed_precision.global_policy()}")


# ===========================================================================
# Utilitas: copy folder dari Drive ke local SSD /tmp/
# ===========================================================================

def copy_to_local(src_dir, dst_dir, desc=""):
    """
    Copy semua PNG dari Google Drive ke /tmp/ (local SSD).
    Hanya dilakukan sekali per session. Jauh lebih cepat saat training.
    """
    if os.path.exists(dst_dir) and len(glob.glob(f"{dst_dir}/*.png")) > 0:
        n = len(glob.glob(f"{dst_dir}/*.png"))
        print(f"[cache] {desc}: {n} PNG sudah ada di {dst_dir}")
        return dst_dir

    print(f"[copy] {desc}: {src_dir} -> {dst_dir} ...")
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(f"{src_dir}/*.png")
    if not files:
        print(f"  [!] Tidak ada PNG di {src_dir}")
        return src_dir  # fallback ke Drive
    for f in files:
        shutil.copy2(f, dst_dir)
    print(f"  Selesai: {len(files)} file disalin")
    return dst_dir


# ===========================================================================
# Parsing label RAVDESS
# ===========================================================================

def parse_ravdess_label(filename):
    """
    Ekstrak label integer [0,7] dari nama file RAVDESS.
    Format: 03-01-[EMOSI]-..._{channel}.png
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


# ===========================================================================
# Bangun daftar file (paths + labels) tanpa load pixel ke RAM
# ===========================================================================

def build_file_list(spec_dir):
    """
    Scan semua PNG, kelompokkan per sampel (4 channel per sampel).
    Return: list of (mel_path, mfcc_path, cqt_path, chroma_path, label_int)
    Tidak ada array numpy besar -- hanya list of strings!
    """
    all_png = glob.glob(f"{spec_dir}/*.png")
    if not all_png:
        raise FileNotFoundError(f"Tidak ada PNG di {spec_dir}")

    # Kelompokkan per basename sampel
    sample_files = defaultdict(dict)
    for p in all_png:
        name = os.path.splitext(os.path.basename(p))[0]
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in CHANNEL_ORDER:
            sample_files[parts[0]][parts[1]] = p

    file_list = []
    skipped = 0
    for basename, fd in sample_files.items():
        if not all(ch in fd for ch in CHANNEL_ORDER):
            skipped += 1
            continue
        label = parse_ravdess_label(f"{basename}_mel.png")
        if label is None:
            skipped += 1
            continue
        file_list.append((
            fd['mel'], fd['mfcc'], fd['cqt'], fd['chroma'], label
        ))

    if skipped:
        print(f"  Dilewati: {skipped} sampel (channel tidak lengkap)")
    print(f"  Total sampel valid: {len(file_list)}")
    return file_list


def split_file_list(file_list):
    """Stratified split 70/15/15 pada file_list (list of tuples)."""
    labels = [fl[4] for fl in file_list]
    idx = list(range(len(file_list)))

    idx_temp, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    labels_temp = [labels[i] for i in idx_temp]
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=val_ratio,
        random_state=RANDOM_SEED, stratify=labels_temp
    )
    train_list = [file_list[i] for i in idx_train]
    val_list   = [file_list[i] for i in idx_val]
    test_list  = [file_list[i] for i in idx_test]
    print(f"  Split: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")
    return train_list, val_list, test_list


# ===========================================================================
# tf.data pipeline -- baca PNG on-the-fly, proses di GPU
# ===========================================================================

def load_and_fuse(mel_p, mfcc_p, cqt_p, chroma_p, label,
                  use_4ch=True):
    """
    Load 4 PNG dari disk, fuse jadi tensor (H, W, 4).
    Kalau use_4ch=False, hanya pakai mel dan replikasi 4x.
    Ini berjalan di tf.data pipeline (lazy, per-sample).
    """
    H, W = IMG_SIZE

    def read_one(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_png(raw, channels=1)   # (H, W, 1) uint8
        img = tf.image.resize(img, [H, W])            # resize ke 224x224
        img = tf.cast(img, tf.float32) / 255.0        # normalize [0,1]
        return img  # (H, W, 1)

    mel   = read_one(mel_p)
    if use_4ch:
        mfcc  = read_one(mfcc_p)
        cqt   = read_one(cqt_p)
        chroma = read_one(chroma_p)
        x = tf.concat([mel, mfcc, cqt, chroma], axis=-1)  # (H, W, 4)
    else:
        x = tf.repeat(mel, 4, axis=-1)   # (H, W, 4) semua mel

    y = tf.one_hot(label, N_CLASSES)
    return x, y


def spec_augment_tf(x):
    """SpecAugment: freq + time masking. Berjalan di tf.data map."""
    H = tf.shape(x)[0]
    W = tf.shape(x)[1]
    C = tf.shape(x)[2]

    for _ in range(SPEC_AUG_NUM_FREQ_MASKS):
        f  = tf.random.uniform((), 0, SPEC_AUG_FREQ_MASK_MAX, dtype=tf.int32)
        f0 = tf.random.uniform((), 0, tf.maximum(H - f, 1), dtype=tf.int32)
        top    = tf.ones([f0, W, C], dtype=x.dtype)
        mid    = tf.zeros([f,  W, C], dtype=x.dtype)
        bot    = tf.ones([tf.maximum(H - f0 - f, 0), W, C], dtype=x.dtype)
        mask   = tf.concat([top, mid, bot], axis=0)[:H]
        x = x * mask

    for _ in range(SPEC_AUG_NUM_TIME_MASKS):
        t  = tf.random.uniform((), 0, SPEC_AUG_TIME_MASK_MAX, dtype=tf.int32)
        t0 = tf.random.uniform((), 0, tf.maximum(W - t, 1), dtype=tf.int32)
        lft = tf.ones([H, t0, C], dtype=x.dtype)
        mid = tf.zeros([H, t,  C], dtype=x.dtype)
        rgt = tf.ones([H, tf.maximum(W - t0 - t, 0), C], dtype=x.dtype)
        mask = tf.concat([lft, mid, rgt], axis=1)[:, :W, :]
        x = x * mask

    return x


def augment_tf(x, y):
    """Flip + zoom + SpecAugment. Dipakai di training pipeline."""
    H, W = IMG_SIZE
    # Horizontal flip
    x = tf.image.random_flip_left_right(x)
    # Random zoom 85-100%
    scale  = tf.random.uniform((), 0.85, 1.0)
    new_H  = tf.cast(tf.cast(H, tf.float32) * scale, tf.int32)
    new_W  = tf.cast(tf.cast(W, tf.float32) * scale, tf.int32)
    x = tf.image.random_crop(x, [new_H, new_W, 4])
    x = tf.image.resize(x, [H, W])
    # SpecAugment
    x = spec_augment_tf(x)
    return x, y


def make_dataset(file_list, use_4ch=True, augment=False, shuffle=True):
    """
    Buat tf.data.Dataset dari file_list.
    Data dibaca dari disk on-the-fly -- tidak ada array numpy besar di RAM.
    """
    # Pisahkan paths dan labels menjadi 5 kolom tensor
    mel_paths   = [fl[0] for fl in file_list]
    mfcc_paths  = [fl[1] for fl in file_list]
    cqt_paths   = [fl[2] for fl in file_list]
    chroma_paths = [fl[3] for fl in file_list]
    labels      = [fl[4] for fl in file_list]

    ds = tf.data.Dataset.from_tensor_slices(
        (mel_paths, mfcc_paths, cqt_paths, chroma_paths, labels)
    )

    # Map: baca PNG dan fuse channels
    _use_4ch = use_4ch
    ds = ds.map(
        lambda m, mf, c, ch, l: load_and_fuse(m, mf, c, ch, l, _use_4ch),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:
        ds = ds.map(augment_tf, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(512, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ===========================================================================
# SE Block
# ===========================================================================

def se_block(x, ratio=SE_RATIO):
    """Squeeze-and-Excitation channel attention. Hu et al., CVPR 2018."""
    C = x.shape[-1]
    sq = tf.keras.layers.GlobalAveragePooling2D()(x)
    ex = tf.keras.layers.Dense(
        max(C // ratio, 1), activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(sq)
    ex = tf.keras.layers.Dense(
        C, activation='sigmoid',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(ex)
    ex = tf.keras.layers.Reshape((1, 1, C))(ex)
    return tf.keras.layers.Multiply()([x, ex])


# ===========================================================================
# Bangun model Opsi C
# ===========================================================================

def build_model_opsi_c(input_shape=(224, 224, 4), n_classes=8):
    """
    ResNet50V2 + SE Block + BiGRU.
    Output head pakai float32 meskipun policy mixed_float16
    (syarat keras untuk numerik stability di softmax).
    """
    inputs = tf.keras.Input(shape=input_shape, name='input_4ch')

    # Channel projection: 4ch -> 3ch (agar sesuai pretrained ResNet50V2)
    x = tf.keras.layers.Conv2D(
        3, (1, 1), padding='same', use_bias=False, name='channel_proj',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='proj_bn')(x)
    x = tf.keras.layers.Activation('relu', name='proj_relu')(x)

    # Backbone ResNet50V2
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet',
        input_shape=(224, 224, 3), pooling=None
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    # x: (batch, 7, 7, 2048)

    # SE Block
    x = se_block(x, ratio=SE_RATIO)

    # BiGRU temporal modeling
    x = tf.keras.layers.Reshape((7 * 7, 2048), name='reshape_to_seq')(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            64, return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            recurrent_regularizer=tf.keras.regularizers.l2(L2_REG / 10),
        ),
        name='bigru'
    )(x)
    # x: (batch, 128)

    # Head
    x = tf.keras.layers.Dense(
        128, kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name='head_dense'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.Activation('relu', name='head_relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name='head_dropout')(x)

    # Cast ke float32 sebelum softmax (wajib untuk mixed precision stability)
    x = tf.keras.layers.Lambda(
        lambda t: tf.cast(t, tf.float32), name='cast_fp32'
    )(x)
    outputs = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='predictions',
        dtype='float32'
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SER_OpsiC')
    return model, base_model


# ===========================================================================
# Training satu kondisi ablasi
# ===========================================================================

def train_one_condition(train_list, val_list, test_list,
                         condition_name, use_4ch=True, use_aug=True):
    """
    Latih model untuk satu kondisi ablasi (B1-B4).
    Semua data dibaca on-the-fly dari disk -- tidak load ke RAM.
    """
    print(f"\n{'='*60}")
    print(f"Kondisi: {condition_name}")
    print(f"  4-channel: {use_4ch}  |  Augmentasi: {use_aug}")
    print(f"  Train: {len(train_list)}  Val: {len(val_list)}  Test: {len(test_list)}")
    print(f"{'='*60}")

    tf.keras.backend.clear_session()

    # --- Bangun tf.data pipeline (on-the-fly) ---
    train_ds = make_dataset(train_list, use_4ch=use_4ch,
                             augment=use_aug, shuffle=True)
    val_ds   = make_dataset(val_list,   use_4ch=use_4ch,
                             augment=False, shuffle=False)
    test_ds  = make_dataset(test_list,  use_4ch=use_4ch,
                             augment=False, shuffle=False)

    # --- Bangun model ---
    model, base_model = build_model_opsi_c(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 4),
        n_classes=N_CLASSES
    )
    model.summary(line_length=80, print_fn=lambda s: None)  # silent summary

    # ===========================================================
    # Phase 1: Latih head (backbone frozen)
    # ===========================================================
    print(f"\n--- Phase 1: Backbone frozen ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=['accuracy']
    )

    ckpt_path = os.path.join(MODEL_DIR, f"best_{condition_name}_opsiC.keras")
    cb_p1 = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_accuracy',
            save_best_only=True, verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=10,
            min_lr=1e-6, verbose=1
        )
    ]

    t0 = time.time()
    hist1 = model.fit(
        train_ds, epochs=EPOCHS_PHASE1,
        validation_data=val_ds,
        callbacks=cb_p1, verbose=1
    )

    # ===========================================================
    # Phase 2: Partial unfreeze + fine-tune
    # ===========================================================
    print(f"\n--- Phase 2: Partial unfreeze ({UNFREEZE_LAST_N} layer terakhir) ---")

    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-UNFREEZE_LAST_N:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Trainable: {unfrozen}/{len(base_model.layers)} layer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=['accuracy']
    )

    cb_p2 = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_accuracy',
            save_best_only=True, verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=10,
            min_lr=1e-7, verbose=1
        )
    ]

    hist2 = model.fit(
        train_ds, epochs=EPOCHS_PHASE2,
        validation_data=val_ds,
        callbacks=cb_p2, verbose=1
    )
    training_time = time.time() - t0

    # --- Evaluasi test set ---
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred  = np.argmax(y_pred_probs, axis=1)
    y_true  = np.array([fl[4] for fl in test_list])

    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix
    )
    test_acc = accuracy_score(y_true, y_pred)
    test_f1  = f1_score(y_true, y_pred, average='weighted')
    report   = classification_report(
        y_true, y_pred, target_names=EMOTION_LABELS, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    best_val_acc = max(
        max(hist1.history.get('val_accuracy', [0])),
        max(hist2.history.get('val_accuracy', [0]))
    )

    print(f"\n  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Test F1-Score: {test_f1*100:.2f}%")
    print(f"  Best Val Acc : {best_val_acc*100:.2f}%")
    print(f"  Waktu latih  : {training_time:.0f}s")

    return {
        'condition':          condition_name,
        'architecture':       'ResNet50V2+SE+BiGRU (mixed_float16, on-the-fly)',
        'use_4ch':            use_4ch,
        'use_aug':            use_aug,
        'test_accuracy':      round(float(test_acc), 4),
        'test_f1_weighted':   round(float(test_f1), 4),
        'best_val_accuracy':  round(float(best_val_acc), 4),
        'training_time_s':    round(training_time, 1),
        'per_class':          report,
        'confusion_matrix':   cm,
        'phase1_history': {
            'accuracy':     [round(v, 4) for v in hist1.history.get('accuracy', [])],
            'val_accuracy': [round(v, 4) for v in hist1.history.get('val_accuracy', [])],
        },
        'phase2_history': {
            'accuracy':     [round(v, 4) for v in hist2.history.get('accuracy', [])],
            'val_accuracy': [round(v, 4) for v in hist2.history.get('val_accuracy', [])],
        },
    }


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("STEP 5: Training SER Opsi C (RAM-efficient)")
    print("    Backbone : ResNet50V2 (pretrained ImageNet)")
    print("    Input    : 224x224x4")
    print("    Precision: mixed_float16")
    print("    Data     : on-the-fly PNG loading (tidak load npz ke RAM)")
    print(f"    Batch    : {BATCH_SIZE}")
    print("=" * 60)

    # --- Step 1: Copy spektrogram ke local SSD ---
    # Google Drive sangat lambat dibaca berulang kali per epoch.
    # Copy sekali ke /tmp/ (SSD Colab), lalu baca dari sana.
    print("\n[1/4] Copy spektrogram ke local SSD...")
    local_nodip = copy_to_local(SPEC_DIR_DRIVE,     LOCAL_SPEC_DIR,
                                 "tanpa DIP (B1/B3)")
    local_dip   = copy_to_local(ENHANCED_DIR_DRIVE, LOCAL_ENHANCED_DIR,
                                 "dengan DIP (B2/B4)")

    # --- Step 2: Bangun file list (hanya path strings, NO numpy array) ---
    print("\n[2/4] Bangun file list...")
    print("  Tanpa DIP:")
    fl_nodip = build_file_list(local_nodip)
    print("  Dengan DIP:")
    fl_dip   = build_file_list(local_dip)

    # --- Step 3: Split ---
    print("\n[3/4] Split dataset...")
    print("  Tanpa DIP:")
    tr_nd, val_nd, te_nd = split_file_list(fl_nodip)
    print("  Dengan DIP:")
    tr_dip, val_dip, te_dip = split_file_list(fl_dip)

    # --- Step 4: Latih 4 kondisi ablasi ---
    print("\n[4/4] Training 4 kondisi ablasi...")
    # B1: Mel only, tanpa DIP, tanpa augmentasi
    # B2: Mel only, dengan DIP, dengan augmentasi
    # B3: 4ch,      tanpa DIP, tanpa augmentasi
    # B4: 4ch,      dengan DIP, dengan augmentasi (proposed method)
    conditions = [
        ('B1', tr_nd,  val_nd,  te_nd,  False, False),
        ('B2', tr_dip, val_dip, te_dip, False, True),
        ('B3', tr_nd,  val_nd,  te_nd,  True,  False),
        ('B4', tr_dip, val_dip, te_dip, True,  True),
    ]

    results = []
    summary_path = os.path.join(RESULT_DIR, "ablation_summary.json")

    for name, tr, vl, te, use_4ch, use_aug in conditions:
        result = train_one_condition(
            tr, vl, te,
            condition_name=name,
            use_4ch=use_4ch,
            use_aug=use_aug
        )
        results.append(result)
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Progress disimpan -> {summary_path}")

    # --- Ringkasan ---
    print("\n" + "=" * 60)
    print("RINGKASAN ABLASI -- Opsi C (ResNet50V2 + SE + BiGRU)")
    print("=" * 60)
    print(f"{'Kondisi':<10} {'Test Acc':>10} {'Test F1':>10} {'Val Acc':>10}")
    print("-" * 44)
    for r in results:
        print(f"{r['condition']:<10} "
              f"{r['test_accuracy']*100:>9.2f}% "
              f"{r['test_f1_weighted']*100:>9.2f}% "
              f"{r['best_val_accuracy']*100:>9.2f}%")

    print(f"\nHasil lengkap -> {summary_path}")
    print("Step 5 selesai.")
