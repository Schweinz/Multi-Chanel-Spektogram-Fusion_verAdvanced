"""
step5_train.py  —  Opsi C: ResNet50V2 + SE Block + BiGRU + SpecAugment
=======================================================================
Arsitektur baru (Opsi C):
  Input  : 224x224x4  (4-channel fused spectrogram, ukuran ImageNet)
  Proj   : Conv2D(3, 1x1)        -> ubah 4ch -> 3ch agar ResNet50V2 bisa dipakai
  Backbone: ResNet50V2 pretrained ImageNet, feature map 7x7x2048
  SE     : Squeeze-and-Excitation block (channel attention, ratio=16)
  BiGRU  : reshape feature map ke sekuens (49, 2048), lalu Bidirectional GRU(64)
           -> tangkap pola temporal emosi di dimensi waktu spektrogram
  Head   : Dense(128) + BN + Dropout -> Dense(8, softmax)

Training 2 fase:
  Phase 1 (frozen backbone): 30 epoch, LR=5e-4
  Phase 2 (partial unfreeze last 100 layer): 100 epoch, LR=5e-5

Regularisasi anti-overfitting:
  - Partial unfreeze (bukan full fine-tune)
  - Dropout 0.5
  - L2 regularization 1e-4
  - Label smoothing 0.1
  - SpecAugment (freq + time masking)
  - Flip horizontal + slight zoom

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
import numpy as np

# ---------------------------------------------------------------------------
# Self-patching config (agar %run di Colab selalu pakai path yang benar)
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
_cfg.IMG_SIZE     = (224, 224)   # Opsi C
_cfg.create_dirs()

FUSED_DIR      = _cfg.FUSED_DIR
MODEL_DIR      = _cfg.MODEL_DIR
RESULT_DIR     = _cfg.RESULT_DIR
N_CLASSES      = _cfg.N_CLASSES
EMOTION_LABELS = _cfg.EMOTION_LABELS
IMG_SIZE       = _cfg.IMG_SIZE   # (224, 224)

SE_RATIO               = _cfg.SE_RATIO              # 16
SPEC_AUG_FREQ_MASK_MAX = _cfg.SPEC_AUG_FREQ_MASK_MAX   # 20
SPEC_AUG_TIME_MASK_MAX = _cfg.SPEC_AUG_TIME_MASK_MAX   # 25
SPEC_AUG_NUM_FREQ_MASKS = _cfg.SPEC_AUG_NUM_FREQ_MASKS  # 2
SPEC_AUG_NUM_TIME_MASKS = _cfg.SPEC_AUG_NUM_TIME_MASKS  # 2

# ---------------------------------------------------------------------------
# Hyperparameter training (override config untuk Opsi C)
# ---------------------------------------------------------------------------
BATCH_SIZE      = 16        # lebih kecil krn 224x224 lebih berat di memori GPU
LR_PHASE1       = 5e-4      # Phase 1: hanya head + proj + SE + BiGRU
LR_PHASE2       = 5e-5      # Phase 2: partial unfreeze, LR lebih kecil
DROPOUT_RATE    = 0.5       # dropout agresif untuk RAVDESS yg kecil (~1000 train)
L2_REG          = 1e-4      # L2 regularization
LABEL_SMOOTHING = 0.1       # mencegah model terlalu yakin pada label
EPOCHS_PHASE1   = 30
EPOCHS_PHASE2   = 100
PATIENCE        = 20        # early stopping patience

# Jumlah layer ResNet50V2 yang di-unfreeze di Phase 2
# ResNet50V2 punya ~190 layer; unfreeze 100 layer terakhir
UNFREEZE_LAST_N = 100

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")


# ===========================================================================
# SpecAugment  (augmentasi pada domain spektrogram)
# ===========================================================================

def spec_augment(x, training=False):
    """
    SpecAugment: masking acak frekuensi + waktu.
    Park et al., Interspeech 2019.

    x: tensor (H, W, C) float32
    Hanya dijalankan saat training=True.
    """
    if not training:
        return x

    H = tf.shape(x)[0]
    W = tf.shape(x)[1]
    C = tf.shape(x)[2]

    # Frequency masking (masker pada dimensi frekuensi = sumbu vertikal H)
    for _ in range(SPEC_AUG_NUM_FREQ_MASKS):
        f = tf.random.uniform((), 0, SPEC_AUG_FREQ_MASK_MAX, dtype=tf.int32)
        f0 = tf.random.uniform((), 0, tf.maximum(H - f, 1), dtype=tf.int32)
        # Buat mask: set baris f0..f0+f ke 0
        mask_top    = tf.ones([f0, W, C], dtype=x.dtype)
        mask_middle = tf.zeros([f, W, C], dtype=x.dtype)
        mask_bottom = tf.ones([tf.maximum(H - f0 - f, 0), W, C], dtype=x.dtype)
        mask = tf.concat([mask_top, mask_middle, mask_bottom], axis=0)
        mask = mask[:H]
        x = x * mask

    # Time masking (masker pada dimensi waktu = sumbu horizontal W)
    for _ in range(SPEC_AUG_NUM_TIME_MASKS):
        t = tf.random.uniform((), 0, SPEC_AUG_TIME_MASK_MAX, dtype=tf.int32)
        t0 = tf.random.uniform((), 0, tf.maximum(W - t, 1), dtype=tf.int32)
        mask_left   = tf.ones([H, t0, C], dtype=x.dtype)
        mask_middle = tf.zeros([H, t, C], dtype=x.dtype)
        mask_right  = tf.ones([H, tf.maximum(W - t0 - t, 0), C], dtype=x.dtype)
        mask = tf.concat([mask_left, mask_middle, mask_right], axis=1)
        mask = mask[:, :W, :]
        x = x * mask

    return x


# ===========================================================================
# SE Block  (Squeeze-and-Excitation)
# ===========================================================================

def se_block(x, ratio=SE_RATIO):
    """
    SE Block: channel attention.
    Hu et al., CVPR 2018.

    x: tensor (batch, H, W, C) dari output backbone
    Return: tensor (batch, H, W, C) setelah channel recalibration
    """
    C = x.shape[-1]
    # Squeeze: global average pooling -> (batch, C)
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Excitation: 2 Dense layers dengan bottleneck C/ratio
    excitation = tf.keras.layers.Dense(
        max(C // ratio, 1), activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(squeeze)
    excitation = tf.keras.layers.Dense(
        C, activation='sigmoid',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(excitation)
    # Reshape ke (batch, 1, 1, C) untuk broadcasting
    excitation = tf.keras.layers.Reshape((1, 1, C))(excitation)
    # Scale: element-wise multiply
    return tf.keras.layers.Multiply()([x, excitation])


# ===========================================================================
# Membangun model Opsi C
# ===========================================================================

def build_model_opsi_c(input_shape=(224, 224, 4), n_classes=8):
    """
    Arsitektur lengkap Opsi C:
      Input (224,224,4)
      -> Conv2D projection (224,224,3)
      -> ResNet50V2 pretrained ImageNet (output 7,7,2048)
      -> SE Block channel attention
      -> Reshape ke (49, 2048)
      -> BiGRU(64) temporal modeling
      -> Dense head dengan Dropout + BN
      -> Softmax output
    """
    inputs = tf.keras.Input(shape=input_shape, name='input_4ch')

    # --- Channel Projection: 4ch -> 3ch ---
    # ResNet50V2 pretrained mengharapkan 3-channel input (RGB).
    # Proyeksi 1x1 Conv2D belajar kombinasi optimal dari 4 channel spektrogram.
    x = tf.keras.layers.Conv2D(
        3, (1, 1), padding='same', use_bias=False,
        name='channel_proj',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='proj_bn')(x)
    x = tf.keras.layers.Activation('relu', name='proj_relu')(x)

    # --- Backbone: ResNet50V2 pretrained ImageNet ---
    # include_top=False -> kita pakai feature map 7x7x2048
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling=None   # kita handle sendiri
    )
    base_model.trainable = False   # frozen di Phase 1
    x = base_model(x, training=False)
    # x sekarang: (batch, 7, 7, 2048)

    # --- SE Block: channel attention ---
    x = se_block(x, ratio=SE_RATIO)
    # x masih: (batch, 7, 7, 2048)

    # --- BiGRU: temporal modeling ---
    # Reshape feature map (7, 7, 2048) -> sekuens (49, 2048)
    # 49 = 7x7 lokasi spatial, tiap lokasi = vektor 2048 dimensi
    # BiGRU akan menangkap pola temporal antar lokasi spektrogram
    x = tf.keras.layers.Reshape((7 * 7, 2048), name='reshape_to_seq')(x)

    # Bidirectional GRU: forward + backward -> output size 2x64=128
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            64,
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            recurrent_regularizer=tf.keras.regularizers.l2(L2_REG / 10),
        ),
        name='bigru'
    )(x)
    # x: (batch, 128)

    # --- Head ---
    x = tf.keras.layers.Dense(
        128,
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name='head_dense'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.Activation('relu', name='head_relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name='head_dropout')(x)

    outputs = tf.keras.layers.Dense(
        n_classes,
        activation='softmax',
        name='predictions'
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SER_OpsiC')
    return model, base_model


# ===========================================================================
# Data augmentation (di luar SpecAugment)
# ===========================================================================

def augment_sample(x, y):
    """Flip horizontal + slight zoom + SpecAugment. Dipakai di tf.data pipeline."""
    # Horizontal flip (membalik dimensi waktu spektrogram)
    x = tf.image.random_flip_left_right(x)
    # Random zoom: crop acak 85-100% lalu resize kembali
    H, W, C = IMG_SIZE[0], IMG_SIZE[1], 4
    scale = tf.random.uniform((), 0.85, 1.0)
    new_H = tf.cast(tf.cast(H, tf.float32) * scale, tf.int32)
    new_W = tf.cast(tf.cast(W, tf.float32) * scale, tf.int32)
    x = tf.image.random_crop(x, [new_H, new_W, C])
    x = tf.image.resize(x, [H, W])
    # SpecAugment
    x = spec_augment(x, training=True)
    return x, y


def augment_sample_mel(x, y):
    """Augmentasi untuk kondisi Mel-only: pakai channel 0 saja di semua posisi."""
    mel = x[:, :, 0:1]
    x_mel = tf.repeat(mel, 4, axis=-1)
    x_mel = tf.image.random_flip_left_right(x_mel)
    H, W, C = IMG_SIZE[0], IMG_SIZE[1], 4
    scale = tf.random.uniform((), 0.85, 1.0)
    new_H = tf.cast(tf.cast(H, tf.float32) * scale, tf.int32)
    new_W = tf.cast(tf.cast(W, tf.float32) * scale, tf.int32)
    x_mel = tf.image.random_crop(x_mel, [new_H, new_W, C])
    x_mel = tf.image.resize(x_mel, [H, W])
    x_mel = spec_augment(x_mel, training=True)
    return x_mel, y


# ===========================================================================
# Training satu kondisi ablasi
# ===========================================================================

def train_one_condition(X_train, y_train, X_val, y_val, X_test, y_test,
                         condition_name, use_4ch=True, use_aug=True):
    """
    Latih model untuk satu kondisi ablasi (B1-B4).

    condition_name: 'B1', 'B2', 'B3', atau 'B4'
    use_4ch : True  -> gunakan semua 4 channel
              False -> hanya Mel (channel 0, direplikasi 4x)
    use_aug : True  -> pakai SpecAugment + flip + zoom
              False -> tanpa augmentasi
    """
    print(f"\n{'='*60}")
    print(f"Kondisi: {condition_name}")
    print(f"  4-channel: {use_4ch}  |  Augmentasi: {use_aug}")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"{'='*60}")

    tf.keras.backend.clear_session()

    # --- Siapkan data ---
    if not use_4ch:
        # Hanya Mel: ambil channel 0, replikasi ke 4 channel
        mel_train = np.repeat(X_train[:, :, :, 0:1], 4, axis=-1)
        mel_val   = np.repeat(X_val[:, :, :, 0:1],   4, axis=-1)
        mel_test  = np.repeat(X_test[:, :, :, 0:1],  4, axis=-1)
    else:
        mel_train = X_train
        mel_val   = X_val
        mel_test  = X_test

    # One-hot encoding
    y_train_oh = tf.keras.utils.to_categorical(y_train, N_CLASSES)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   N_CLASSES)

    # --- tf.data pipeline ---
    train_ds = tf.data.Dataset.from_tensor_slices((mel_train, y_train_oh))
    if use_aug:
        if not use_4ch:
            train_ds = train_ds.map(augment_sample_mel,
                                    num_parallel_calls=tf.data.AUTOTUNE)
        else:
            train_ds = train_ds.map(augment_sample,
                                    num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = (train_ds
                .shuffle(1024, reshuffle_each_iteration=True)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset
              .from_tensor_slices((mel_val, y_val_oh))
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))

    # --- Bangun model ---
    model, base_model = build_model_opsi_c(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 4),
        n_classes=N_CLASSES
    )

    # ===========================================================
    # Phase 1: Latih head (backbone frozen)
    # ===========================================================
    print(f"\n--- Phase 1: Backbone frozen, latih head ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=['accuracy']
    )

    ckpt_path = os.path.join(MODEL_DIR, f"best_{condition_name}_opsiC.h5")
    callbacks_p1 = [
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
        callbacks=callbacks_p1,
        verbose=1
    )

    # ===========================================================
    # Phase 2: Partial unfreeze + fine-tune
    # ===========================================================
    print(f"\n--- Phase 2: Partial unfreeze ({UNFREEZE_LAST_N} layer terakhir) ---")

    # Freeze semua dulu, kemudian unfreeze N layer terakhir
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-UNFREEZE_LAST_N:]:
        # Jangan unfreeze BatchNormalization (menjaga statistik pretrained)
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Layer trainable: {unfrozen}/{len(base_model.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=['accuracy']
    )

    callbacks_p2 = [
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
        callbacks=callbacks_p2,
        verbose=1
    )
    training_time = time.time() - t0

    # --- Evaluasi test set ---
    # Muat checkpoint terbaik (val_accuracy terbaik)
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    test_ds = (tf.data.Dataset
               .from_tensor_slices((mel_test, y_test))
               .batch(BATCH_SIZE))
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Hitung metrik
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix
    )
    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average='weighted')
    report   = classification_report(
        y_test, y_pred,
        target_names=EMOTION_LABELS,
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    best_val_acc = max(
        max(hist1.history.get('val_accuracy', [0])),
        max(hist2.history.get('val_accuracy', [0]))
    )

    print(f"\n  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"  Best Val Acc : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Waktu latih  : {training_time:.0f}s")

    return {
        'condition':          condition_name,
        'architecture':       'ResNet50V2+SE+BiGRU',
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
# Main: jalankan semua ablasi
# ===========================================================================

if __name__ == "__main__":
    print("STEP 5: Training SER Opsi C")
    print("    Backbone : ResNet50V2 (pretrained ImageNet)")
    print("    Input    : 224x224x4")
    print("    Attention: SE Block (channel, ratio=16)")
    print("    Temporal : BiGRU(64) bidirectional")
    print("    Augmen.  : SpecAugment + flip + zoom")
    print("=" * 60)

    # --- Muat dataset ---
    dataset_path = os.path.join(FUSED_DIR, "dataset.npz")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {dataset_path}\n"
            "Pastikan step2 -> step3 -> step4 sudah dijalankan dengan IMG_SIZE=224."
        )

    data = np.load(dataset_path)
    X_train = data['X_train'].astype(np.float32)
    y_train = data['y_train'].astype(np.int32)
    X_val   = data['X_val'].astype(np.float32)
    y_val   = data['y_val'].astype(np.int32)
    X_test  = data['X_test'].astype(np.float32)
    y_test  = data['y_test'].astype(np.int32)

    # Validasi ukuran input sesuai Opsi C (224x224)
    actual_size = X_train.shape[1:3]
    expected = (IMG_SIZE[0], IMG_SIZE[1])
    if actual_size != expected:
        raise ValueError(
            f"Dataset size {actual_size} != {expected}.\n"
            "Hapus folder spectrograms/ dan fused_dataset/ lalu jalankan "
            "step2 -> step3 -> step4 ulang dengan IMG_SIZE=(224,224)."
        )

    print(f"\nDataset dimuat:")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}")
    print(f"  X_test : {X_test.shape}")

    # --- Cek dataset tanpa DIP untuk ablasi B1/B3 ---
    dataset_nodip_path = os.path.join(FUSED_DIR, "dataset_nodip.npz")
    if os.path.exists(dataset_nodip_path):
        data_nodip = np.load(dataset_nodip_path)
        X_train_nd = data_nodip['X_train'].astype(np.float32)
        y_train_nd = data_nodip['y_train'].astype(np.int32)
        X_val_nd   = data_nodip['X_val'].astype(np.float32)
        y_val_nd   = data_nodip['y_val'].astype(np.int32)
        X_test_nd  = data_nodip['X_test'].astype(np.float32)
        y_test_nd  = data_nodip['y_test'].astype(np.int32)
        print(f"\nDataset tanpa DIP ditemukan: {dataset_nodip_path}")
    else:
        print(f"\n[INFO] dataset_nodip.npz tidak ada -> B1/B3 pakai dataset.npz (DIP ON)")
        print("       Untuk ablasi ketat B1/B3, jalankan step4 dengan SPEC_DIR (bukan ENHANCED_DIR).")
        X_train_nd, y_train_nd = X_train, y_train
        X_val_nd,   y_val_nd   = X_val,   y_val
        X_test_nd,  y_test_nd  = X_test,  y_test

    # --- Jalankan 4 kondisi ablasi ---
    # B1: Mel only, tanpa DIP, tanpa augmentasi
    # B2: Mel only, dengan DIP, dengan augmentasi
    # B3: 4ch, tanpa DIP, tanpa augmentasi
    # B4: 4ch, dengan DIP, dengan augmentasi (kondisi utama / proposed method)
    conditions = [
        # (name, X_tr, y_tr, X_v, y_v, X_te, y_te, use_4ch, use_aug)
        ('B1', X_train_nd, y_train_nd, X_val_nd, y_val_nd, X_test_nd, y_test_nd, False, False),
        ('B2', X_train,    y_train,    X_val,    y_val,    X_test,    y_test,    False, True),
        ('B3', X_train_nd, y_train_nd, X_val_nd, y_val_nd, X_test_nd, y_test_nd, True,  False),
        ('B4', X_train,    y_train,    X_val,    y_val,    X_test,    y_test,    True,  True),
    ]

    results = []
    summary_path = os.path.join(RESULT_DIR, "ablation_summary.json")

    for cond in conditions:
        name, Xtr, ytr, Xv, yv, Xte, yte, use_4ch, use_aug = cond
        result = train_one_condition(
            Xtr, ytr, Xv, yv, Xte, yte,
            condition_name=name,
            use_4ch=use_4ch,
            use_aug=use_aug
        )
        results.append(result)

        # Simpan progress (agar tidak hilang jika Colab crash)
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Progress disimpan -> {summary_path}")

    # --- Ringkasan akhir ---
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
    print("Step 5 selesai. Lanjut ke step6_evaluation.py untuk visualisasi.")
