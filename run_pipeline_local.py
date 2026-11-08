"""
run_pipeline_local.py
=====================
Runner untuk menjalankan seluruh pipeline di environment lokal
(tanpa Google Colab / Drive).

Langkah:
  1. Install library yang dibutuhkan
  2. Ekstraksi 4 spektrogram dari 1440 file .wav
  3. DIP enhancement (CLAHE + Unsharp Masking)
  4. Build dataset 4-channel → dataset.npz
  5. Training ablation B1-B4
  6. Evaluasi dan Grad-CAM

Semua data RAVDESS asli dari /home/claude/ravdess_data/
"""

import os
import sys
import subprocess

# Tambah path supaya config_local bisa ditemukan
sys.path.insert(0, '/home/claude/multi-chanel-spektogram-fusion-advanced')

# Gunakan config lokal — override sys.modules supaya import 'config' di step lain
# menggunakan config_local
import config_local as cfg
sys.modules['config'] = cfg   # intercept semua "from config import ..."

cfg.create_dirs()
print("Direktori output:")
for attr in ['SPEC_DIR', 'ENHANCED_DIR', 'FUSED_DIR', 'MODEL_DIR', 'RESULT_DIR']:
    print(f"  {attr}: {getattr(cfg, attr)}")
print(f"  RAW_DATA_DIR: {cfg.RAW_DATA_DIR}")

# Verifikasi data ada
import glob
wav_files = glob.glob(f"{cfg.RAW_DATA_DIR}/**/*.wav", recursive=True)
print(f"\nFile .wav ditemukan: {len(wav_files)}")
if len(wav_files) == 0:
    print("ERROR: Tidak ada file .wav!")
    sys.exit(1)

# =============================================================================
# STEP 2: Ekstraksi Spektrogram
# =============================================================================
print("\n" + "="*60)
print("STEP 2: Ekstraksi Spektrogram (1440 file → 5760 PNG)")
print("="*60)

import numpy as np
import librosa
import cv2
from tqdm import tqdm

def load_audio(filepath, target_duration=3.0, sr=22050):
    target_samples = int(sr * target_duration)
    y, _ = librosa.load(filepath, sr=sr)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='reflect')
    elif len(y) > target_samples:
        start = (len(y) - target_samples) // 2
        y = y[start: start + target_samples]
    return y

def spectrogram_to_image(spec_data, size=(128,128)):
    mn, mx = spec_data.min(), spec_data.max()
    if mx - mn > 1e-8:
        norm = (spec_data - mn) / (mx - mn) * 255
    else:
        norm = np.zeros_like(spec_data)
    return cv2.resize(norm.astype(np.uint8), size, interpolation=cv2.INTER_AREA)

def process_wav(wav_path, out_dir, sr=22050):
    base = os.path.splitext(os.path.basename(wav_path))[0]
    types = ['mel','mfcc','cqt','chroma']
    if all(os.path.exists(os.path.join(out_dir, f"{base}_{t}.png")) for t in types):
        return True
    try:
        y = load_audio(wav_path, sr=sr)
        specs = {
            'mel':    librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512), ref=np.max),
            'mfcc':   librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, n_fft=2048, hop_length=512),
            'cqt':    librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, n_bins=128, bins_per_octave=24, hop_length=512)), ref=np.max),
            'chroma': librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512),
        }
        for t, S in specs.items():
            img = spectrogram_to_image(S)
            cv2.imwrite(os.path.join(out_dir, f"{base}_{t}.png"), img)
        return True
    except Exception as e:
        print(f"\nError {os.path.basename(wav_path)}: {e}")
        return False

existing_png = glob.glob(f"{cfg.SPEC_DIR}/*.png")
if len(existing_png) >= 5750:
    print(f"Spektrogram sudah ada ({len(existing_png)} file), skip step 2.")
else:
    print(f"Memproses {len(wav_files)} file .wav ...")
    ok = sum(process_wav(f, cfg.SPEC_DIR) for f in tqdm(wav_files, unit='file'))
    print(f"Step 2 selesai: {ok}/{len(wav_files)} berhasil")
    print(f"PNG dihasilkan: {len(glob.glob(cfg.SPEC_DIR+'/*.png'))}")

# =============================================================================
# STEP 3: DIP Enhancement
# =============================================================================
print("\n" + "="*60)
print("STEP 3: DIP Enhancement (CLAHE + Unsharp Masking)")
print("="*60)

def apply_clahe(img):
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return c.apply(img)

def apply_unsharp_masking(img, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img, 1+strength, blurred, -strength, 0)

def apply_dip(img):
    return apply_unsharp_masking(apply_clahe(img))

spec_pngs = glob.glob(f"{cfg.SPEC_DIR}/*.png")
existing_enh = glob.glob(f"{cfg.ENHANCED_DIR}/*.png")
if len(existing_enh) >= len(spec_pngs) - 10:
    print(f"Enhanced PNG sudah ada ({len(existing_enh)} file), skip step 3.")
else:
    print(f"Memproses {len(spec_pngs)} gambar ...")
    ok = 0
    for path in tqdm(spec_pngs, unit='file'):
        out = os.path.join(cfg.ENHANCED_DIR, os.path.basename(path))
        if os.path.exists(out):
            ok += 1
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            cv2.imwrite(out, apply_dip(img))
            ok += 1
    print(f"Step 3 selesai: {ok}/{len(spec_pngs)} berhasil")

# =============================================================================
# STEP 4: Build Dataset
# =============================================================================
print("\n" + "="*60)
print("STEP 4: Fusion 4-Channel + Split Dataset")
print("="*60)

from collections import defaultdict
from sklearn.model_selection import train_test_split

out_npz = os.path.join(cfg.FUSED_DIR, "dataset.npz")

if os.path.exists(out_npz):
    print(f"dataset.npz sudah ada, skip step 4.")
    data = np.load(out_npz)
    X_train, y_train = data['X_train'], data['y_train']
    X_val,   y_val   = data['X_val'],   data['y_val']
    X_test,  y_test  = data['X_test'],  data['y_test']
    print(f"  train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
else:
    all_png = glob.glob(f"{cfg.ENHANCED_DIR}/*.png")
    sample_files = defaultdict(dict)
    for p in all_png:
        base = os.path.splitext(os.path.basename(p))[0]
        parts = base.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in cfg.CHANNEL_ORDER:
            sample_files[parts[0]][parts[1]] = p

    X_list, y_list = [], []
    for basename, fd in tqdm(sample_files.items(), desc="Fusing", unit="sampel"):
        if not all(ch in fd for ch in cfg.CHANNEL_ORDER):
            continue
        ravdess_base = basename.rsplit('_', 1)[0] if '_' not in basename else basename
        parts = os.path.basename(basename).split('-')
        if len(parts) < 3:
            continue
        emo_name = cfg.EMOTION_MAP.get(parts[2])
        if emo_name is None:
            continue
        label = cfg.EMOTION_LABELS.index(emo_name)

        channels = []
        for ch in cfg.CHANNEL_ORDER:
            img = cv2.imread(fd[ch], cv2.IMREAD_GRAYSCALE)
            if img is None:
                channels.append(np.zeros((128,128), dtype=np.float32))
            else:
                if img.shape[:2] != (128,128):
                    img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
                channels.append(img.astype(np.float32) / 255.0)
        X_list.append(np.stack(channels, axis=-1))
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Dataset: {len(X)} sampel, shape={X.shape}")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, random_state=42, stratify=y_tmp
    )
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    np.savez_compressed(out_npz,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    size_mb = os.path.getsize(out_npz)/1024/1024
    print(f"Tersimpan: {out_npz} ({size_mb:.1f} MB)")

# =============================================================================
# STEP 5: Training B4 (Proposed Method) — 4-Channel + DIP + SpecAugment
# =============================================================================
print("\n" + "="*60)
print("STEP 5: Training Model B4 (EfficientNetB0 + SE Block + SpecAugment)")
print("="*60)

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
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

print(f"TF version: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

def se_block(x, ratio=16):
    channels = x.shape[-1]
    squeeze = GlobalAveragePooling2D()(x)
    excitation = Dense(max(1, channels // ratio), activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, channels))(excitation)
    return Multiply()([x, excitation])

def build_model_4ch():
    inp = Input(shape=(128, 128, 4), name='input_4ch')
    x = Conv2D(3, (1,1), padding='same', name='channel_projection')(inp)
    base = EfficientNetB0(input_shape=(128,128,3), include_top=False, weights='imagenet')
    base.trainable = False
    x = base(x, training=False)
    x = se_block(x, ratio=16)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    out = Dense(8, activation='softmax', name='output')(x)
    model = Model(inp, out, name='SER_EfficientNetB0_SE_4ch')
    return model, base

def build_model_3ch():
    base = EfficientNetB0(input_shape=(128,128,3), include_top=False, weights='imagenet')
    base.trainable = False
    inp = Input(shape=(128, 128, 3), name='input_3ch')
    x = base(inp, training=False)
    x = se_block(x, ratio=16)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    out = Dense(8, activation='softmax', name='output')(x)
    model = Model(inp, out, name='SER_EfficientNetB0_SE_3ch')
    return model, base

def apply_specaugment(X, freq_max=20, time_max=25, n_freq=2, n_time=2):
    X_aug = X.copy()
    N, H, W, C = X_aug.shape
    for i in range(N):
        for _ in range(n_freq):
            f = np.random.randint(0, freq_max+1)
            f0 = np.random.randint(0, max(1, H-f))
            X_aug[i, f0:f0+f, :, :] = 0.0
        for _ in range(n_time):
            t = np.random.randint(0, time_max+1)
            t0 = np.random.randint(0, max(1, W-t))
            X_aug[i, :, t0:t0+t, :] = 0.0
    return X_aug

def train_2phase(model, base, X_tr, y_tr, X_vl, y_vl, name, use_aug=False):
    X_in = apply_specaugment(X_tr) if use_aug else X_tr
    y_tr_oh = to_categorical(y_tr, 8)
    y_vl_oh = to_categorical(y_vl, 8)

    # Phase 1 — frozen backbone
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    h1 = model.fit(
        X_in, y_tr_oh, validation_data=(X_vl, y_vl_oh),
        batch_size=32, epochs=10,
        callbacks=[
            CSVLogger(os.path.join(cfg.MODEL_DIR, f"{name}_ph1.csv")),
            ModelCheckpoint(os.path.join(cfg.MODEL_DIR, f"{name}_best.keras"),
                            monitor='val_accuracy', save_best_only=True, verbose=0),
        ], verbose=1
    )

    # Phase 2 — unfreeze semua
    base.trainable = True
    X_in2 = apply_specaugment(X_tr) if use_aug else X_tr
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    h2 = model.fit(
        X_in2, y_tr_oh, validation_data=(X_vl, y_vl_oh),
        batch_size=32, epochs=100,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(os.path.join(cfg.MODEL_DIR, f"{name}_best.keras"),
                            monitor='val_accuracy', save_best_only=True, verbose=0),
            CSVLogger(os.path.join(cfg.MODEL_DIR, f"{name}_ph2.csv")),
        ], verbose=1
    )
    return h1, h2

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns

ablation_results = {}

# Fungsi bantu untuk data 3-channel (Mel duplikat)
def to_3ch(X):
    mel = X[:, :, :, 0:1]
    return np.concatenate([mel, mel, mel], axis=-1)

# ─── B1: Single-Mel, No DIP, No Aug ─────────────────────────────────────────
print("\n--- B1: Single-Mel, No DIP, No SpecAugment ---")
# Load data tanpa DIP (dari SPEC_DIR)
b1_pngs = glob.glob(f"{cfg.SPEC_DIR}/*.png")
b1_samples = defaultdict(dict)
for p in b1_pngs:
    base = os.path.splitext(os.path.basename(p))[0]
    parts = base.rsplit('_',1)
    if len(parts)==2 and parts[1] in cfg.CHANNEL_ORDER:
        b1_samples[parts[0]][parts[1]] = p

X_raw, y_raw = [], []
for basename, fd in b1_samples.items():
    if not all(ch in fd for ch in cfg.CHANNEL_ORDER): continue
    parts2 = os.path.basename(basename).split('-')
    if len(parts2) < 3: continue
    emo_name = cfg.EMOTION_MAP.get(parts2[2])
    if emo_name is None: continue
    label = cfg.EMOTION_LABELS.index(emo_name)
    channels = []
    for ch in cfg.CHANNEL_ORDER:
        img = cv2.imread(fd[ch], cv2.IMREAD_GRAYSCALE)
        if img is None: channels.append(np.zeros((128,128), dtype=np.float32))
        else:
            if img.shape[:2]!=(128,128): img=cv2.resize(img,(128,128))
            channels.append(img.astype(np.float32)/255.0)
    X_raw.append(np.stack(channels, axis=-1))
    y_raw.append(label)

X_raw = np.array(X_raw, dtype=np.float32)
y_raw = np.array(y_raw, dtype=np.int32)
Xr_tmp, Xr_test, yr_tmp, yr_test = train_test_split(X_raw,y_raw,test_size=0.15,random_state=42,stratify=y_raw)
Xr_train,Xr_val,yr_train,yr_val = train_test_split(Xr_tmp,yr_tmp,test_size=0.15/0.85,random_state=42,stratify=yr_tmp)

b1_keras = os.path.join(cfg.MODEL_DIR, "B1_mel_nodip_noaug_best.keras")
if os.path.exists(b1_keras):
    print("Model B1 sudah ada, skip training.")
    from tensorflow.keras.models import load_model as lm
    model_b1 = lm(b1_keras)
else:
    model_b1, base_b1 = build_model_3ch()
    train_2phase(model_b1, base_b1, to_3ch(Xr_train), yr_train,
                 to_3ch(Xr_val), yr_val, 'B1_mel_nodip_noaug', use_aug=False)
    model_b1 = tf.keras.models.load_model(b1_keras)

y_pred_b1 = np.argmax(model_b1.predict(to_3ch(Xr_test), verbose=0), axis=1)
acc_b1 = accuracy_score(yr_test, y_pred_b1)
f1_b1  = f1_score(yr_test, y_pred_b1, average='weighted')
print(f"B1 — Accuracy: {acc_b1*100:.2f}%, F1: {f1_b1*100:.2f}%")
ablation_results['B1'] = {'accuracy': round(float(acc_b1),4), 'f1': round(float(f1_b1),4)}

# ─── B2: Single-Mel, DIP, SpecAugment ────────────────────────────────────────
print("\n--- B2: Single-Mel, DIP, SpecAugment ---")
b2_keras = os.path.join(cfg.MODEL_DIR, "B2_mel_dip_aug_best.keras")
if os.path.exists(b2_keras):
    print("Model B2 sudah ada, skip training.")
    model_b2 = tf.keras.models.load_model(b2_keras)
else:
    model_b2, base_b2 = build_model_3ch()
    train_2phase(model_b2, base_b2, to_3ch(X_train), y_train,
                 to_3ch(X_val), y_val, 'B2_mel_dip_aug', use_aug=True)
    model_b2 = tf.keras.models.load_model(b2_keras)

y_pred_b2 = np.argmax(model_b2.predict(to_3ch(X_test), verbose=0), axis=1)
acc_b2 = accuracy_score(y_test, y_pred_b2)
f1_b2  = f1_score(y_test, y_pred_b2, average='weighted')
print(f"B2 — Accuracy: {acc_b2*100:.2f}%, F1: {f1_b2*100:.2f}%")
ablation_results['B2'] = {'accuracy': round(float(acc_b2),4), 'f1': round(float(f1_b2),4)}

# ─── B3: 4-Channel, No DIP, No Aug ───────────────────────────────────────────
print("\n--- B3: 4-Channel, No DIP, No SpecAugment ---")
b3_keras = os.path.join(cfg.MODEL_DIR, "B3_4ch_nodip_noaug_best.keras")
if os.path.exists(b3_keras):
    print("Model B3 sudah ada, skip training.")
    model_b3 = tf.keras.models.load_model(b3_keras)
else:
    model_b3, base_b3 = build_model_4ch()
    train_2phase(model_b3, base_b3, Xr_train, yr_train,
                 Xr_val, yr_val, 'B3_4ch_nodip_noaug', use_aug=False)
    model_b3 = tf.keras.models.load_model(b3_keras)

y_pred_b3 = np.argmax(model_b3.predict(Xr_test, verbose=0), axis=1)
acc_b3 = accuracy_score(yr_test, y_pred_b3)
f1_b3  = f1_score(yr_test, y_pred_b3, average='weighted')
print(f"B3 — Accuracy: {acc_b3*100:.2f}%, F1: {f1_b3*100:.2f}%")
ablation_results['B3'] = {'accuracy': round(float(acc_b3),4), 'f1': round(float(f1_b3),4)}

# ─── B4: 4-Channel, DIP, SpecAugment — PROPOSED ──────────────────────────────
print("\n--- B4: 4-Channel, DIP, SpecAugment — PROPOSED METHOD ---")
b4_keras = os.path.join(cfg.MODEL_DIR, "B4_4ch_dip_aug_best.keras")
if os.path.exists(b4_keras):
    print("Model B4 sudah ada, skip training.")
    model_b4 = tf.keras.models.load_model(b4_keras)
else:
    model_b4, base_b4 = build_model_4ch()
    train_2phase(model_b4, base_b4, X_train, y_train,
                 X_val, y_val, 'B4_4ch_dip_aug', use_aug=True)
    model_b4 = tf.keras.models.load_model(b4_keras)

y_pred_b4 = np.argmax(model_b4.predict(X_test, verbose=0), axis=1)
acc_b4 = accuracy_score(y_test, y_pred_b4)
f1_b4  = f1_score(y_test, y_pred_b4, average='weighted')
print(f"B4 — Accuracy: {acc_b4*100:.2f}%, F1: {f1_b4*100:.2f}%")
ablation_results['B4'] = {'accuracy': round(float(acc_b4),4), 'f1': round(float(f1_b4),4)}
print(classification_report(y_test, y_pred_b4, target_names=cfg.EMOTION_LABELS))

# ─── Simpan ringkasan ablation ────────────────────────────────────────────────
# Format yang dibutuhkan step6
ablation_simple = {k: v['accuracy'] for k, v in ablation_results.items()}
with open(os.path.join(cfg.RESULT_DIR, "ablation_summary.json"), 'w') as f:
    json.dump(ablation_simple, f, indent=2)
with open(os.path.join(cfg.RESULT_DIR, "ablation_full.json"), 'w') as f:
    json.dump(ablation_results, f, indent=2)

# ─── Confusion matrix B4 ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_b4)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(11,9))
sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=cfg.EMOTION_LABELS, yticklabels=cfg.EMOTION_LABELS,
            ax=ax, cbar_kws={'label':'%'})
ax.set_xlabel('Prediksi', fontsize=12)
ax.set_ylabel('Aktual', fontsize=12)
ax.set_title('Confusion Matrix — B4 EfficientNetB0+SE+SpecAug', fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
cm_path = os.path.join(cfg.RESULT_DIR, "confusion_matrix_B4.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Confusion matrix disimpan: {cm_path}")

# ─── Ablation bar chart ───────────────────────────────────────────────────────
conditions = list(ablation_results.keys())
accs  = [ablation_results[c]['accuracy']*100 for c in conditions]
f1s   = [ablation_results[c]['f1']*100 for c in conditions]
colors = ['#95a5a6','#3498db','#e67e22','#2ecc71']
labels = ['B1\nMel,NoDIP\nNoAug', 'B2\nMel,DIP\n+Aug', 'B3\n4Ch,NoDIP\nNoAug', 'B4\n4Ch,DIP\n+Aug\n(Proposed)']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
fig.suptitle('Ablation Study — EfficientNetB0 + SE Block + SpecAugment\nDataset RAVDESS 1440 sampel', fontsize=13)
for ax, vals, title in [(ax1, accs, 'Accuracy (%)'), (ax2, f1s, 'F1-Score Weighted (%)')]:
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{v:.2f}%', ha='center', fontweight='bold', fontsize=10)
    ax.set_ylim(0, min(100, max(vals)+10))
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
plt.tight_layout()
chart_path = os.path.join(cfg.RESULT_DIR, "ablation_comparison.png")
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Ablation chart disimpan: {chart_path}")

# ─── RINGKASAN AKHIR ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("HASIL AKHIR ABLATION STUDY")
print("="*60)
print(f"{'Kondisi':<10} {'Accuracy':>12} {'F1 Weighted':>14}")
print("-"*38)
for cond in conditions:
    r = ablation_results[cond]
    print(f"{cond:<10} {r['accuracy']*100:>11.2f}% {r['f1']*100:>13.2f}%")
print(f"\nSemua hasil disimpan di: {cfg.RESULT_DIR}")
print("Pipeline selesai!")
