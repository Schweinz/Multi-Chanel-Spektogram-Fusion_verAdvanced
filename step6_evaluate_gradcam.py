"""
step6_evaluate_gradcam.py  --  Evaluasi Opsi C (ResNet50V2 + SE + BiGRU)
=========================================================================
Kompatibel dengan step5 versi RAM-efficient (on-the-fly PNG loading).

Bagian:
  A) Load model terbaik dari MODEL_DIR (checkpoint best_{cond}_opsiC.keras)
  B) Load test data on-the-fly dari PNG (sama persis dengan step5)
  C) Evaluasi: accuracy, weighted F1, confusion matrix, per-class report
  D) Grad-CAM: heatmap dari layer conv terakhir ResNet50V2
  E) Channel importance: rata-rata aktivasi Grad-CAM tiap channel per emosi
  F) Ablation comparison chart dari ablation_summary.json
"""

import os
import sys
import json
import glob
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

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

SPEC_DIR_DRIVE    = _cfg.SPEC_DIR
ENHANCED_DIR_DRIVE = _cfg.ENHANCED_DIR
MODEL_DIR         = _cfg.MODEL_DIR
RESULT_DIR        = _cfg.RESULT_DIR
N_CLASSES         = _cfg.N_CLASSES
EMOTION_LABELS    = _cfg.EMOTION_LABELS
EMOTION_MAP       = _cfg.EMOTION_MAP
CHANNEL_ORDER     = _cfg.CHANNEL_ORDER
IMG_SIZE          = _cfg.IMG_SIZE          # (224, 224)
RANDOM_SEED       = _cfg.RANDOM_SEED
TEST_SIZE         = _cfg.TEST_SIZE
VAL_SIZE          = _cfg.VAL_SIZE

import tensorflow as tf
# Mixed precision harus sama dengan step5
tf.keras.mixed_precision.set_global_policy('mixed_float16')


# Custom layer yang sama dengan step5 -- wajib ada agar load_model berhasil
class CastToFloat32(tf.keras.layers.Layer):
    """Cast tensor ke float32. Pengganti Lambda untuk mixed precision output."""
    def call(self, x):
        return tf.cast(x, tf.float32)

    def get_config(self):
        return super().get_config()

BATCH_SIZE = 8   # sama dengan step5

# Local SSD cache (sama dengan step5)
LOCAL_SPEC_DIR     = '/tmp/specs_nodip'
LOCAL_ENHANCED_DIR = '/tmp/specs_dip'


# ===========================================================================
# Utilitas: copy Drive -> /tmp/ (sama dengan step5)
# ===========================================================================

def copy_to_local(src_dir, dst_dir, desc=""):
    if os.path.exists(dst_dir) and len(glob.glob(f"{dst_dir}/*.png")) > 0:
        n = len(glob.glob(f"{dst_dir}/*.png"))
        print(f"[cache] {desc}: {n} PNG sudah ada di {dst_dir}")
        return dst_dir
    print(f"[copy] {desc}: {src_dir} -> {dst_dir} ...")
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(f"{src_dir}/*.png")
    if not files:
        return src_dir
    for f in files:
        shutil.copy2(f, dst_dir)
    print(f"  Selesai: {len(files)} file disalin")
    return dst_dir


# ===========================================================================
# File list + split (sama persis dengan step5 agar test set identik)
# ===========================================================================

def parse_ravdess_label(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    ravdess_part = base.rsplit('_', 1)[0]
    parts = ravdess_part.split('-')
    if len(parts) < 3:
        return None
    emotion_name = EMOTION_MAP.get(parts[2])
    if emotion_name is None:
        return None
    return EMOTION_LABELS.index(emotion_name)


def build_file_list(spec_dir):
    all_png = glob.glob(f"{spec_dir}/*.png")
    if not all_png:
        raise FileNotFoundError(f"Tidak ada PNG di {spec_dir}")
    sample_files = defaultdict(dict)
    for p in all_png:
        name = os.path.splitext(os.path.basename(p))[0]
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in CHANNEL_ORDER:
            sample_files[parts[0]][parts[1]] = p
    file_list = []
    for basename, fd in sample_files.items():
        if not all(ch in fd for ch in CHANNEL_ORDER):
            continue
        label = parse_ravdess_label(f"{basename}_mel.png")
        if label is None:
            continue
        file_list.append((fd['mel'], fd['mfcc'], fd['cqt'], fd['chroma'], label))
    return file_list


def get_test_list(spec_dir):
    """
    Gunakan RANDOM_SEED dan TEST_SIZE yang sama dengan step5
    agar test set persis identik.
    """
    file_list = build_file_list(spec_dir)
    labels = [fl[4] for fl in file_list]
    idx = list(range(len(file_list)))
    idx_temp, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    test_list = [file_list[i] for i in idx_test]
    print(f"  Test set: {len(test_list)} sampel")
    return test_list


def load_test_numpy(test_list, use_4ch=True):
    """
    Load test set ke numpy array (kecil -- hanya 216 sampel ~400MB untuk 224x224x4).
    Dipakai untuk Grad-CAM yang butuh akses per-sampel.
    """
    H, W = IMG_SIZE
    X, y = [], []
    for mel_p, mfcc_p, cqt_p, chroma_p, label in test_list:
        def read(p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros((H, W), dtype=np.float32)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            return img.astype(np.float32) / 255.0

        mel = read(mel_p)
        if use_4ch:
            channels = [mel, read(mfcc_p), read(cqt_p), read(chroma_p)]
        else:
            channels = [mel, mel, mel, mel]
        X.append(np.stack(channels, axis=-1))
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ===========================================================================
# Load model
# ===========================================================================

def load_best_model(condition='B4'):
    """
    Load checkpoint step5 Opsi C.
    Nama file: best_{condition}_opsiC.keras  atau  best_{condition}_opsiC.h5

    Gunakan custom_objects={'CastToFloat32': CastToFloat32} agar layer
    CastToFloat32 dikenali saat deserialisasi.
    Kalau model lama masih pakai Lambda, fallback ke safe_mode=False.
    """
    custom_objs = {'CastToFloat32': CastToFloat32}

    keras_path = os.path.join(MODEL_DIR, f"best_{condition}_opsiC.keras")
    h5_path    = os.path.join(MODEL_DIR, f"best_{condition}_opsiC.h5")

    for path in [keras_path, h5_path]:
        if not os.path.exists(path):
            continue
        print(f"Loading: {path}")
        # Coba dulu dengan custom_objects (model baru pakai CastToFloat32)
        try:
            return tf.keras.models.load_model(path, custom_objects=custom_objs)
        except Exception as e1:
            print(f"  [info] Load normal gagal: {e1}")
            # Fallback: safe_mode=False untuk model lama yang masih pakai Lambda
            try:
                print("  [info] Mencoba safe_mode=False (model lama dengan Lambda layer)...")
                return tf.keras.models.load_model(
                    path, custom_objects=custom_objs, safe_mode=False
                )
            except Exception as e2:
                print(f"  [error] safe_mode=False juga gagal: {e2}")
                raise

    raise FileNotFoundError(
        f"Model {condition} tidak ditemukan.\n"
        f"  Dicari: {keras_path}\n"
        f"  Dicari: {h5_path}\n"
        "Pastikan step5 sudah selesai training."
    )


# ===========================================================================
# Evaluasi
# ===========================================================================

def evaluate_model(model, X_test, y_test, condition_name):
    """Hitung accuracy, F1, confusion matrix, per-class report."""
    y_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    report_str = classification_report(
        y_test, y_pred, target_names=EMOTION_LABELS, digits=4
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=EMOTION_LABELS, output_dict=True
    )

    print(f"\nEvaluasi {condition_name}:")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  F1 (weighted): {f1*100:.2f}%")
    print(f"\nClassification Report:\n{report_str}")

    result = {
        'condition':    condition_name,
        'accuracy':     round(float(acc), 4),
        'f1_weighted':  round(float(f1), 4),
        'per_class':    report_dict
    }
    with open(os.path.join(RESULT_DIR, f"eval_{condition_name}.json"), 'w') as f:
        json.dump(result, f, indent=2)

    return y_pred, y_pred_prob, result


def plot_confusion_matrix(y_true, y_pred, condition_name):
    """Confusion matrix ternormalisasi (%)."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_norm, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS,
        ax=ax, cbar_kws={'label': '%'}
    )
    ax.set_xlabel('Prediksi', fontsize=12)
    ax.set_ylabel('Aktual', fontsize=12)
    ax.set_title(f'Confusion Matrix -- {condition_name} (ResNet50V2+SE+BiGRU)', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    out = os.path.join(RESULT_DIR, f"confusion_matrix_{condition_name}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix -> {out}")


# ===========================================================================
# Grad-CAM  (kompatibel dengan ResNet50V2)
# ===========================================================================

def get_last_conv_layer(model):
    """
    Cari layer konvolusi terakhir yang outputnya spatial (H,W,C) di ResNet50V2.
    ResNet50V2 biasanya punya 'post_bn' atau 'conv5_block3_out' sebagai output terakhir.
    """
    try:
        backbone = model.get_layer('resnet50v2')
        # Cari layer spatial terakhir di backbone
        for layer in reversed(backbone.layers):
            if hasattr(layer, 'output_shape'):
                s = layer.output_shape
                if isinstance(s, (list, tuple)) and len(s) == 4:
                    return layer, backbone
    except ValueError:
        pass

    # Fallback: cari di model utama
    for layer in reversed(model.layers):
        if hasattr(layer, 'output_shape'):
            s = layer.output_shape
            if isinstance(s, (list, tuple)) and len(s) == 4:
                return layer, model
    return None, model


def compute_gradcam(model, img_array, class_idx):
    """
    Grad-CAM untuk model ResNet50V2+SE+BiGRU.
    img_array: numpy (1, 224, 224, 4), float32
    Returns: heatmap (224, 224) float32 [0,1]
    """
    H, W = IMG_SIZE

    try:
        backbone = model.get_layer('resnet50v2')
    except ValueError:
        return np.zeros((H, W), dtype=np.float32)

    # Cari layer conv terakhir yang outputnya spatial
    target_layer = None
    for layer in reversed(backbone.layers):
        if hasattr(layer, 'output_shape'):
            s = layer.output_shape
            if isinstance(s, (list, tuple)) and len(s) == 4:
                target_layer = layer
                break

    if target_layer is None:
        return np.zeros((H, W), dtype=np.float32)

    # Build grad model: input model -> (feature_map, predictions)
    try:
        # Dapatkan output backbone dari input model
        backbone_input  = model.input
        backbone_output = backbone(model.get_layer('proj_relu').output, training=False)

        # Model terpisah untuk dapat feature map backbone
        feat_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[backbone_output, model.output]
        )

        with tf.GradientTape() as tape:
            inp = tf.cast(img_array, tf.float32)
            inp_var = tf.Variable(inp, trainable=False)
            feat_out, preds = feat_model(inp_var, training=False)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, feat_out)
        if grads is None:
            return np.zeros((H, W), dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        feat_map = feat_out[0]
        heatmap = feat_map @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap_np = heatmap.numpy()
        return cv2.resize(heatmap_np, (W, H)).astype(np.float32)

    except Exception as e:
        print(f"  [Grad-CAM] Error: {e} -- skip")
        return np.zeros((H, W), dtype=np.float32)


def visualize_gradcam_per_channel(model, X_sample, y_true, sample_idx=0):
    """
    Visualisasikan Grad-CAM overlay untuk satu sampel, 4 channel.
    Baris atas: spektrogram asli. Baris bawah: Grad-CAM overlay.
    """
    H, W = IMG_SIZE
    img = X_sample[sample_idx:sample_idx+1]
    true_label = EMOTION_LABELS[y_true[sample_idx]]

    y_pred_prob = model.predict(img, batch_size=1, verbose=0)
    pred_idx   = int(np.argmax(y_pred_prob))
    pred_label = EMOTION_LABELS[pred_idx]
    pred_conf  = float(y_pred_prob[0, pred_idx])

    heatmap = compute_gradcam(model, img, pred_idx)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"Grad-CAM (ResNet50V2+SE+BiGRU) -- Sampel #{sample_idx}\n"
        f"Aktual: {true_label} | Prediksi: {pred_label} ({pred_conf:.1%})",
        fontsize=12
    )

    for col, ch in enumerate(CHANNEL_ORDER):
        orig_img = img[0, :, :, col]

        # Baris atas: spektrogram asli
        axes[0, col].imshow(orig_img, cmap='viridis', aspect='auto', origin='lower')
        axes[0, col].set_title(ch.upper())
        axes[0, col].axis('off')

        # Baris bawah: Grad-CAM overlay
        orig_uint8 = (orig_img * 255).astype(np.uint8)
        orig_rgb   = cv2.applyColorMap(orig_uint8, cv2.COLORMAP_VIRIDIS)
        heat_uint8 = (heatmap * 255).astype(np.uint8)
        heat_colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_rgb, 0.6, heat_colored, 0.4, 0)
        axes[1, col].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                            aspect='auto', origin='lower')
        axes[1, col].set_title(f"Grad-CAM ({ch.upper()})")
        axes[1, col].axis('off')

    plt.tight_layout()
    out = os.path.join(RESULT_DIR, f"gradcam_sample{sample_idx}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM -> {out}")


def compute_channel_importance(model, X_test, y_test, n_samples=80):
    """Channel importance: rata-rata Grad-CAM × channel per emosi."""
    H, W = IMG_SIZE
    importance = np.zeros((N_CLASSES, len(CHANNEL_ORDER)), dtype=np.float32)
    counts = np.zeros(N_CLASSES, dtype=np.int32)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

    print(f"\nMenghitung channel importance ({len(indices)} sampel)...")
    for idx in indices:
        img = X_test[idx:idx+1]
        true_cls = int(y_test[idx])
        heatmap = compute_gradcam(model, img, true_cls)
        for ch_idx in range(len(CHANNEL_ORDER)):
            ch_act = float((img[0, :, :, ch_idx] * heatmap).mean())
            importance[true_cls, ch_idx] += ch_act
        counts[true_cls] += 1

    for i in range(N_CLASSES):
        if counts[i] > 0:
            importance[i] /= counts[i]

    row_max = importance.max(axis=1, keepdims=True) + 1e-8
    importance_norm = importance / row_max

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        importance_norm, annot=True, fmt='.2f', cmap='YlOrRd',
        xticklabels=[c.upper() for c in CHANNEL_ORDER],
        yticklabels=EMOTION_LABELS,
        ax=ax, linewidths=0.5
    )
    ax.set_title("Channel Importance per Emosi (ResNet50V2+SE+BiGRU)", fontsize=12)
    ax.set_xlabel("Channel Spektrogram")
    ax.set_ylabel("Emosi")
    plt.tight_layout()

    out = os.path.join(RESULT_DIR, "channel_importance.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Channel importance -> {out}")

    with open(os.path.join(RESULT_DIR, "channel_importance.json"), 'w') as f:
        json.dump({
            EMOTION_LABELS[i]: {
                CHANNEL_ORDER[j]: round(float(importance_norm[i, j]), 4)
                for j in range(len(CHANNEL_ORDER))
            }
            for i in range(N_CLASSES)
        }, f, indent=2)
    return importance_norm


# ===========================================================================
# Ablation comparison chart
# ===========================================================================

def plot_ablation_comparison():
    """
    Bar chart B1-B4 dari ablation_summary.json (format list-of-dicts step5 baru).
    """
    summary_path = os.path.join(RESULT_DIR, "ablation_summary.json")
    if not os.path.exists(summary_path):
        print("ablation_summary.json tidak ditemukan. Skip.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    # Format baru: list of dicts dengan key 'condition' dan 'test_accuracy'
    if isinstance(summary, list):
        conditions = [r['condition'] for r in summary]
        accuracies = [r['test_accuracy'] * 100 for r in summary]
        f1_scores  = [r['test_f1_weighted'] * 100 for r in summary]
    else:
        # Format lama fallback
        conditions = list(summary.keys())
        accuracies = [summary[c] * 100 for c in conditions]
        f1_scores  = accuracies

    labels_full = [
        'B1\nMel, No DIP\nNo Aug',
        'B2\nMel, DIP\n+SpecAugment',
        'B3\n4-Ch, No DIP\nNo Aug',
        'B4 (Proposed)\n4-Ch, DIP\n+SpecAugment',
    ]
    # Sesuaikan jumlah label
    labels_full = labels_full[:len(conditions)]
    colors = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71'][:len(conditions)]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(conditions))
    w = 0.35
    bars_acc = ax.bar(x - w/2, accuracies, w, label='Accuracy', color=colors, alpha=0.9)
    bars_f1  = ax.bar(x + w/2, f1_scores,  w, label='F1 Weighted',
                      color=colors, alpha=0.55, edgecolor='gray')

    for bar, val in zip(bars_acc, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, val in zip(bars_f1, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='#555')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_full, fontsize=10)
    ax.set_ylim(0, min(100, max(accuracies + f1_scores) + 12))
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Ablation Study -- ResNet50V2 + SE Block + BiGRU + SpecAugment', fontsize=13)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out = os.path.join(RESULT_DIR, "ablation_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ablation chart -> {out}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("STEP 6: Evaluasi Model + Grad-CAM (Opsi C)")
    print("    Backbone : ResNet50V2 + SE + BiGRU")
    print("    Input    : 224x224x4")
    print("=" * 55)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # --- 1. Copy spektrogram ke local SSD (kalau belum ada) ---
    print("\n[1/5] Pastikan spektrogram tersedia lokal...")
    local_dip = copy_to_local(ENHANCED_DIR_DRIVE, LOCAL_ENHANCED_DIR,
                               "dengan DIP (B2/B4)")

    # --- 2. Bangun test list yang identik dengan step5 ---
    print("\n[2/5] Bangun test list...")
    test_list = get_test_list(local_dip)

    # --- 3. Load test data ke numpy (hanya ~216 sampel, aman di RAM) ---
    print("\n[3/5] Load test set ke numpy...")
    X_test, y_test = load_test_numpy(test_list, use_4ch=True)
    print(f"  X_test: {X_test.shape}  y_test: {y_test.shape}")

    # --- 4. Load model B4 dan evaluasi ---
    print("\n[4/5] Load model B4 dan evaluasi...")
    model_b4 = load_best_model('B4')
    model_b4.summary(line_length=90)

    y_pred, y_pred_prob, result = evaluate_model(model_b4, X_test, y_test, 'B4')
    plot_confusion_matrix(y_test, y_pred, 'B4')

    # --- 5. Grad-CAM (satu sampel per emosi, max 4) ---
    print("\n[5/5] Grad-CAM...")
    shown_emotions = set()
    for idx in range(len(y_test)):
        emotion = int(y_test[idx])
        if emotion not in shown_emotions:
            visualize_gradcam_per_channel(model_b4, X_test, y_test, sample_idx=idx)
            shown_emotions.add(emotion)
        if len(shown_emotions) >= 4:
            break

    importance = compute_channel_importance(model_b4, X_test, y_test, n_samples=80)
    plot_ablation_comparison()

    # --- Ringkasan akhir ---
    print("\n" + "=" * 55)
    print(f"HASIL AKHIR B4 (Proposed Method):")
    print(f"  Accuracy    : {result['accuracy']*100:.2f}%")
    print(f"  F1 Weighted : {result['f1_weighted']*100:.2f}%")
    print(f"\nSemua hasil disimpan di: {RESULT_DIR}")
    print("Step 6 selesai.")
