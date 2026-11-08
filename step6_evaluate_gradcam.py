"""
step6_evaluate_gradcam.py
=========================
STEP KEENAM — evaluasi model + Grad-CAM per channel.

Sama strukturnya dengan paper pertama, disesuaikan untuk:
1. EfficientNetB0 sebagai backbone (bukan MobileNetV2)
2. SE Block menambah satu lapisan di antara backbone dan GAP
3. Layer target Grad-CAM: layer konvolusi terakhir EfficientNetB0
   sebelum Global Average Pooling

Bagian-bagian:
  A) Load model terbaik dari MODEL_DIR
  B) Evaluasi: accuracy, weighted F1, confusion matrix
  C) Grad-CAM: heatmap per channel dari layer terakhir EfficientNetB0
  D) Channel importance: rata-rata aktivasi Grad-CAM tiap channel per emosi
  E) Ringkasan perbandingan B1-B4 dari ablation_summary.json
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

sys.path.append('/content/drive/MyDrive/SER_Advanced')
from config import (
    FUSED_DIR, MODEL_DIR, RESULT_DIR,
    EMOTION_LABELS, N_CLASSES, CHANNEL_ORDER
)

import tensorflow as tf
from tensorflow.keras.models import load_model


# ── LOAD MODEL ───────────────────────────────────────────────────────────────

def load_best_model(condition='B4'):
    """
    Load model terbaik dari MODEL_DIR.
    Kalau tidak ada file .keras, coba .h5 sebagai fallback.
    """
    name_map = {
        'B1': 'B1_mel_nodip_noaug',
        'B2': 'B2_mel_dip_aug',
        'B3': 'B3_4ch_nodip_noaug',
        'B4': 'B4_4ch_dip_aug',
    }
    model_name = name_map[condition]
    keras_path = os.path.join(MODEL_DIR, f"{model_name}_best.keras")
    h5_path    = os.path.join(MODEL_DIR, f"{model_name}_best.h5")

    if os.path.exists(keras_path):
        print(f"Loading model dari {keras_path}")
        return load_model(keras_path)
    elif os.path.exists(h5_path):
        print(f"Loading model dari {h5_path}")
        return load_model(h5_path)
    else:
        raise FileNotFoundError(
            f"Model untuk {condition} tidak ditemukan. Jalankan step5 dulu."
        )


def load_test_data():
    data_path = os.path.join(FUSED_DIR, "dataset.npz")
    data = np.load(data_path)
    return data['X_test'], data['y_test']


# ── EVALUASI DASAR ────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, condition_name):
    """
    Evaluasi dengan accuracy dan weighted F1.
    Weighted F1 lebih informatif dari accuracy kalau distribusi emosi tidak rata.
    """
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(
        y_test, y_pred, target_names=EMOTION_LABELS, digits=4
    )

    print(f"\nEvaluasi {condition_name}:")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  F1 (weighted): {f1*100:.2f}%")
    print(f"\nClassification Report:\n{report}")

    result = {
        'accuracy': round(float(acc), 4),
        'f1_weighted': round(float(f1), 4),
        'per_class': {}
    }
    for i, label in enumerate(EMOTION_LABELS):
        mask = (y_test == i)
        if mask.sum() > 0:
            result['per_class'][label] = {
                'accuracy': round(float(accuracy_score(y_test[mask], y_pred[mask])), 4),
                'n_samples': int(mask.sum())
            }

    with open(os.path.join(RESULT_DIR, f"eval_{condition_name}.json"), 'w') as f:
        json.dump(result, f, indent=2)

    return y_pred, y_pred_prob, result


def plot_confusion_matrix(y_true, y_pred, condition_name):
    """
    Confusion matrix ternormalisasi (%).
    Diagonal = per-emosi recall. Off-diagonal = pola kebingungan.
    """
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
    ax.set_title(f'Confusion Matrix — {condition_name} (EfficientNetB0+SE)', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    out = os.path.join(RESULT_DIR, f"confusion_matrix_{condition_name}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix disimpan ke {out}")


# ── GRAD-CAM ─────────────────────────────────────────────────────────────────

def get_last_conv_layer_name(model):
    """
    Cari nama layer konvolusi terakhir sebelum GlobalAveragePooling.
    Untuk EfficientNetB0, biasanya 'top_activation' atau nama serupa.
    """
    # Cari semua layer Conv/Activation dalam backbone EfficientNetB0
    for layer in reversed(model.layers):
        if 'efficientnetb0' in layer.name:
            # Dapatkan layer dalam sub-model
            inner = model.get_layer('efficientnetb0')
            for inner_layer in reversed(inner.layers):
                if isinstance(inner_layer, (
                    tf.keras.layers.Conv2D,
                    tf.keras.layers.Activation,
                    tf.keras.layers.BatchNormalization
                )):
                    # Layer terakhir sebelum flatten / GAP
                    if len(inner_layer.output_shape) == 4:
                        return inner_layer.name, inner
    # Fallback: cari langsung di model utama
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name, model
    return None, model


def compute_gradcam(model, img_array, class_idx):
    """
    Grad-CAM: ukur seberapa kuat gradient kelas tertentu mengalir
    ke tiap pixel di feature map layer terakhir backbone.

    img_array: (1, 128, 128, C)
    Returns: heatmap (128, 128) float32
    """
    # Dapatkan sub-model EfficientNetB0
    efficientnet = model.get_layer('efficientnetb0')

    # Cari layer conv terakhir yang outputnya spatial (H,W,C)
    target_layer = None
    for layer in reversed(efficientnet.layers):
        if hasattr(layer, 'output') and len(layer.output_shape) == 4:
            target_layer = layer
            break

    if target_layer is None:
        return np.zeros((128, 128), dtype=np.float32)

    # Buat model dari input → target layer output + prediksi akhir
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_idx]

    # Gradient dari kelas terhadap feature map
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Bobot feature map dengan gradient
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    # Resize ke ukuran gambar asli
    heatmap_np = heatmap.numpy()
    heatmap_resized = cv2.resize(heatmap_np, (128, 128))
    return heatmap_resized.astype(np.float32)


def visualize_gradcam_per_channel(model, X_sample, y_true, sample_idx=0):
    """
    Visualisasikan Grad-CAM untuk satu sampel.
    Tampilkan: (atas) spektrogram asli 4 channel
               (bawah) heatmap Grad-CAM overlay
    Ini memperlihatkan di bagian mana gambar yang paling memengaruhi prediksi.
    """
    img = X_sample[sample_idx:sample_idx+1]
    true_label = EMOTION_LABELS[y_true[sample_idx]]

    y_pred_prob = model.predict(img, verbose=0)
    pred_idx = int(np.argmax(y_pred_prob))
    pred_label = EMOTION_LABELS[pred_idx]
    pred_conf = float(y_pred_prob[0, pred_idx])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"Grad-CAM (EfficientNetB0+SE) — Sampel #{sample_idx}\n"
        f"Aktual: {true_label} | Prediksi: {pred_label} ({pred_conf:.1%})",
        fontsize=12
    )

    for col, ch in enumerate(CHANNEL_ORDER):
        orig_img = img[0, :, :, col]
        heatmap = compute_gradcam(model, img, pred_idx)

        # Baris atas: spektrogram asli
        axes[0, col].imshow(orig_img, cmap='viridis', aspect='auto', origin='lower')
        axes[0, col].set_title(ch.upper())
        axes[0, col].axis('off')

        # Baris bawah: overlay Grad-CAM
        orig_rgb = cv2.applyColorMap(
            (orig_img * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
        )
        heat_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(orig_rgb, 0.6, heat_colored, 0.4, 0)
        axes[1, col].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                            aspect='auto', origin='lower')
        axes[1, col].set_title(f"Grad-CAM ({ch.upper()})")
        axes[1, col].axis('off')

    plt.tight_layout()
    out = os.path.join(RESULT_DIR, f"gradcam_sample{sample_idx}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM disimpan ke {out}")


def compute_channel_importance(model, X_test, y_test, n_samples=100):
    """
    Channel importance: hitung rata-rata skor Grad-CAM per channel per emosi.
    n_samples: batasi jumlah sampel supaya tidak terlalu lama.

    Visualisasi sebagai heatmap 8×4 (emosi × channel).
    """
    importance = np.zeros((N_CLASSES, len(CHANNEL_ORDER)), dtype=np.float32)
    counts = np.zeros(N_CLASSES, dtype=np.int32)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

    print(f"\nMenghitung channel importance ({len(indices)} sampel)...")
    for idx in indices:
        img = X_test[idx:idx+1]
        true_cls = int(y_test[idx])
        heatmap = compute_gradcam(model, img, true_cls)
        mean_act = float(heatmap.mean())

        # Tiap channel punya gambar berbeda tapi heatmap sama
        # (heatmap dihitung dari input gabungan 4-channel)
        # Untuk per-channel importance, kita kalikan heatmap dengan masing-masing channel
        for ch_idx in range(len(CHANNEL_ORDER)):
            ch_act = float((img[0, :, :, ch_idx] * heatmap).mean())
            importance[true_cls, ch_idx] += ch_act
        counts[true_cls] += 1

    for i in range(N_CLASSES):
        if counts[i] > 0:
            importance[i] /= counts[i]

    # Normalisasi per baris supaya mudah dibaca
    row_max = importance.max(axis=1, keepdims=True) + 1e-8
    importance_norm = importance / row_max

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        importance_norm, annot=True, fmt='.2f', cmap='YlOrRd',
        xticklabels=[c.upper() for c in CHANNEL_ORDER],
        yticklabels=EMOTION_LABELS,
        ax=ax, linewidths=0.5
    )
    ax.set_title("Channel Importance per Emosi (EfficientNetB0+SE)", fontsize=12)
    ax.set_xlabel("Channel Spektrogram")
    ax.set_ylabel("Emosi")
    plt.tight_layout()

    out = os.path.join(RESULT_DIR, "channel_importance.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Channel importance disimpan ke {out}")

    with open(os.path.join(RESULT_DIR, "channel_importance.json"), 'w') as f:
        json.dump({
            EMOTION_LABELS[i]: {
                CHANNEL_ORDER[j]: round(float(importance_norm[i, j]), 4)
                for j in range(len(CHANNEL_ORDER))
            }
            for i in range(N_CLASSES)
        }, f, indent=2)
    return importance_norm


# ── RINGKASAN ABLATION ────────────────────────────────────────────────────────

def plot_ablation_comparison():
    """
    Baca ablation_summary.json dari step5 dan buat bar chart perbandingan B1-B4.
    """
    summary_path = os.path.join(RESULT_DIR, "ablation_summary.json")
    if not os.path.exists(summary_path):
        print("ablation_summary.json tidak ditemukan. Skip chart ablation.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    conditions = list(summary.keys())
    accuracies  = [summary[c] * 100 for c in conditions]
    colors = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71']
    labels_full = [
        'B1\nMel, No DIP\nNo Aug',
        'B2\nMel, DIP\n+SpecAugment',
        'B3\n4-Ch, No DIP\nNo Aug',
        'B4 (Proposed)\n4-Ch, DIP\n+SpecAugment',
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels_full, accuracies, color=colors, width=0.55, edgecolor='white')

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11
        )

    ax.set_ylim(0, min(100, max(accuracies) + 8))
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study — EfficientNetB0 + SE Block + SpecAugment', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out = os.path.join(RESULT_DIR, "ablation_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ablation chart disimpan ke {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("STEP 6: Evaluasi Model + Grad-CAM")
    print("=" * 55)
    os.makedirs(RESULT_DIR, exist_ok=True)

    X_test, y_test = load_test_data()
    print(f"Test set: {len(X_test)} sampel")

    # Evaluasi model terbaik (B4 proposed method)
    model_b4 = load_best_model('B4')
    model_b4.summary(line_length=90)

    y_pred, y_pred_prob, result = evaluate_model(
        model_b4, X_test, y_test, 'B4'
    )
    plot_confusion_matrix(y_test, y_pred, 'B4')

    # Grad-CAM — tampilkan beberapa sampel (satu per emosi kalau ada)
    shown_emotions = set()
    for idx in range(len(y_test)):
        emotion = int(y_test[idx])
        if emotion not in shown_emotions:
            visualize_gradcam_per_channel(model_b4, X_test, y_test, sample_idx=idx)
            shown_emotions.add(emotion)
        if len(shown_emotions) >= 4:   # cukup 4 emosi untuk demo
            break

    # Channel importance
    importance = compute_channel_importance(model_b4, X_test, y_test, n_samples=100)

    # Ablation comparison chart
    plot_ablation_comparison()

    print("\n" + "="*55)
    print(f"HASIL AKHIR B4 (Proposed Method):")
    print(f"  Accuracy    : {result['accuracy']*100:.2f}%")
    print(f"  F1 Weighted : {result['f1_weighted']*100:.2f}%")
    print(f"\nSemua hasil disimpan di: {RESULT_DIR}")
    print("Step 6 selesai.")
