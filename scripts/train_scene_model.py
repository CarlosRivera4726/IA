"""
Reentrena models/sentinel_security_model.keras con una CNN Residual + GroupNormalization
(sin BatchNormalization para evitar el sesgo en inferencia de Keras 3 + PyTorch backend).
- Clases canónicas: ['Arma_Blanca', 'Arma_De_Fuego', 'Ataque_Agresion', 'Escena_Normal', 'Hurto_Sospechoso']
- Entrada: (224, 224, 3) normalizada en [-1, 1]
- Evalúa estrictamente sobre datasets/holdout/ (0% fuga de datos) y actualiza models/model_metadata.json.
"""

import json
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))

import keras
import numpy as np
import pandas as pd
from keras import layers
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

random.seed(42)
np.random.seed(42)

SCENE_CLASSES = [
    "Arma_Blanca",
    "Arma_De_Fuego",
    "Ataque_Agresion",
    "Escena_Normal",
    "Hurto_Sospechoso",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(SCENE_CLASSES)}

DATASET_SCENE = PROJECT_ROOT / "datasets" / "sentinel_real_dataset"
DATASET_HOLDOUT = PROJECT_ROOT / "datasets" / "holdout"
HAR_DIR = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "meetnagadia"
    / "human-action-recognition-har-dataset"
    / "versions"
    / "1"
    / "Human Action Recognition"
)
OUT_MODEL = PROJECT_ROOT / "models" / "sentinel_security_model.keras"
META_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def load_image(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def build_datasets():
    holdout_names = set()
    X_ho, y_ho = [], []
    for cls in SCENE_CLASSES:
        for f in sorted((DATASET_HOLDOUT / cls).glob("*")):
            if f.is_file():
                holdout_names.add(f.name)
                try:
                    X_ho.append(load_image(f))
                    y_ho.append(CLASS_TO_IDX[cls])
                except Exception:
                    continue

    seen_names = set(holdout_names)
    samples = []
    for cls in SCENE_CLASSES:
        for f in sorted((DATASET_SCENE / cls).glob("*")):
            if f.is_file() and f.name not in seen_names:
                seen_names.add(f.name)
                samples.append((f, CLASS_TO_IDX[cls]))

    # Enriquecer Escena_Normal, Ataque_Agresion y Hurto_Sospechoso desde HAR cache sin tocar holdout
    har_csv = HAR_DIR / "Training_set.csv"
    har_train = HAR_DIR / "train"
    if har_csv.is_file() and har_train.is_dir():
        df = pd.read_csv(har_csv)
        normal_extra, fight_extra, hurto_extra = [], [], []
        for _, row in df.iterrows():
            fname = str(row["filename"])
            if fname in seen_names:
                continue
            fpath = har_train / fname
            if not fpath.is_file():
                continue
            lbl = str(row["label"])
            if lbl in ("sitting", "using_laptop", "sleeping", "drinking", "listening_to_music", "laughing"):
                normal_extra.append(fpath)
            elif lbl == "fighting":
                fight_extra.append(fpath)
            elif lbl in ("calling", "texting", "clapping", "eating"):
                hurto_extra.append(fpath)
        random.shuffle(normal_extra)
        random.shuffle(fight_extra)
        random.shuffle(hurto_extra)
        for f in normal_extra[:400]:
            seen_names.add(f.name)
            samples.append((f, CLASS_TO_IDX["Escena_Normal"]))
        for f in fight_extra[:200]:
            seen_names.add(f.name)
            samples.append((f, CLASS_TO_IDX["Ataque_Agresion"]))
        for f in hurto_extra[:200]:
            seen_names.add(f.name)
            samples.append((f, CLASS_TO_IDX["Hurto_Sospechoso"]))

    random.shuffle(samples)
    X_all, y_all = [], []
    for f, lbl in samples:
        try:
            X_all.append(load_image(f))
            y_all.append(lbl)
        except Exception:
            continue

    X_all = np.stack(X_all)
    y_all = np.array(y_all, dtype=np.int64)
    X_ho = np.stack(X_ho)
    y_ho = np.array(y_ho, dtype=np.int64)

    tr_idx, va_idx = [], []
    for c in range(len(SCENE_CLASSES)):
        idx_c = np.where(y_all == c)[0]
        np.random.shuffle(idx_c)
        n_va = max(14, int(len(idx_c) * 0.15))
        va_idx.extend(idx_c[:n_va])
        tr_idx.extend(idx_c[n_va:])
    np.random.shuffle(tr_idx)
    np.random.shuffle(va_idx)

    return X_all[tr_idx], y_all[tr_idx], X_all[va_idx], y_all[va_idx], X_ho, y_ho


class SceneSeq(keras.utils.PyDataset):
    def __init__(self, X, y, bs=32, shuffle=False, **kw):
        super().__init__(**kw)
        self.X, self.y, self.bs, self.shuffle = X, y, bs, shuffle
        self.indices = np.arange(len(y))
        if shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.y) / self.bs))

    def __getitem__(self, i):
        idx = self.indices[i * self.bs : (i + 1) * self.bs]
        return (self.X[idx].astype("float32") / 127.5) - 1.0, self.y[idx]


def res_block(x, filters, downsample=True):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.GroupNormalization(groups=8)(shortcut)

    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    y = layers.GroupNormalization(groups=8)(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.GroupNormalization(groups=8)(y)

    out = layers.Add()([shortcut, y])
    out = layers.Activation("relu")(out)
    if downsample:
        out = layers.MaxPooling2D(2)(out)
    return out


def main():
    print("Cargando datasets de escenas (holdout aislado)...")
    X_tr, y_tr, X_va, y_va, X_ho, y_ho = build_datasets()
    print("Train:", X_tr.shape, np.bincount(y_tr))
    print("Val:  ", X_va.shape, np.bincount(y_va))
    print("Hold: ", X_ho.shape, np.bincount(y_ho))

    counts = np.bincount(y_tr, minlength=len(SCENE_CLASSES)).astype(np.float64)
    raw_w = len(y_tr) / (len(SCENE_CLASSES) * np.maximum(counts, 1.0))
    class_weight = {i: float(np.clip(raw_w[i] ** 0.7, 0.75, 2.5)) for i in range(len(SCENE_CLASSES))}
    print("Class weights:", {SCENE_CLASSES[k]: round(v, 2) for k, v in class_weight.items()})

    inp = layers.Input(shape=(224, 224, 3))
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.Resizing(112, 112)(x)

    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Activation("relu")(x)

    x = res_block(x, 48, downsample=True)
    x = res_block(x, 96, downsample=True)
    x = res_block(x, 144, downsample=True)
    x = res_block(x, 192, downsample=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(len(SCENE_CLASSES), activation="softmax")(x)
    model = keras.Model(inp, out, name="Custom_ResCNN_SentinelCV_GroupNorm")

    tr_ds = SceneSeq(X_tr, y_tr, bs=32, shuffle=True)
    va_ds = SceneSeq(X_va, y_va, bs=32, shuffle=False)
    ho_ds = SceneSeq(X_ho, y_ho, bs=32, shuffle=False)

    EPOCHS = 28
    lr = keras.optimizers.schedules.CosineDecay(1.5e-3, EPOCHS * len(tr_ds), alpha=0.05)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    ckpt = keras.callbacks.ModelCheckpoint(
        str(OUT_MODEL), monitor="val_accuracy", save_best_only=True, verbose=1
    )
    model.fit(
        tr_ds,
        validation_data=va_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[ckpt],
        verbose=2,
    )

    best_model = keras.models.load_model(str(OUT_MODEL), compile=False)
    ho_probs = best_model.predict(ho_ds, verbose=0)
    ho_pred = ho_probs.argmax(axis=1)

    acc = float(accuracy_score(y_ho, ho_pred))
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_ho, ho_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_ho, ho_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_ho, ho_pred).tolist()
    rep = classification_report(y_ho, ho_pred, target_names=SCENE_CLASSES, output_dict=True, zero_division=0)

    print(f"\nHoldout Accuracy: {acc:.4f} | Macro F1: {f1_mac:.4f}")
    for c_i, c_name in enumerate(SCENE_CLASSES):
        c_rec = float((ho_pred[y_ho == c_i] == c_i).mean())
        print(f"  {c_name:18s} recall={c_rec:.3f} precision={rep[c_name]['precision']:.3f}")

    if META_PATH.is_file():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        meta["scene_champion"] = {
            "Modelo": "ResCNN GroupNorm (sin BatchNorm)",
            "Accuracy": acc,
            "Precision (macro)": float(p_mac),
            "Recall (macro)": float(r_mac),
            "F1 (macro)": float(f1_mac),
            "Precision (weighted)": float(p_w),
            "Recall (weighted)": float(r_w),
            "F1 (weighted)": float(f1_w),
            "classification_report": rep,
            "confusion_matrix": cm,
        }
        META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Metadata actualizada en", META_PATH)


if __name__ == "__main__":
    main()
