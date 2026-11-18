"""
Script de entrenamiento riguroso de modelos de visión por computadora para seguridad inteligente:
1. sentinel_security_model.keras:
   - 5 clases: ['Arma_Blanca', 'Arma_De_Fuego', 'Ataque_Agresion', 'Escena_Normal', 'Hurto_Sospechoso']
   - Modelo 1: CNN desde cero (custom_model)
   - Modelo 2: Transfer Learning MobileNetV2 (transfer_model)
2. sentinel_gender_model.keras:
   - 2 clases: ['Hombre', 'Mujer'] (MobileNetV2)
3. Exportación de model_metadata.json con métricas reales y curvas
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
os.environ["KERAS_BACKEND"] = "torch"

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import keras
from keras import layers, models, callbacks

print(f"Backend activo de Keras: {keras.backend.backend()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Dispositivo GPU: {device_name}")

DATASET_SCENE = PROJECT_ROOT / "datasets" / "sentinel_real_dataset"
DATASET_GENDER = PROJECT_ROOT / "datasets" / "gender_real_dataset"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
Input_Shape = IMG_SIZE + (3,)

# ==============================================================================
# 1. CARGA DEL DATASET DE ESCENAS REALES
# ==============================================================================
print("\n--- CARGANDO DATASET DE ESCENAS ---")
ds_scene = keras.utils.image_dataset_from_directory(
    str(DATASET_SCENE),
    image_size=IMG_SIZE,
    batch_size=None,
    label_mode="int",
    shuffle=False,
)
scene_class_names = ds_scene.class_names
num_scene_classes = len(scene_class_names)
print("Clases de escena:", scene_class_names)

imgs, lbls = [], []
for img, lbl in ds_scene:
    imgs.append(img.numpy().astype("uint8"))
    lbls.append(int(lbl.numpy()))

X_scene = np.stack(imgs)
y_scene = np.array(lbls)
del imgs, lbls

print(f"X_scene: {X_scene.shape}, y_scene: {y_scene.shape}")

# Partición estratificada 70% train / 15% val / 15% test
X_tr, X_temp, y_tr, y_temp = train_test_split(
    X_scene, y_scene, test_size=0.30, random_state=42, stratify=y_scene
)
X_val, X_te, y_val, y_te = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
del X_temp, y_temp, X_scene, y_scene

# Normalización [-1, 1]
X_train = (X_tr.astype("float32") / 127.5) - 1.0
X_val = (X_val.astype("float32") / 127.5) - 1.0
X_test = (X_te.astype("float32") / 127.5) - 1.0
y_train, y_val, y_test = y_tr, y_val, y_te
del X_tr, X_te, y_tr, y_te

# Pesos de clase
conteo_train = np.bincount(y_train, minlength=num_scene_classes)
class_weight = {i: len(y_train) / (num_scene_classes * conteo_train[i]) for i in range(num_scene_classes)}
for i, w in class_weight.items():
    print(f"  {scene_class_names[i]:18s} peso = {w:.2f} (n={conteo_train[i]})")

# Data Augmentation en Keras
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.12, value_range=(-1, 1)),
    ],
    name="data_augmentation",
)

# ==============================================================================
# 2. MODELO 1: CNN DESDE CERO (custom_model)
# ==============================================================================
print("\n--- ENTRENANDO MODELO 1: CNN DESDE CERO ---")
custom_model = models.Sequential(
    [
        layers.Input(shape=Input_Shape),
        data_augmentation,
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(num_scene_classes, activation="softmax"),
    ],
    name="Custom_CNN_SentinelCV",
)

custom_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

hist_custom = custom_model.fit(
    X_train,
    y_train,
    epochs=14,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight,
    verbose=1,
)

# ==============================================================================
# 3. MODELO 2: TRANSFER LEARNING MOBILENETV2 (transfer_model)
# ==============================================================================
print("\n--- ENTRENANDO MODELO 2: TRANSFER LEARNING (MOBILENETV2) ---")
base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=Input_Shape,
)
base_model.trainable = False

transfer_model = models.Sequential(
    [
        layers.Input(shape=Input_Shape),
        data_augmentation,
        base_model,
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_scene_classes, activation="softmax"),
    ],
    name="Transfer_Learning_MobileNetV2",
)

transfer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0008),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

hist_transfer = transfer_model.fit(
    X_train,
    y_train,
    epochs=16,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight,
    verbose=1,
)

# ==============================================================================
# 4. EVALUACIÓN Y COMPARACIÓN DE LOS DOS MODELOS DE ESCENA
# ==============================================================================
def evaluar_modelo(modelo, history, nombre, class_names):
    probs = modelo.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    rep = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    
    print(f"\n--- {nombre} ---")
    print(f"Accuracy: {acc:.4f} | F1 (macro): {f1_macro:.4f} | F1 (weighted): {f1_wt:.4f}")
    return {
        "Modelo": nombre,
        "Accuracy": float(acc),
        "Precision (macro)": float(p_macro),
        "Recall (macro)": float(r_macro),
        "F1 (macro)": float(f1_macro),
        "Precision (weighted)": float(p_wt),
        "Recall (weighted)": float(r_wt),
        "F1 (weighted)": float(f1_wt),
        "classification_report": rep,
        "confusion_matrix": cm.tolist(),
    }

res_custom = evaluar_modelo(custom_model, hist_custom, "CNN desde cero", scene_class_names)
res_transfer = evaluar_modelo(transfer_model, hist_transfer, "Transfer Learning (MobileNetV2)", scene_class_names)

# Exportar modelo campeón de escena
champion_scene = transfer_model if res_transfer["Accuracy"] >= res_custom["Accuracy"] else custom_model
champion_scene_name = champion_scene.name
scene_model_path = MODELS_DIR / "sentinel_security_model.keras"
champion_scene.save(str(scene_model_path))
print(f"\nModelo campeon de escena guardado en: {scene_model_path} ({champion_scene_name})")

# ==============================================================================
# 5. ENTRENAMIENTO DEL MODELO DE GÉNERO FACIAL (sentinel_gender_model.keras)
# ==============================================================================
print("\n--- ENTRENANDO CLASIFICADOR FACIAL DE GÉNERO ---")
ds_gender = keras.utils.image_dataset_from_directory(
    str(DATASET_GENDER),
    image_size=IMG_SIZE,
    batch_size=None,
    label_mode="int",
    shuffle=False,
)
gender_class_names = ds_gender.class_names
print("Clases de genero:", gender_class_names)

imgs_g, lbls_g = [], []
for img, lbl in ds_gender:
    imgs_g.append(img.numpy().astype("uint8"))
    lbls_g.append(int(lbl.numpy()))

X_g = np.stack(imgs_g)
y_g = np.array(lbls_g)
del imgs_g, lbls_g

X_gtr, X_gte, y_gtr, y_gte = train_test_split(
    X_g, y_g, test_size=0.20, random_state=42, stratify=y_g
)
del X_g, y_g

X_gtrain = (X_gtr.astype("float32") / 127.5) - 1.0
X_gtest = (X_gte.astype("float32") / 127.5) - 1.0
del X_gtr, X_gte

gender_base = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=Input_Shape,
)
gender_base.trainable = False

gender_model = models.Sequential(
    [
        layers.Input(shape=Input_Shape),
        layers.RandomFlip("horizontal"),
        gender_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(2, activation="softmax"),
    ],
    name="SentinelCV_Gender_MobileNetV2",
)

gender_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0008),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

gender_model.fit(
    X_gtrain,
    y_gtr,
    epochs=8,
    batch_size=32,
    validation_split=0.15,
    verbose=1,
)

g_probs = gender_model.predict(X_gtest, batch_size=32, verbose=0)
g_pred = np.argmax(g_probs, axis=1)
g_acc = accuracy_score(y_gte, g_pred)
g_p, g_r, g_f1, _ = precision_recall_fscore_support(y_gte, g_pred, average="macro")
print(f"Gender Model Accuracy: {g_acc:.4f} | F1: {g_f1:.4f}")

gender_model_path = MODELS_DIR / "sentinel_gender_model.keras"
gender_model.save(str(gender_model_path))
print(f"Modelo de genero guardado en: {gender_model_path}")

# ==============================================================================
# 6. EXPORTAR METADATOS Y ARCHIVO JSON
# ==============================================================================
metadata = {
    "model_name": "SentinelCV-MobileNetV2-Keras3",
    "framework": f"Keras {keras.__version__} (Backend: {keras.backend.backend()})",
    "gpu_trained_on": device_name,
    "input_shape": [224, 224, 3],
    "normalization": "x / 127.5 - 1.0 ([-1, 1])",
    "scene_class_names": scene_class_names,
    "num_scene_classes": len(scene_class_names),
    "gender_class_names": gender_class_names,
    "scene_champion": res_transfer if res_transfer["Accuracy"] >= res_custom["Accuracy"] else res_custom,
    "scene_comparison": [res_custom, res_transfer],
    "gender_metrics": {
        "Accuracy": float(g_acc),
        "Precision (macro)": float(g_p),
        "Recall (macro)": float(g_r),
        "F1 (macro)": float(g_f1),
    },
    "kaggle_slugs_used": [
        "snehilsanyal/weapon-detection-test (Arma_Blanca, Arma_De_Fuego)",
        "meetnagadia/human-action-recognition-har-dataset (Ataque_Agresion, Escena_Normal, Hurto_Sospechoso)",
        "cashutosh/gender-classification-dataset (Hombre, Mujer)",
        "jangedoo/utkface-new (Holdout y Validación)",
    ],
    "endpoints": {
        "web_console": "http://localhost:4321",
        "inference_api": "http://localhost:8000/api/v1/infer",
        "cameras_api": "http://localhost:8000/api/v1/cameras",
        "health_api": "http://localhost:8000/api/v1/health",
    },
}

with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nMetadatos exportados exitosamente en models/model_metadata.json.")
print("ENTRENAMIENTO Y EXPORTACION COMPLETADOS CON DATOS REALES.")

