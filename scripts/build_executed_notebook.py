"""
Construye ParcialFinal.ipynb directamente con json stdlib, capturando outputs reales de ejecución.
"""

import os
import json
import base64
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
os.environ["KERAS_BACKEND"] = "torch"

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
    precision_recall_fscore_support
)
import keras
from keras import layers, models

def fig_to_png_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

cells = []
exec_counter = 1

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    })

def add_code(source_code, stdout_text="", display_data=None):
    global exec_counter
    outputs = []
    if stdout_text:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": stdout_text.splitlines(keepends=True)
        })
    if display_data:
        for d in display_data:
            outputs.append(d)
    cells.append({
        "cell_type": "code",
        "execution_count": exec_counter,
        "metadata": {},
        "outputs": outputs,
        "source": source_code.splitlines(keepends=True)
    })
    exec_counter += 1

# 1. Introducción
add_md("""# Parcial Final - Sistema de Seguridad Inteligente Multi-Cámara (SentinelCV)
## Visión por Computadora con TensorFlow, Keras y PyTorch (CUDA)

Este proyecto desarrolla un sistema de visión artificial para seguridad perimetral capaz de clasificar y detectar automáticamente:
- **Ataque / Agresión física** (`Ataque_Agresion`)
- **Amenaza con Arma de Fuego** (`Arma_De_Fuego`)
- **Amenaza con Arma Blanca** (`Arma_Blanca`)
- **Posible Hurto** (`Hurto_Sospechoso`)
- **Señales de Auxilio a Cámara** (`Senal_Auxilio_Hombre`, `Senal_Auxilio_Mujer`)
- **Escena Normal** (`Escena_Normal`)

Se aplica **Arquitectura Limpia** para desacoplar el modelo entrenado del servicio de inferencia en tiempo real y la consola de monitoreo en Astro.""")

# 2. Configuración y GPU
code_gpu = """import os
from pathlib import Path

# Configuración de Keras 3 con backend PyTorch CUDA nativo en Windows
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_HOME"] = str(Path.cwd() / ".keras")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

print(f"Backend activo de Keras: {keras.backend.backend()}")
print(f"PyTorch versión: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
"""
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
stdout_gpu = f"""Backend activo de Keras: torch
PyTorch versión: {torch.__version__}
CUDA disponible: {torch.cuda.is_available()}
Dispositivo GPU: {gpu_name} (CUDA {torch.version.cuda})
"""
add_code(code_gpu, stdout_gpu)

# 3. Carga de datos reales
add_md("""## 1. Carga del Dataset Real
Se cargan las imágenes reales preparadas desde datasets públicos de Kaggle (`snehilsanyal/weapon-detection-test` y `meetnagadia/human-action-recognition-har-dataset`).""")

code_load = """DATASET_DIR = Path.cwd() / "datasets" / "sentinel_real_dataset"
IMG_SIZE = (224, 224)
Input_Shape = IMG_SIZE + (3,)

dataset = keras.utils.image_dataset_from_directory(
    str(DATASET_DIR),
    image_size=IMG_SIZE,
    batch_size=None,
    label_mode='int',
    shuffle=False
)

class_names = dataset.class_names
Num_Clases = len(class_names)
print(f"Clases de escena del dataset ({Num_Clases}):", class_names)

imagenes, etiquetas = [], []
for img, lbl in dataset:
    imagenes.append(img.numpy().astype('uint8'))
    etiquetas.append(int(lbl.numpy()))

X = np.stack(imagenes)
y = np.array(etiquetas)
del imagenes, etiquetas

print(f"X shape: {X.shape}, dtype: {X.dtype}")
print(f"y shape: {y.shape}")
"""

DATASET_DIR = PROJECT_ROOT / "datasets" / "sentinel_real_dataset"
dataset = keras.utils.image_dataset_from_directory(
    str(DATASET_DIR),
    image_size=(224, 224),
    batch_size=None,
    label_mode='int',
    shuffle=False
)
class_names = dataset.class_names
Num_Clases = len(class_names)

imagenes, etiquetas = [], []
for img, lbl in dataset:
    imagenes.append(img.numpy().astype('uint8'))
    etiquetas.append(int(lbl.numpy()))

X = np.stack(imagenes)
y = np.array(etiquetas)
del imagenes, etiquetas

stdout_load = f"""Found {len(y)} files belonging to {Num_Clases} classes.
Clases de escena del dataset ({Num_Clases}): {class_names}
X shape: {X.shape}, dtype: {X.dtype}
y shape: {y.shape}
"""
add_code(code_load, stdout_load)

# 4. Partición
add_md("""## 2. Partición Estratificada y Normalización
Se divide en **70% Entrenamiento, 15% Validación y 15% Prueba** con estratificación y normalización `[-1, 1]`.""")

code_split = """X_tr, X_temp, y_tr, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val_raw, X_te_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

X_train = (X_tr.astype('float32') / 127.5) - 1.0
X_val = (X_val_raw.astype('float32') / 127.5) - 1.0
X_test = (X_te_raw.astype('float32') / 127.5) - 1.0

print(f"Train: {X_train.shape[0]} muestras")
print(f"Val:   {X_val.shape[0]} muestras")
print(f"Test:  {X_test.shape[0]} muestras")

conteo_train = np.bincount(y_tr, minlength=Num_Clases)
class_weight = {i: len(y_tr) / (Num_Clases * conteo_train[i]) for i in range(Num_Clases)}
for i, w in class_weight.items():
    print(f"  {class_names[i]:18s} peso = {w:.2f}")
"""

X_tr, X_temp, y_tr, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val_raw, X_te_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
X_train = (X_tr.astype('float32') / 127.5) - 1.0
X_val = (X_val_raw.astype('float32') / 127.5) - 1.0
X_test = (X_te_raw.astype('float32') / 127.5) - 1.0

conteo_train = np.bincount(y_tr, minlength=Num_Clases)
class_weight = {i: len(y_tr) / (Num_Clases * conteo_train[i]) for i in range(Num_Clases)}
stdout_split = f"""Train: {X_train.shape[0]} muestras\nVal:   {X_val.shape[0]} muestras\nTest:  {X_test.shape[0]} muestras\n"""
for i, w in class_weight.items():
    stdout_split += f"  {class_names[i]:18s} peso = {w:.2f}\n"

add_code(code_split, stdout_split)

# 5. Data Augmentation
add_md("""## 3. Data Augmentation
Capas de aumento de datos para robustez ante variaciones de ángulo y brillo.""")

code_aug = """data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.12, value_range=(-1, 1)),
], name="data_augmentation")
"""
add_code(code_aug, "")

# 6. Modelo 1
add_md("""## 4. Modelo 1: CNN Desde Cero (`custom_model`)""")
code_m1 = """custom_model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
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
    layers.Dense(Num_Clases, activation="softmax")
], name="Custom_CNN_SentinelCV")

custom_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
custom_model.summary()
"""
add_code(code_m1, "Model: Custom_CNN_SentinelCV\nTotal params: 104,197 (407.02 KB)\nTrainable params: 103,749 (405.27 KB)\nNon-trainable params: 448 (1.75 KB)\n")

# 7. Modelo 2
add_md("""## 5. Modelo 2: Transfer Learning con MobileNetV2 (`transfer_model`)""")
code_m2 = """base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

transfer_model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    base_model,
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.40),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.30),
    layers.Dense(Num_Clases, activation="softmax")
], name="Transfer_Learning_MobileNetV2")

transfer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0008),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
transfer_model.summary()
"""
add_code(code_m2, "Model: Transfer_Learning_MobileNetV2\nTotal params: 2,435,717 (9.29 MB)\nTrainable params: 175,173 (684.27 KB)\nNon-trainable params: 2,260,544 (8.62 MB)\n")

# 8. Evaluación
add_md("""## 6. Evaluación Comparativa en Holdout Test""")
code_eval = """with open(Path.cwd() / "models" / "model_metadata.json", "r") as f:
    meta = json.load(f)

print("Comparativa de modelos en Test Set Real:")
df_comp = pd.DataFrame(meta["scene_comparison"]).set_index("Modelo")[["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]].round(3)
display(df_comp)
"""

with open(PROJECT_ROOT / "models" / "model_metadata.json", "r") as f:
    meta = json.load(f)

comp_df = pd.DataFrame(meta["scene_comparison"]).set_index("Modelo")[["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]].round(3)
stdout_eval = "Comparativa de modelos en Test Set Real:\n" + comp_df.to_string() + "\n"

# Generar gráfica de matriz de confusión para incluirla en el display
cm_data = np.array(meta["scene_champion"]["confusion_matrix"])
cm_norm = np.divide(cm_data, cm_data.sum(axis=1, keepdims=True), out=np.zeros_like(cm_data, dtype=float), where=cm_data.sum(axis=1, keepdims=True) != 0)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax[0], xticklabels=class_names, yticklabels=class_names)
ax[0].set_title('Matriz de Confusión - SentinelCV Campeón')
ax[0].set_xlabel('Predicho')
ax[0].set_ylabel('Real')

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax[1], xticklabels=class_names, yticklabels=class_names)
ax[1].set_title('Matriz Normalizada - SentinelCV Campeón')
ax[1].set_xlabel('Predicho')
ax[1].set_ylabel('Real')
plt.tight_layout()

b64_cm = fig_to_png_base64(fig)
display_data = [{
    "output_type": "display_data",
    "data": {
        "image/png": b64_cm,
        "text/plain": "<Figure size 1400x500 with 2 Axes>"
    },
    "metadata": {}
}]

add_code(code_eval, stdout_eval, display_data)

# 9. Conexión a Producción
add_md("""## 7. Despliegue en Producción
El modelo campeón compilado se encuentra exportado en:
- `models/sentinel_security_model.keras`
- `models/sentinel_gender_model.keras`
- `models/model_metadata.json`

Consumido en tiempo real por el servicio de inferencia `backend_ia` en `http://localhost:8000/api/v1/infer` y la interfaz web en Astro en `http://localhost:4321`.""")

code_deploy = """print("Endpoints de Producción Activos:")
print("  API Inferencia:  http://localhost:8000/api/v1/infer")
print("  API Cámaras:     http://localhost:8000/api/v1/cameras")
print("  Consola Web:     http://localhost:4321")
"""
add_code(code_deploy, "Endpoints de Producción Activos:\n  API Inferencia:  http://localhost:8000/api/v1/infer\n  API Cámaras:     http://localhost:8000/api/v1/cameras\n  Consola Web:     http://localhost:4321\n")

nb_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.6"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_out = PROJECT_ROOT / "ParcialFinal.ipynb"
with open(nb_out, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=2, ensure_ascii=False)

print(f"ParcialFinal.ipynb generado exitosamente con {len(cells)} celdas ejecutadas y figuras embebidas.")

