"""
Genera y ejecuta ParcialFinal.ipynb con todas las celdas de código y markdown reales.
"""

import os
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["KERAS_HOME"] = str(PROJECT_ROOT / ".keras")
os.environ["KERAS_BACKEND"] = "torch"

nb = nbf.v4.new_notebook()

cells = []

# Celda 1 - Título
cells.append(nbf.v4.new_markdown_cell("""# Parcial Final - Sistema de Seguridad Inteligente Multi-Cámara (SentinelCV)
## Visión por Computadora con TensorFlow, Keras y PyTorch (CUDA)

Este proyecto desarrolla un sistema de visión artificial para seguridad perimetral capaz de detectar de forma automática:
- **Ataque / Agresión física** (`Ataque_Agresion`)
- **Amenaza con Arma de Fuego** (`Arma_De_Fuego`)
- **Amenaza con Arma Blanca** (`Arma_Blanca`)
- **Posible Hurto** (`Hurto_Sospechoso`)
- **Señales de Auxilio dirigidas a cámara por Hombre o Mujer** (`Senal_Auxilio_Hombre`, `Senal_Auxilio_Mujer`)
- **Escena Normal** (`Escena_Normal`)

Se utiliza una **Arquitectura Limpia** desacoplada entre el entrenamiento del modelo, el motor de inferencia backend y la plataforma de monitoreo en Astro con grilla de 6 cámaras."""))

# Celda 2 - Librerías y CUDA
cells.append(nbf.v4.new_code_cell("""import os
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
from keras import layers, models, callbacks
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
"""))

# Celda 3 - Carga de Datos Reales
cells.append(nbf.v4.new_markdown_cell("""## 1. Preparación y Carga de Datasets Reales
Los datos provienen de datasets públicos reales de seguridad y acciones:
- `snehilsanyal/weapon-detection-test`: Armas de fuego reales (`Arma_De_Fuego`) y armas blancas (`Arma_Blanca`).
- `meetnagadia/human-action-recognition-har-dataset`: Peleas/agresiones reales (`Ataque_Agresion`), actividades cotidianas (`Escena_Normal`) e interacción con objetos (`Hurto_Sospechoso`).
- `cashutosh/gender-classification-dataset` y `jangedoo/utkface-new`: Rostros reales para clasificación de género.
"""))

cells.append(nbf.v4.new_code_cell("""DATASET_DIR = Path.cwd() / "datasets" / "sentinel_real_dataset"
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
"""))

# Celda 4 - Partición
cells.append(nbf.v4.new_markdown_cell("""## 2. Partición Estratificada y Normalización
Se divide el conjunto en **70% Entrenamiento, 15% Validación y 15% Prueba** con estratificación para conservar la proporción de cada clase. La normalización se realiza en el rango `[-1, 1]`."""))

cells.append(nbf.v4.new_code_cell("""X_tr, X_temp, y_tr, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val_raw, X_te_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Escalado a [-1, 1]
X_train = (X_tr.astype('float32') / 127.5) - 1.0
X_val = (X_val_raw.astype('float32') / 127.5) - 1.0
X_test = (X_te_raw.astype('float32') / 127.5) - 1.0

print(f"Train: {X_train.shape[0]} muestras")
print(f"Val:   {X_val.shape[0]} muestras")
print(f"Test:  {X_test.shape[0]} muestras")

# Cálculo de pesos para mitigar desbalance
conteo_train = np.bincount(y_tr, minlength=Num_Clases)
class_weight = {i: len(y_tr) / (Num_Clases * conteo_train[i]) for i in range(Num_Clases)}
for i, w in class_weight.items():
    print(f"  {class_names[i]:18s} peso = {w:.2f}")
"""))

# Celda 5 - Data Augmentation
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Augmentation
Se definen transformaciones de rotación, zoom, volteo horizontal y contraste para robustecer el modelo frente a variaciones de iluminación y perspectiva de las cámaras de seguridad."""))

cells.append(nbf.v4.new_code_cell("""data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.12, value_range=(-1, 1)),
], name="data_augmentation")
"""))

# Celda 6 - Modelo 1 CNN
cells.append(nbf.v4.new_markdown_cell("""## 4. Modelo 1: CNN Diseñada Desde Cero (`custom_model`)
Arquitectura convolucional profunda con bloques Conv2D, Batch Normalization, Max Pooling y Dropout regularizador."""))

cells.append(nbf.v4.new_code_cell("""custom_model = models.Sequential([
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
    layers.Dense(Num_Clases, activation="softmax")
], name="Custom_CNN_SentinelCV")

custom_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
custom_model.summary()

history_custom = custom_model.fit(
    X_train, y_tr,
    epochs=14,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight,
    verbose=1
)
"""))

# Celda 7 - Modelo 2 Transfer Learning
cells.append(nbf.v4.new_markdown_cell("""## 5. Modelo 2: Transfer Learning con MobileNetV2 (`transfer_model`)
Uso del backbone preentrenado `MobileNetV2` sobre ImageNet con congelamiento de capas base y cabezal denso personalizado."""))

cells.append(nbf.v4.new_code_cell("""base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=Input_Shape
)
base_model.trainable = False

transfer_model = models.Sequential([
    layers.Input(shape=Input_Shape),
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

history_transfer = transfer_model.fit(
    X_train, y_tr,
    epochs=16,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight,
    verbose=1
)
"""))

# Celda 8 - Evaluación
cells.append(nbf.v4.new_markdown_cell("""## 6. Evaluación Rigurosa y Comparativa de los Modelos
Evaluación en el conjunto de prueba independiente (`X_test`), reporte de clasificación y matriz de confusión."""))

cells.append(nbf.v4.new_code_cell("""def evaluar_modelo(modelo, history, nombre):
    print("=" * 70)
    print(f"EVALUACIÓN: {nombre}")
    print("=" * 70)
    
    probs = modelo.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy en prueba: {acc:.4f}\\n")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3, zero_division=0))
    
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(Num_Clases))
    cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], xticklabels=class_names, yticklabels=class_names)
    ax[0].set_title(f'Matriz de Confusión - {nombre}')
    ax[0].set_xlabel('Predicho')
    ax[0].set_ylabel('Real')
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax[1], xticklabels=class_names, yticklabels=class_names)
    ax[1].set_title(f'Matriz Normalizada - {nombre}')
    ax[1].set_xlabel('Predicho')
    ax[1].set_ylabel('Real')
    plt.tight_layout()
    plt.show()
    
    return {
        "Modelo": nombre,
        "Accuracy": acc,
        "Precision (macro)": p_m,
        "Recall (macro)": r_m,
        "F1 (macro)": f1_m,
        "Precision (weighted)": p_w,
        "Recall (weighted)": r_w,
        "F1 (weighted)": f1_w,
    }

res_custom = evaluar_modelo(custom_model, history_custom, "CNN desde cero")
res_transfer = evaluar_modelo(transfer_model, history_transfer, "Transfer Learning (MobileNetV2)")

df_comparativo = pd.DataFrame([res_custom, res_transfer]).set_index("Modelo").round(3)
print("\\nTABLA COMPARATIVA FINAL")
display(df_comparativo)
"""))

# Celda 9 - Exportación
cells.append(nbf.v4.new_markdown_cell("""## 7. Exportación del Modelo Compilado a Producción
Se guarda el modelo ganador en formato `.keras` para ser consumido por el servidor backend de inferencia."""))

cells.append(nbf.v4.new_code_cell("""OUTPUT_DIR = Path.cwd() / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

campeon = transfer_model if res_transfer["Accuracy"] >= res_custom["Accuracy"] else custom_model
modelo_path = OUTPUT_DIR / "sentinel_security_model.keras"
campeon.save(str(modelo_path))
print(f"Modelo campeón compilado guardado en: {modelo_path}")
"""))

nb.cells = cells

# Guardar notebook
output_nb_path = PROJECT_ROOT / "ParcialFinal.ipynb"
with open(output_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook estructurado guardado en: {output_nb_path}")
print("Ejecutando notebook con nbclient...")

client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

with open(output_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook ParcialFinal.ipynb ejecutado exitosamente con todas las celdas y outputs.")

