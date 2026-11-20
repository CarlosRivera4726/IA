"""
Reentrena models/sentinel_gender_model.keras con mas datos (cashutosh/gender-classification-dataset)
y fine-tuning de MobileNetV2. Mismas clases/orden: ['Hombre', 'Mujer'], entrada 224x224, x/127.5-1.
Uso:  python scripts/train_gender_model.py [N_POR_CLASE]
"""
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
from keras import layers
from PIL import Image

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
N_VAL = 1000
SRC = Path.home() / ".cache" / "kagglehub" / "datasets" / "cashutosh" / "gender-classification-dataset" / "versions" / "1"
OUT = PROJECT_ROOT / "models" / "sentinel_gender_model.keras"
random.seed(42)
np.random.seed(42)


def load(split, n):
    X, y = [], []
    for label, folder in enumerate(["male", "female"]):  # 0=Hombre, 1=Mujer
        files = sorted((SRC / split / folder).glob("*.jpg"))
        random.shuffle(files)
        for f in files[:n]:
            try:
                im = Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR)
            except Exception:
                continue
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(label)
    X = np.stack(X)
    y = np.array(y)
    p = np.random.permutation(len(y))
    return X[p], y[p]


print("Cargando datos...")
X_tr, y_tr = load("Training", N_TRAIN)
X_va, y_va = load("Validation", N_VAL)
print(X_tr.shape, np.bincount(y_tr), X_va.shape, np.bincount(y_va))


class Seq(keras.utils.PyDataset):
    def __init__(self, X, y, bs=32, **kw):
        super().__init__(**kw)
        self.X, self.y, self.bs = X, y, bs

    def __len__(self):
        return int(np.ceil(len(self.y) / self.bs))

    def __getitem__(self, i):
        s = slice(i * self.bs, (i + 1) * self.bs)
        return self.X[s].astype("float32") / 127.5 - 1.0, self.y[s]


# NOTA: BatchNorm y los pesos ImageNet se comportan mal con backend torch en este entorno
# (inferencia colapsa a una sola clase). Se usa CNN propia con GroupNorm (sin estado train/inferencia).
def block(x, f):
    x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Activation("relu")(x)
    return layers.MaxPooling2D()(x)


inp = layers.Input(shape=(224, 224, 3))
x = layers.RandomFlip("horizontal")(inp)
x = layers.RandomRotation(0.06)(x)
x = layers.RandomZoom(0.1)(x)
x = layers.RandomBrightness(0.2, value_range=(-1.0, 1.0))(x)
x = layers.Resizing(112, 112)(x)
for f in (32, 64, 96, 128, 192):
    x = block(x, f)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inp, out, name="SentinelCV_Gender_CNN")
tr, va = Seq(X_tr, y_tr), Seq(X_va, y_va)
EPOCHS = 16
steps = EPOCHS * len(tr)
lr = keras.optimizers.schedules.CosineDecay(1e-3, steps)
model.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
best = keras.callbacks.ModelCheckpoint(str(OUT), monitor="val_accuracy", save_best_only=True, verbose=1)
model.fit(tr, validation_data=va, epochs=EPOCHS, verbose=2, callbacks=[best])
model = keras.models.load_model(str(OUT), compile=False)
probs = model.predict(va, verbose=0)
pred = probs.argmax(1)
print("Val accuracy:", float((pred == y_va).mean()))
for c, n in enumerate(["Hombre", "Mujer"]):
    print(f"  recall {n}: {float((pred[y_va == c] == c).mean()):.3f}")

model.save(str(OUT))
print("Guardado en", OUT)



