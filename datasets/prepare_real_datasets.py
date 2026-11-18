"""
Preparador y curador de datasets reales de seguridad y género.
Toma los datasets descargados desde kagglehub:
1. snehilsanyal/weapon-detection-test (Arma_Blanca, Arma_De_Fuego)
2. meetnagadia/human-action-recognition-har-dataset (Ataque_Agresion, Escena_Normal, Hurto_Sospechoso)
3. cashutosh/gender-classification-dataset (Hombre, Mujer)

Genera:
- datasets/sentinel_real_dataset/ (train/val para ParcialFinal.ipynb)
- datasets/holdout/ (conjunto no visto para auditoría y certificación QA)
- datasets/gender_real_dataset/ (train/val de rostros reales)
"""

import os
import shutil
import random
from pathlib import Path
import pandas as pd
from PIL import Image

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_CACHE = Path(os.environ.get("USERPROFILE", "C:/Users/Carlos")) / ".cache" / "kagglehub" / "datasets"

WEAPON_DIR = KAGGLE_CACHE / "snehilsanyal" / "weapon-detection-test" / "versions" / "5"
HAR_DIR = KAGGLE_CACHE / "meetnagadia" / "human-action-recognition-har-dataset" / "versions" / "1" / "Human Action Recognition"
GENDER_DIR = KAGGLE_CACHE / "cashutosh" / "gender-classification-dataset" / "versions" / "1"

DEST_SCENE = PROJECT_ROOT / "datasets" / "sentinel_real_dataset"
DEST_HOLDOUT = PROJECT_ROOT / "datasets" / "holdout"
DEST_GENDER = PROJECT_ROOT / "datasets" / "gender_real_dataset"

for d in [DEST_SCENE, DEST_HOLDOUT, DEST_GENDER]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

SCENE_CLASSES = [
    "Arma_Blanca",
    "Arma_De_Fuego",
    "Ataque_Agresion",
    "Escena_Normal",
    "Hurto_Sospechoso",
]

for cls_name in SCENE_CLASSES:
    (DEST_SCENE / cls_name).mkdir(parents=True, exist_ok=True)
    (DEST_HOLDOUT / cls_name).mkdir(parents=True, exist_ok=True)

for g in ["Hombre", "Mujer"]:
    (DEST_GENDER / g).mkdir(parents=True, exist_ok=True)
    (DEST_HOLDOUT / f"Genero_{g}").mkdir(parents=True, exist_ok=True)

print("1. Procesando Armas (Arma_Blanca y Arma_De_Fuego)...")
meta_weapon = pd.read_csv(WEAPON_DIR / "metadata.csv")
meta_weapon["cat"] = meta_weapon["imagefile"].apply(lambda x: str(x).split("_")[0])

weapon_images_dir = WEAPON_DIR / "weapon_detection" / "train" / "images"
if not weapon_images_dir.exists():
    # buscar recursivamente
    for p in WEAPON_DIR.rglob("*.jpeg"):
        weapon_images_dir = p.parent
        break

arma_blanca_files = []
arma_fuego_files = []

for _, row in meta_weapon.iterrows():
    img_name = row["imagefile"]
    cat = row["cat"]
    # Buscar archivo
    found = list(WEAPON_DIR.rglob(img_name))
    if found:
        fpath = found[0]
        if cat in ["Knife", "Sword"]:
            arma_blanca_files.append(fpath)
        elif cat in ["Handgun", "Automatic Rifle", "Shotgun", "SMG", "Sniper"]:
            arma_fuego_files.append(fpath)

print(f"  Encontradas {len(arma_blanca_files)} imágenes de Arma_Blanca y {len(arma_fuego_files)} de Arma_De_Fuego.")

# Distribuir Arma_Blanca
random.shuffle(arma_blanca_files)
n_holdout = max(25, int(len(arma_blanca_files) * 0.2))
for f in arma_blanca_files[:n_holdout]:
    shutil.copy2(f, DEST_HOLDOUT / "Arma_Blanca" / f.name)
for f in arma_blanca_files[n_holdout:]:
    shutil.copy2(f, DEST_SCENE / "Arma_Blanca" / f.name)

# Distribuir Arma_De_Fuego
random.shuffle(arma_fuego_files)
n_holdout = max(35, int(len(arma_fuego_files) * 0.15))
for f in arma_fuego_files[:n_holdout]:
    shutil.copy2(f, DEST_HOLDOUT / "Arma_De_Fuego" / f.name)
for f in arma_fuego_files[n_holdout:]:
    shutil.copy2(f, DEST_SCENE / "Arma_De_Fuego" / f.name)

print("2. Procesando HAR (Ataque_Agresion, Escena_Normal, Hurto_Sospechoso)...")
har_meta = pd.read_csv(HAR_DIR / "Training_set.csv")
har_train_dir = HAR_DIR / "train"

fighting_files = []
normal_files = []
hurto_files = []

for _, row in har_meta.iterrows():
    fpath = har_train_dir / row["filename"]
    if not fpath.exists():
        continue
    lbl = row["label"]
    if lbl == "fighting":
        fighting_files.append(fpath)
    elif lbl in ["sitting", "using_laptop", "sleeping", "drinking"]:
        normal_files.append(fpath)
    elif lbl in ["calling", "texting", "clapping", "eating"]:
        hurto_files.append(fpath)

print(f"  Encontradas {len(fighting_files)} fighting, {len(normal_files)} normal, {len(hurto_files)} hurto/interacción.")

# Ataque_Agresion
random.shuffle(fighting_files)
n_hold = 40
for f in fighting_files[:n_hold]:
    shutil.copy2(f, DEST_HOLDOUT / "Ataque_Agresion" / f.name)
for f in fighting_files[n_hold:400]:
    shutil.copy2(f, DEST_SCENE / "Ataque_Agresion" / f.name)

# Escena_Normal
random.shuffle(normal_files)
for f in normal_files[:n_hold]:
    shutil.copy2(f, DEST_HOLDOUT / "Escena_Normal" / f.name)
for f in normal_files[n_hold:400]:
    shutil.copy2(f, DEST_SCENE / "Escena_Normal" / f.name)

# Hurto_Sospechoso
random.shuffle(hurto_files)
for f in hurto_files[:n_hold]:
    shutil.copy2(f, DEST_HOLDOUT / "Hurto_Sospechoso" / f.name)
for f in hurto_files[n_hold:400]:
    shutil.copy2(f, DEST_SCENE / "Hurto_Sospechoso" / f.name)

print("3. Procesando Género Facial (cashutosh/gender-classification-dataset)...")
# El dataset de cashutosh tiene Training/male, Training/female, Validation/male, Validation/female
male_train = list((GENDER_DIR / "Training" / "male").glob("*.jpg"))
female_train = list((GENDER_DIR / "Training" / "female").glob("*.jpg"))
male_val = list((GENDER_DIR / "Validation" / "male").glob("*.jpg"))
female_val = list((GENDER_DIR / "Validation" / "female").glob("*.jpg"))

print(f"  Encontradas {len(male_train)} male train, {len(female_train)} female train.")

# Holdout de género: usar validation
for f in male_val[:50]:
    shutil.copy2(f, DEST_HOLDOUT / "Genero_Hombre" / f.name)
for f in female_val[:50]:
    shutil.copy2(f, DEST_HOLDOUT / "Genero_Mujer" / f.name)

# Dataset de entrenamiento: tomar 500 por clase para entrenamiento veloz y preciso
for f in male_train[:500]:
    shutil.copy2(f, DEST_GENDER / "Hombre" / f.name)
for f in female_train[:500]:
    shutil.copy2(f, DEST_GENDER / "Mujer" / f.name)

print("\n--- RESUMEN DE DATASETS REALES PREPARADOS ---")
print("DEST_SCENE:")
for sub in DEST_SCENE.iterdir():
    print(f"  {sub.name}: {len(list(sub.glob('*.*')))} imágenes")

print("DEST_HOLDOUT:")
for sub in DEST_HOLDOUT.iterdir():
    print(f"  {sub.name}: {len(list(sub.glob('*.*')))} imágenes")

print("DEST_GENDER:")
for sub in DEST_GENDER.iterdir():
    print(f"  {sub.name}: {len(list(sub.glob('*.*')))} imágenes")

print("Datasets reales preparados con éxito sin fuga de datos.")

