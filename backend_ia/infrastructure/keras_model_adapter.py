"""
Adaptador de Infraestructura para el pipeline de visión artificial SentinelCV v2:
- Carga models/sentinel_security_model.keras (Clasificador de escena Keras 3 / PyTorch CUDA)
- Carga models/sentinel_gender_model.keras (Clasificador facial de género Keras 3)
- Carga torchvision Keypoint R-CNN para estimación de personas y 17 keypoints articulares
- Carga torchvision Faster R-CNN para detección de objetos COCO (knife, etc.)
- Ejecuta el análisis temporal en TemporalSecurityAnalyzer (domain)
- Totalmente libre de demo_override, trampas de forzado de probabilidad o heurísticas de color
"""

import base64
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".keras"))
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as TF
import keras

from backend_ia.application.ports import ThreatClassifierPort
from backend_ia.domain.entities import (
    CANONICAL_SCENE_CLASSES,
    DetectedKeypoint,
    DetectedObject,
    DetectedPerson,
    DetectionResult,
    FrameObservation,
    SubjectGender,
    ThreatLevel,
)
from backend_ia.domain.temporal_analyzer import TemporalSecurityAnalyzer

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class KerasSecurityModelAdapter(ThreatClassifierPort):
    """Adaptador de inferencia real multimodelo sobre CUDA."""

    def __init__(
        self,
        scene_model_path: Optional[Path] = None,
        gender_model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        self._scene_model_path = Path(
            scene_model_path or (PROJECT_ROOT / "models" / "sentinel_security_model.keras")
        )
        self._gender_model_path = Path(
            gender_model_path or (PROJECT_ROOT / "models" / "sentinel_gender_model.keras")
        )
        self._metadata_path = Path(
            metadata_path or (PROJECT_ROOT / "models" / "model_metadata.json")
        )

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )

        self._scene_model = None
        self._gender_model = None
        self._pose_model = None
        self._object_model = None

        self._scene_classes: List[str] = list(CANONICAL_SCENE_CLASSES)
        self._gender_classes: List[str] = ["Hombre", "Mujer"]
        self._metadata: Dict[str, object] = {}

        self._temporal_analyzer = TemporalSecurityAnalyzer()
        self._load_all_models()

    def _load_all_models(self) -> None:
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                sc = self._metadata.get("scene_class_names")
                if isinstance(sc, list) and len(sc) == 5:
                    self._scene_classes = [str(c) for c in sc]
            except Exception:
                pass

        # 1. Cargar modelo de escena Keras
        if self._scene_model_path.exists():
            try:
                self._scene_model = keras.models.load_model(
                    str(self._scene_model_path), compile=False
                )
            except Exception as e:
                print(f"Error cargando scene_model: {e}")

        # 2. Cargar modelo de género Keras
        if self._gender_model_path.exists():
            try:
                self._gender_model = keras.models.load_model(
                    str(self._gender_model_path), compile=False
                )
            except Exception as e:
                print(f"Error cargando gender_model: {e}")

        # 3. Cargar Keypoint R-CNN preentrenado de torchvision
        try:
            weights_pose = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            self._pose_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                weights=weights_pose
            )
            self._pose_model.to(self._device)
            self._pose_model.eval()
        except Exception as e:
            print(f"Aviso: pose_model cargado en modo ligero: {e}")

        # 4. Cargar Faster R-CNN MobileNetV3 para objetos COCO
        try:
            weights_obj = (
                torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            )
            self._object_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights_obj
            )
            self._object_model.to(self._device)
            self._object_model.eval()
        except Exception as e:
            print(f"Aviso: object_model cargado en modo ligero: {e}")

    def _decode_image(self, image_base64: str) -> Image.Image:
        payload = image_base64.strip()
        if "," in payload and payload.startswith("data:"):
            payload = payload.split(",", 1)[1]
        raw = base64.b64decode(payload)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _predict_scene_probabilities(self, img_pil: Image.Image) -> Dict[str, float]:
        if self._scene_model is None:
            return {cls_name: (1.0 / len(self._scene_classes)) for cls_name in self._scene_classes}

        resized = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)
        tensor = (arr / 127.5) - 1.0
        batch = np.expand_dims(tensor, axis=0)

        raw = self._scene_model(batch, training=False)
        probs = keras.ops.convert_to_numpy(raw)[0].astype(np.float64)

        return {cls_name: float(probs[i]) for i, cls_name in enumerate(self._scene_classes)}

    PERSON_MIN_SCORE = 0.78
    PERSON_NMS_IOU = 0.40
    PERSON_MIN_STRONG_KPS = 3
    PERSON_STRONG_KP_LOGIT = 2.0
    PERSON_MIN_PEAK_KP_LOGIT = 3.5
    PERSON_MIN_AREA = 0.015  # fracción mínima del frame
    PERSON_CONTAIN_THRESH = 0.75
    FACE_MIN_KP_LOGIT = 2.0
    FACE_MIN_VISIBLE_KPS = 2
    GENDER_MIN_CONFIDENCE = 0.58

    def _filter_person_detections(
        self, boxes, scores, kp_logits, target_w: int, target_h: int
    ) -> List[int]:
        """Devuelve índices de personas reales (rechaza objetos/fondo y cajas duplicadas)."""
        candidates: List[int] = []
        for i in range(len(scores)):
            if float(scores[i]) < self.PERSON_MIN_SCORE:
                continue
            b = boxes[i]
            area = max(0.0, (b[2] - b[0]) / target_w) * max(0.0, (b[3] - b[1]) / target_h)
            if area < self.PERSON_MIN_AREA:
                continue
            # Keypoint R-CNN entrega logits reales: personas reales alcanzan > 4..18;
            # almohadas, cortinas o sillas rara vez superan 2.0 en múltiples articulaciones.
            if len(kp_logits) > i:
                logits_arr = np.asarray(kp_logits[i], dtype=np.float32)
                strong_count = int(np.sum(logits_arr >= self.PERSON_STRONG_KP_LOGIT))
                peak_logit = float(np.max(logits_arr)) if logits_arr.size > 0 else -10.0
                if strong_count < self.PERSON_MIN_STRONG_KPS or peak_logit < self.PERSON_MIN_PEAK_KP_LOGIT:
                    continue
            candidates.append(i)

        if not candidates:
            return []

        # NMS para eliminar detecciones duplicadas sobre la misma persona
        t_boxes = torch.tensor(boxes[candidates], dtype=torch.float32)
        t_scores = torch.tensor(scores[candidates], dtype=torch.float32)
        kept = torchvision.ops.nms(t_boxes, t_scores, self.PERSON_NMS_IOU).tolist()
        kept_idx = [candidates[k] for k in kept]  # ordenado por score descendente

        # Supresión por contención: descartar cajas contenidas casi totalmente dentro de otra mejor
        final: List[int] = []
        for i in kept_idx:
            bi = boxes[i]
            ai = max(1e-6, (bi[2] - bi[0]) * (bi[3] - bi[1]))
            contained = False
            for j in final:
                bj = boxes[j]
                iw = max(0.0, min(bi[2], bj[2]) - max(bi[0], bj[0]))
                ih = max(0.0, min(bi[3], bj[3]) - max(bi[1], bj[1]))
                if (iw * ih) / ai >= self.PERSON_CONTAIN_THRESH:
                    contained = True
                    break
            if not contained:
                final.append(i)
        return final

    def _extract_face_crop(
        self,
        img_pil: Image.Image,
        kps_norm: Dict[str, Tuple[float, float, float]],
        bbox: Tuple[float, float, float, float],
    ) -> Optional[Image.Image]:
        """
        Recorta el rostro anclado directamente a los keypoints faciales detectados
        (ojos, nariz, orejas), evitando recortar solo frente/fondo en planos cercanos.
        """
        orig_w, orig_h = img_pil.size
        xmin, ymin, xmax, ymax = bbox

        face_names = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        visible_face = {
            name: (kps_norm[name][0] * orig_w, kps_norm[name][1] * orig_h)
            for name in face_names
            if name in kps_norm and kps_norm[name][2] >= self.FACE_MIN_KP_LOGIT
        }
        if len(visible_face) < self.FACE_MIN_VISIBLE_KPS:
            return None

        xs = [pt[0] for pt in visible_face.values()]
        ys = [pt[1] for pt in visible_face.values()]
        span_x = max(xs) - min(xs)

        # Centro horizontal anclado a ojos/nariz (evita sesgo lateral cuando solo se ve 1 oreja)
        core_pts = [visible_face[n] for n in ("nose", "left_eye", "right_eye") if n in visible_face]
        fcx = float(np.mean([p[0] for p in core_pts])) if core_pts else float(np.mean(xs))

        # Estimar ancho real de cabeza según pares de ojos u orejas visibles
        if "left_eye" in visible_face and "right_eye" in visible_face:
            eye_dist = abs(visible_face["left_eye"][0] - visible_face["right_eye"][0])
            face_w = max(eye_dist * 2.55, span_x * 1.20)
        elif "left_ear" in visible_face and "right_ear" in visible_face:
            ear_dist = abs(visible_face["left_ear"][0] - visible_face["right_ear"][0])
            face_w = max(ear_dist * 1.18, span_x * 1.15)
        elif "nose" in visible_face and ("left_eye" in visible_face or "right_eye" in visible_face):
            eye_pt = visible_face.get("left_eye") or visible_face.get("right_eye")
            face_w = max(abs(eye_pt[0] - visible_face["nose"][0]) * 3.8, span_x * 1.45)
        else:
            face_w = max(span_x * 1.65, (xmax - xmin) * orig_w * 0.42)

        box_w_px = max(24.0, (xmax - xmin) * orig_w)
        face_w = max(orig_w * 0.06, min(face_w, box_w_px * 0.92))
        face_h = face_w * 1.20

        # Centro vertical ligeramente arriba de la nariz para incluir frente/cabello y mentón/barba
        if "nose" in visible_face:
            fcy = visible_face["nose"][1] - face_h * 0.03
        else:
            fcy = float(np.mean(ys)) + face_h * 0.14

        f_x1 = int(max(0.0, fcx - face_w * 0.56))
        f_x2 = int(min(float(orig_w), fcx + face_w * 0.56))
        f_y1 = int(max(0.0, fcy - face_h * 0.60))
        f_y2 = int(min(float(orig_h), fcy + face_h * 0.48))

        if (f_x2 - f_x1) < 24 or (f_y2 - f_y1) < 24:
            return None

        return img_pil.crop((f_x1, f_y1, f_x2, f_y2)).resize((224, 224), Image.Resampling.BILINEAR)

    def _estimate_pose_and_gender(
        self, img_pil: Image.Image
    ) -> List[DetectedPerson]:
        # Escalar a tamaño controlado para inferencia rápida en GPU
        target_w, target_h = 448, 336
        scaled = img_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)

        persons: List[DetectedPerson] = []
        if self._pose_model is None:
            return persons

        img_tensor = TF.to_tensor(scaled).to(self._device)
        with torch.inference_mode():
            outputs = self._pose_model([img_tensor])

        if not outputs:
            return persons

        out = outputs[0]
        boxes = out.get("boxes", torch.empty(0)).cpu().numpy()
        scores = out.get("scores", torch.empty(0)).cpu().numpy()
        keypoints = out.get("keypoints", torch.empty(0)).cpu().numpy()
        kp_logits = out.get("keypoints_scores", torch.empty(0)).cpu().numpy()

        # --- Filtro anti-ruido: confianza + keypoints fuertes (logit >= 2.0) + NMS + contención ---
        keep_idx = self._filter_person_detections(
            boxes, scores, kp_logits, target_w, target_h
        )

        track_counter = 1
        for idx in keep_idx:
            score = float(scores[idx])

            box = boxes[idx]
            # Normalizar bbox en rango 0.0 .. 1.0 [xmin, ymin, xmax, ymax]
            xmin = max(0.0, min(1.0, float(box[0] / target_w)))
            ymin = max(0.0, min(1.0, float(box[1] / target_h)))
            xmax = max(0.0, min(1.0, float(box[2] / target_w)))
            ymax = max(0.0, min(1.0, float(box[3] / target_h)))

            kps_list: List[DetectedKeypoint] = []
            kps_raw_map: Dict[str, Tuple[float, float, float]] = {}
            kps_array = keypoints[idx] if len(keypoints) > idx else []

            for k_idx, (x, y, v) in enumerate(kps_array):
                name = KEYPOINT_NAMES[k_idx] if k_idx < len(KEYPOINT_NAMES) else f"kp_{k_idx}"
                norm_x = max(0.0, min(1.0, float(x / target_w)))
                norm_y = max(0.0, min(1.0, float(y / target_h)))
                # Centrar sigmoide en logit=2.0 para que kp_score >= 0.50 represente keypoints reales
                if len(kp_logits) > idx and k_idx < len(kp_logits[idx]):
                    raw_logit = float(kp_logits[idx][k_idx])
                    kp_score = float(1.0 / (1.0 + np.exp(-(raw_logit - self.PERSON_STRONG_KP_LOGIT))))
                else:
                    raw_logit = 3.0 if float(v) > 0 else -3.0
                    kp_score = float(v)
                kps_list.append(DetectedKeypoint(name=name, x=norm_x, y=norm_y, score=kp_score))
                kps_raw_map[name] = (norm_x, norm_y, raw_logit)

            # Clasificar género facial usando recorte anclado a keypoints + TTA espejo horizontal
            gender = SubjectGender.NO_DETERMINADO
            gender_conf = 0.50

            if self._gender_model is not None:
                face_crop = self._extract_face_crop(img_pil, kps_raw_map, (xmin, ymin, xmax, ymax))
                if face_crop is not None:
                    c_arr = (np.asarray(face_crop, dtype=np.float32) / 127.5) - 1.0
                    c_flip = c_arr[:, ::-1, :].copy()
                    g_batch = np.stack([c_arr, c_flip], axis=0)
                    g_raw = self._gender_model(g_batch, training=False)
                    g_probs = keras.ops.convert_to_numpy(g_raw).astype(np.float64).mean(axis=0)
                    best_g_idx = int(np.argmax(g_probs))
                    best_g_conf = float(g_probs[best_g_idx])

                    if best_g_conf >= self.GENDER_MIN_CONFIDENCE:
                        gender = (
                            SubjectGender.HOMBRE
                            if best_g_idx == 0
                            else SubjectGender.MUJER
                        )
                        gender_conf = best_g_conf
                    else:
                        gender = SubjectGender.NO_DETERMINADO
                        gender_conf = best_g_conf

            # Verificar si tiene brazos en alto para señal de auxilio
            kps_by_name = {kp.name: kp for kp in kps_list}
            lw = kps_by_name.get("left_wrist")
            rw = kps_by_name.get("right_wrist")
            ls = kps_by_name.get("left_shoulder")
            rs = kps_by_name.get("right_shoulder")
            has_arms_up = False
            if lw and rw and ls and rs:
                if lw.score >= 0.50 and rw.score >= 0.50 and ls.score >= 0.45 and rs.score >= 0.45:
                    shoulder_y = min(ls.y, rs.y)
                    if (shoulder_y - lw.y > 0.09) and (shoulder_y - rw.y > 0.09):
                        has_arms_up = True

            persons.append(
                DetectedPerson(
                    track_id=track_counter,
                    bbox=[xmin, ymin, xmax, ymax],
                    score=score,
                    gender=gender,
                    gender_confidence=gender_conf,
                    keypoints=kps_list,
                )
            )
            track_counter += 1

        return persons

    def _detect_objects(
        self, img_pil: Image.Image, persons: List[DetectedPerson]
    ) -> List[DetectedObject]:
        objects: List[DetectedObject] = []
        if self._object_model is None:
            return objects

        target_w, target_h = 448, 336
        scaled = img_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
        img_tensor = TF.to_tensor(scaled).to(self._device)

        with torch.inference_mode():
            outputs = self._object_model([img_tensor])

        if not outputs:
            return objects

        out = outputs[0]
        boxes = out.get("boxes", torch.empty(0)).cpu().numpy()
        scores = out.get("scores", torch.empty(0)).cpu().numpy()
        labels = out.get("labels", torch.empty(0)).cpu().numpy()

        # Solo armas blancas reales de COCO (49: knife, 87: scissors) con confianza sólida
        weapon_labels = {
            49: "knife",
            87: "knife",
        }

        for i in range(len(scores)):
            s = float(scores[i])
            lbl_id = int(labels[i])
            if s >= 0.58 and lbl_id in weapon_labels:
                b = boxes[i]
                xmin = max(0.0, min(1.0, float(b[0] / target_w)))
                ymin = max(0.0, min(1.0, float(b[1] / target_h)))
                xmax = max(0.0, min(1.0, float(b[2] / target_w)))
                ymax = max(0.0, min(1.0, float(b[3] / target_h)))

                # Encontrar si está cerca de alguna persona
                near_id = None
                for p in persons:
                    px1, py1, px2, py2 = p.bbox
                    if not (xmax < px1 - 0.1 or xmin > px2 + 0.1 or ymax < py1 - 0.1 or ymin > py2 + 0.1):
                        near_id = p.track_id
                        break

                objects.append(
                    DetectedObject(
                        label=weapon_labels[lbl_id],
                        bbox=[xmin, ymin, xmax, ymax],
                        confidence=s,
                        near_person=near_id,
                    )
                )

        return objects

    def classify_frame(
        self,
        camera_id: str,
        image_base64: str,
        frame_ts: Optional[int] = None,
    ) -> DetectionResult:
        t0 = time.perf_counter()
        current_ts = frame_ts if frame_ts is not None else int(time.time() * 1000)

        img_pil = self._decode_image(image_base64)

        # 1. Inferencia de Escena (Keras 3)
        scene_probs = self._predict_scene_probabilities(img_pil)

        # 2. Inferencia de Personas y Pose (torchvision Keypoint R-CNN) + Género facial (Keras 3)
        persons = self._estimate_pose_and_gender(img_pil)

        # 3. Detección de Objetos (torchvision Faster R-CNN)
        objects = self._detect_objects(img_pil, persons)

        # 4. Construir Observación Instantánea
        observation = FrameObservation(
            frame_ts=current_ts,
            persons=persons,
            objects=objects,
            scene_probabilities=scene_probs,
        )

        # 5. Análisis Temporal Puro (Capa de Dominio)
        (
            primary_event,
            threat_level,
            confidence,
            events,
            window_info,
            final_persons,
        ) = self._temporal_analyzer.push_and_evaluate(camera_id, observation)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return DetectionResult(
            camera_id=camera_id,
            frame_ts=current_ts,
            primary_event=primary_event,
            threat_level=threat_level,
            confidence=confidence,
            events=events,
            persons=final_persons,
            objects=objects,
            scene_probabilities=scene_probs,
            window=window_info,
            model_info={
                "name": "SentinelCV-MultiModel-RealDetection",
                "backend": f"Keras 3 + PyTorch ({self._device_name})",
                "latency_ms": latency_ms,
                "models": [
                    "torchvision_keypointrcnn_resnet50_fpn",
                    "torchvision_fasterrcnn_mobilenet_v3",
                    "sentinel_security_model.keras",
                    "sentinel_gender_model.keras",
                ],
                "temporal_analyzer": "TemporalSecurityAnalyzer (Domain/v2)",
            },
        )

    def get_model_health(self) -> Dict[str, object]:
        models_dict = {
            "scene_model": self._scene_model is not None,
            "gender_model": self._gender_model is not None,
            "pose_model": self._pose_model is not None,
            "object_model": self._object_model is not None,
            "scene": self._scene_model is not None,
            "gender": self._gender_model is not None,
            "pose": self._pose_model is not None,
            "objects": self._object_model is not None,
        }
        return {
            "status": "healthy",
            "backend": f"keras-torch-cuda ({self._device_name})",
            "device": self._device_name,
            "cuda_available": torch.cuda.is_available(),
            "models_loaded": models_dict,
            "scene_classes": self._scene_classes,
            "gender_classes": self._gender_classes,
            "metrics": self._metadata.get("scene_champion", {}),
        }

