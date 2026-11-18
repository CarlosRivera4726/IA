"""
Entidades de dominio puro para el sistema de seguridad SentinelCV (Contrato v2).
Totalmente desacoplado de frameworks de infraestructura, librerías de IA y HTTP.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ThreatLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SubjectGender(str, Enum):
    HOMBRE = "Hombre"
    MUJER = "Mujer"
    NO_DETERMINADO = "No_Determinado"


class CameraStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TEMPORARILY_DISABLED = "TEMPORARILY_DISABLED"
    ERROR = "ERROR"


# 5 Clases de escena del clasificador Keras
CANONICAL_SCENE_CLASSES = [
    "Arma_Blanca",
    "Arma_De_Fuego",
    "Ataque_Agresion",
    "Escena_Normal",
    "Hurto_Sospechoso",
]

# Todos los posibles eventos primarios emitibles por el sistema (incluyendo señales de auxilio por pose)
PRIMARY_EVENTS = [
    "Arma_Blanca",
    "Arma_De_Fuego",
    "Ataque_Agresion",
    "Escena_Normal",
    "Hurto_Sospechoso",
    "Senal_Auxilio_Hombre",
    "Senal_Auxilio_Mujer",
]


@dataclass(frozen=True)
class DetectedKeypoint:
    name: str
    x: float  # Normalizado 0.0 .. 1.0
    y: float  # Normalizado 0.0 .. 1.0
    score: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "score": round(self.score, 3),
        }


@dataclass(frozen=True)
class DetectedPerson:
    track_id: int
    bbox: List[float]  # [xmin, ymin, xmax, ymax] normalizado 0.0 .. 1.0
    score: float
    gender: SubjectGender
    gender_confidence: float
    keypoints: List[DetectedKeypoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "bbox": [round(c, 4) for c in self.bbox],
            "score": round(self.score, 3),
            "gender": self.gender.value,
            "gender_confidence": round(self.gender_confidence, 3),
            "keypoints": [kp.to_dict() for kp in self.keypoints],
        }


@dataclass(frozen=True)
class DetectedObject:
    label: str
    bbox: List[float]  # [xmin, ymin, xmax, ymax]
    confidence: float
    near_person: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "bbox": [round(c, 4) for c in self.bbox],
            "confidence": round(self.confidence, 3),
            "near_person": self.near_person,
        }


@dataclass(frozen=True)
class DetectedEventEvidence:
    type: str
    confidence: float
    evidence: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": self.type,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class FrameObservation:
    """Observación instantánea extraída de un único frame."""
    frame_ts: int
    persons: List[DetectedPerson]
    objects: List[DetectedObject]
    scene_probabilities: Dict[str, float]


@dataclass(frozen=True)
class DetectionResult:
    """Resultado final del análisis temporal para una cámara (Contrato API v2)."""
    camera_id: str
    frame_ts: int
    primary_event: str
    threat_level: ThreatLevel
    confidence: float
    events: List[DetectedEventEvidence]
    persons: List[DetectedPerson]
    objects: List[DetectedObject]
    scene_probabilities: Dict[str, float]
    window: Dict[str, object]  # {"frames": int, "seconds": float}
    model_info: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        events_list = [ev.to_dict() for ev in self.events]
        primary_evidence = (
            events_list[0]
            if events_list
            else {
                "type": self.primary_event,
                "confidence": round(self.confidence, 3),
                "evidence": "Supervisión continua sin anomalías de seguridad.",
                "description": "Supervisión continua sin anomalías de seguridad.",
                "rule_triggered": "baseline_scene",
            }
        )
        persons_list = [p.to_dict() for p in self.persons]
        objects_list = [obj.to_dict() for obj in self.objects]
        return {
            "camera_id": self.camera_id,
            "frame_ts": self.frame_ts,
            "primary_event": self.primary_event,
            "predicted_class": self.primary_event,
            "threat_level": self.threat_level.value,
            "is_threat": self.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL),
            "confidence": round(self.confidence, 3),
            "events": events_list,
            "evidence": primary_evidence,
            "persons": persons_list,
            "detected_persons": persons_list,
            "objects": objects_list,
            "detected_objects": objects_list,
            "scene_probabilities": {
                k: round(v, 4) for k, v in self.scene_probabilities.items()
            },
            "probabilities": {
                k: round(v, 4) for k, v in self.scene_probabilities.items()
            },
            "window": self.window,
            "model_info": self.model_info,
        }


@dataclass(frozen=True)
class CameraNode:
    camera_id: str
    name: str
    location: str
    status: CameraStatus
    is_active: bool
    connection_mode: str
    stream_url: Optional[str] = None
    resolution: Optional[str] = None
    description: Optional[str] = None
    status_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "id": self.camera_id,
            "name": self.name,
            "label": self.name,
            "location": self.location,
            "zone": self.location,
            "location_zone": self.location,
            "status": self.status.value if isinstance(self.status, CameraStatus) else str(self.status),
            "is_active": self.is_active,
            "connection_mode": self.connection_mode,
            "stream_url": self.stream_url,
            "stream_endpoint": self.stream_url,
            "resolution": self.resolution or "640x480",
            "description": self.description or self.status_reason,
            "status_reason": self.status_reason or self.description,
        }
