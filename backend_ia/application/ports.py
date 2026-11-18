"""
Puertos formales de la capa de aplicación (Clean Architecture).
Define contratos abstractos sin dependencias de infraestructura o librerías externas.
"""

from typing import Dict, List, Optional, Protocol
from backend_ia.domain.entities import CameraNode, DetectionResult


class ThreatClassifierPort(Protocol):
    """Puerto para el pipeline de inferencia neuronal y análisis temporal."""

    def classify_frame(
        self,
        camera_id: str,
        image_base64: str,
        frame_ts: Optional[int] = None,
    ) -> DetectionResult:
        """Procesa un frame real de video y retorna el resultado del análisis temporal."""
        ...

    def get_model_health(self) -> Dict[str, object]:
        """Retorna el estado de carga y salud de los modelos neuronales."""
        ...


class CameraRepositoryPort(Protocol):
    """Puerto para la consulta y estado de la grilla de cámaras."""

    def get_all_cameras(self) -> List[CameraNode]:
        """Retorna la lista de las 6 cámaras configuradas en el sistema."""
        ...

    def get_camera_by_id(self, camera_id: str) -> Optional[CameraNode]:
        """Retorna una cámara por su identificador."""
        ...
