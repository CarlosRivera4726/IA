"""
Casos de uso puros de la capa de aplicación (Clean Architecture).
Depende únicamente de domain y de los puertos de application.
"""

from typing import Dict, List, Optional
from backend_ia.application.ports import CameraRepositoryPort, ThreatClassifierPort
from backend_ia.domain.entities import CameraNode, CameraStatus, DetectionResult


class CameraDisabledError(Exception):
    """Excepción lanzada cuando se intenta inferir sobre una cámara deshabilitada."""
    pass


class InvalidImagePayloadError(Exception):
    """Excepción lanzada cuando el payload de imagen es nulo o vacío."""
    pass


class AnalyzeCameraFrameUseCase:
    """Caso de uso: Procesar un frame de video y ejecutar la detección automática."""

    def __init__(
        self,
        classifier: ThreatClassifierPort,
        camera_repo: CameraRepositoryPort,
    ) -> None:
        self._classifier = classifier
        self._camera_repo = camera_repo

    def execute(
        self,
        camera_id: str,
        image_base64: str,
        frame_ts: Optional[int] = None,
    ) -> DetectionResult:
        if not image_base64 or not image_base64.strip():
            raise InvalidImagePayloadError("El frame de video es requerido y no puede estar vacío.")

        camera = self._camera_repo.get_camera_by_id(camera_id)
        if camera is not None and not camera.is_active:
            raise CameraDisabledError(
                f"La cámara '{camera_id}' ({camera.name}) se encuentra temporalmente deshabilitada en el servidor."
            )

        return self._classifier.classify_frame(
            camera_id=camera_id,
            image_base64=image_base64,
            frame_ts=frame_ts,
        )


class GetCameraGridStatusUseCase:
    """Caso de uso: Consultar la topología y estado de las 6 cámaras."""

    def __init__(self, camera_repo: CameraRepositoryPort) -> None:
        self._camera_repo = camera_repo

    def execute(self) -> List[CameraNode]:
        return self._camera_repo.get_all_cameras()
