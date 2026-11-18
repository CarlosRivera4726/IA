"""
Repositorio de las 6 camaras del sistema SentinelCV.
- CAM-01: Activa, conectada a la camara del dispositivo actual.
- CAM-02 a CAM-06: Nodos de servidor temporalmente deshabilitados esperando conexion.
"""

from typing import Dict, List, Optional

from backend_ia.application.ports import CameraRepositoryPort
from backend_ia.domain.entities import CameraNode, CameraStatus


class InMemoryCameraRepository(CameraRepositoryPort):
    """Implementacion del repositorio de nodos de camara."""

    def __init__(self) -> None:
        self._nodes: List[CameraNode] = [
            CameraNode(
                camera_id="CAM-01",
                name="Camara 01 - Terminal local activa",
                location="Puesto de control principal (Dispositivo actual)",
                status=CameraStatus.ACTIVE,
                is_active=True,
                connection_mode="LOCAL_DEVICE_WEBCAM",
                stream_url="local://navigator.mediaDevices.getUserMedia",
                resolution="640x480 (30 FPS)",
                description="Enlace optico local activo con inferencia neuronal en tiempo real.",
                status_reason="Enlace optico local activo con inferencia neuronal en tiempo real.",
            ),
            CameraNode(
                camera_id="CAM-02",
                name="Camara 02 - Acceso peatonal norte",
                location="Perimetro exterior sector A",
                status=CameraStatus.TEMPORARILY_DISABLED,
                is_active=False,
                connection_mode="SERVER_RTSP_NODE",
                stream_url="rtsp://10.24.0.102:554/sentinel/cam02",
                resolution="1920x1080 (Offline)",
                description="Deshabilitada temporalmente en espera de enlace con servidor RTSP.",
                status_reason="Deshabilitada temporalmente en espera de enlace con servidor RTSP.",
            ),
            CameraNode(
                camera_id="CAM-03",
                name="Camara 03 - Boveda y caja principal",
                location="Nivel 1 area restringida",
                status=CameraStatus.TEMPORARILY_DISABLED,
                is_active=False,
                connection_mode="SERVER_RTSP_NODE",
                stream_url="rtsp://10.24.0.103:554/sentinel/cam03",
                resolution="1920x1080 (Offline)",
                description="Deshabilitada temporalmente en espera de enlace con servidor RTSP.",
                status_reason="Deshabilitada temporalmente en espera de enlace con servidor RTSP.",
            ),
            CameraNode(
                camera_id="CAM-04",
                name="Camara 04 - Pasillo de carga oriental",
                location="Muelle logistico sector B",
                status=CameraStatus.TEMPORARILY_DISABLED,
                is_active=False,
                connection_mode="SERVER_RTSP_NODE",
                stream_url="rtsp://10.24.0.104:554/sentinel/cam04",
                resolution="1920x1080 (Offline)",
                description="Deshabilitada temporalmente en espera de enlace con servidor RTSP.",
                status_reason="Deshabilitada temporalmente en espera de enlace con servidor RTSP.",
            ),
            CameraNode(
                camera_id="CAM-05",
                name="Camara 05 - Estacionamiento subterraneo",
                location="Sotano 1 rampa vehicular",
                status=CameraStatus.TEMPORARILY_DISABLED,
                is_active=False,
                connection_mode="SERVER_RTSP_NODE",
                stream_url="rtsp://10.24.0.105:554/sentinel/cam05",
                resolution="1920x1080 (Offline)",
                description="Deshabilitada temporalmente en espera de enlace con servidor RTSP.",
                status_reason="Deshabilitada temporalmente en espera de enlace con servidor RTSP.",
            ),
            CameraNode(
                camera_id="CAM-06",
                name="Camara 06 - Galeria comercial nivel 2",
                location="Corredor central de vitrinas",
                status=CameraStatus.TEMPORARILY_DISABLED,
                is_active=False,
                connection_mode="SERVER_RTSP_NODE",
                stream_url="rtsp://10.24.0.106:554/sentinel/cam06",
                resolution="1920x1080 (Offline)",
                description="Deshabilitada temporalmente en espera de enlace con servidor RTSP.",
                status_reason="Deshabilitada temporalmente en espera de enlace con servidor RTSP.",
            ),
        ]
        self._by_id: Dict[str, CameraNode] = {n.camera_id: n for n in self._nodes}

    def get_all_cameras(self) -> List[CameraNode]:
        return list(self._nodes)

    def get_camera_by_id(self, camera_id: str) -> Optional[CameraNode]:
        return self._by_id.get(camera_id.strip().upper())

    def list_cameras(self) -> List[CameraNode]:
        return self.get_all_cameras()

    def get_camera(self, camera_id: str) -> Optional[CameraNode]:
        return self.get_camera_by_id(camera_id)
