"""Capa de Infraestructura (Clean Architecture): Adaptadores de Keras/CUDA y Repositorios."""

from backend_ia.infrastructure.camera_repository import InMemoryCameraRepository
from backend_ia.infrastructure.keras_model_adapter import KerasSecurityModelAdapter

__all__ = [
    "InMemoryCameraRepository",
    "KerasSecurityModelAdapter",
]

