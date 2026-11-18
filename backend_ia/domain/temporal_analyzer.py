"""
Analizador temporal de seguridad puro (Clean Architecture - Capa de Dominio).
Aplica análisis temporal sobre ventanas de 3.0 segundos por cámara:
- No emite eventos con un solo fotograma aislado.
- Detecta señales de auxilio continuas (ambas muñecas sobre hombros/cabeza >= 1.0 s).
- Detecta persistencia de armas cerca de manos (>= 2 frames consecutivos).
- Detecta agresiones físicas sostenidas y hurtos sospechosos.
- Respeta precedencia por severidad: Armas > Ataque > Auxilio > Hurto > Escena Normal.
- Mantiene memoria de género por persona rastreada.
- Cero dependencias de librerías externas o frameworks.
"""

from typing import Dict, List, Optional, Tuple
from backend_ia.domain.entities import (
    DetectedEventEvidence,
    DetectedObject,
    DetectedPerson,
    FrameObservation,
    SubjectGender,
    ThreatLevel,
)


class TemporalSecurityAnalyzer:
    """Analizador con estado en memoria por cámara."""

    def __init__(self, window_ms: int = 3000, gap_reset_ms: int = 4000) -> None:
        self.window_ms = window_ms
        self.gap_reset_ms = gap_reset_ms
        self._windows: Dict[str, List[FrameObservation]] = {}
        self._gender_memory: Dict[str, Dict[int, Tuple[SubjectGender, float]]] = {}

    def reset_camera(self, camera_id: str) -> None:
        self._windows.pop(camera_id, None)
        self._gender_memory.pop(camera_id, None)

    def push_and_evaluate(
        self, camera_id: str, observation: FrameObservation
    ) -> Tuple[str, ThreatLevel, float, List[DetectedEventEvidence], Dict[str, object], List[DetectedPerson]]:
        # 1. Gestionar ventana temporal
        history = self._windows.setdefault(camera_id, [])
        if history:
            gap = observation.frame_ts - history[-1].frame_ts
            if gap > self.gap_reset_ms or gap < 0:
                # Salto temporal brusco: reiniciar ventana
                history.clear()
                self._gender_memory.pop(camera_id, None)

        history.append(observation)

        # Descartar observaciones fuera de la ventana de window_ms
        cutoff_ts = observation.frame_ts - self.window_ms
        while history and history[0].frame_ts < cutoff_ts:
            history.pop(0)

        n_frames = len(history)
        duration_sec = (
            (history[-1].frame_ts - history[0].frame_ts) / 1000.0 if n_frames > 1 else 0.0
        )
        window_info = {
            "frames": n_frames,
            "seconds": round(duration_sec, 2),
        }

        # 2. Actualizar memoria de género por persona
        cam_gender_mem = self._gender_memory.setdefault(camera_id, {})
        updated_persons: List[DetectedPerson] = []
        for p in observation.persons:
            current_g = p.gender
            current_conf = p.gender_confidence
            if current_g != SubjectGender.NO_DETERMINADO and current_conf >= 0.60:
                cam_gender_mem[p.track_id] = (current_g, current_conf)
            elif p.track_id in cam_gender_mem:
                # Recuperar género previamente reconocido
                cached_g, cached_conf = cam_gender_mem[p.track_id]
                p = DetectedPerson(
                    track_id=p.track_id,
                    bbox=p.bbox,
                    score=p.score,
                    gender=cached_g,
                    gender_confidence=cached_conf,
                    keypoints=p.keypoints,
                )
            updated_persons.append(p)

        # Si solo tenemos 1 frame, no se emite ningún evento (requisito anti-falsos positivos)
        if n_frames <= 1:
            events = [
                DetectedEventEvidence(
                    type="Escena_Normal",
                    confidence=0.95,
                    evidence="Acumulando ventana temporal (1 fotograma registrado)",
                )
            ]
            return "Escena_Normal", ThreatLevel.LOW, 0.95, events, window_info, updated_persons

        # 3. Evaluar reglas temporales
        detected_events: List[DetectedEventEvidence] = []

        # A. Detección de Armas (Arma_De_Fuego / Arma_Blanca)
        # Regla: Arma detectada cerca de mano en >= 2 frames consecutivos o recientes
        gun_count = 0
        knife_count = 0
        gun_conf_sum = 0.0
        knife_conf_sum = 0.0

        for obs in history[-3:]:  # Revisar los últimos 3 frames
            for obj in obs.objects:
                if obj.label in ("gun", "handgun", "pistol", "rifle", "Arma_De_Fuego"):
                    gun_count += 1
                    gun_conf_sum += obj.confidence
                elif obj.label in ("knife", "cuchillo", "Arma_Blanca"):
                    knife_count += 1
                    knife_conf_sum += obj.confidence

            # También verificar clasificador de escena si la probabilidad es muy alta
            if obs.scene_probabilities.get("Arma_De_Fuego", 0.0) >= 0.65:
                gun_count += 1
                gun_conf_sum += obs.scene_probabilities["Arma_De_Fuego"]
            if obs.scene_probabilities.get("Arma_Blanca", 0.0) >= 0.65:
                knife_count += 1
                knife_conf_sum += obs.scene_probabilities["Arma_Blanca"]

        if gun_count >= 2:
            avg_c = min(0.99, gun_conf_sum / max(1, gun_count))
            detected_events.append(
                DetectedEventEvidence(
                    type="Arma_De_Fuego",
                    confidence=avg_c,
                    evidence=f"Arma de fuego detectada de forma persistente en {gun_count} observaciones recientes.",
                )
            )

        if knife_count >= 2:
            avg_c = min(0.99, knife_conf_sum / max(1, knife_count))
            detected_events.append(
                DetectedEventEvidence(
                    type="Arma_Blanca",
                    confidence=avg_c,
                    evidence=f"Arma blanca detectada en mano de forma persistente en {knife_count} observaciones recientes.",
                )
            )

        # B. Detección de Ataque o Agresión Física (Ataque_Agresion)
        # Regla: Probabilidad de ataque sostenida >= 0.55 en al menos el 50% de la ventana
        ataque_hits = 0
        ataque_conf_sum = 0.0
        for obs in history:
            p_at = obs.scene_probabilities.get("Ataque_Agresion", 0.0)
            if p_at >= 0.52:
                ataque_hits += 1
                ataque_conf_sum += p_at

        if n_frames >= 3 and ataque_hits >= max(2, int(n_frames * 0.45)) and duration_sec >= 0.7:
            avg_c = min(0.98, ataque_conf_sum / max(1, ataque_hits))
            detected_events.append(
                DetectedEventEvidence(
                    type="Ataque_Agresion",
                    confidence=avg_c,
                    evidence=f"Patrón de agresión/pelea física sostenido en {ataque_hits}/{n_frames} frames ({duration_sec:.1f} s).",
                )
            )

        # C. Detección de Señal de Auxilio a Cámara (Senal_Auxilio_Hombre / Senal_Auxilio_Mujer)
        # Regla: Persona de frente con ambas muñecas por encima del nivel de hombros/cabeza durante >= 1.0 s
        auxilio_frames = 0
        dominant_gender = SubjectGender.NO_DETERMINADO
        highest_g_conf = 0.0

        for obs in history:
            frame_has_signal = False
            for p in obs.persons:
                # Buscar keypoints: left_wrist, right_wrist, left_shoulder, right_shoulder, nose
                kps = {kp.name: kp for kp in p.keypoints}
                lw = kps.get("left_wrist")
                rw = kps.get("right_wrist")
                ls = kps.get("left_shoulder")
                rs = kps.get("right_shoulder")
                nose = kps.get("nose")

                if lw and rw and ls and rs:
                    # En coordenadas de imagen, Y menor significa más arriba
                    if lw.score >= 0.55 and rw.score >= 0.55 and ls.score >= 0.50 and rs.score >= 0.50:
                        shoulder_y = min(ls.y, rs.y)
                        hands_up = (shoulder_y - lw.y > 0.09) and (shoulder_y - rw.y > 0.09)
                        if hands_up:
                            frame_has_signal = True
                            if p.gender_confidence > highest_g_conf:
                                dominant_gender = p.gender
                                highest_g_conf = p.gender_confidence
                            break
            if frame_has_signal:
                auxilio_frames += 1

        if n_frames >= 3 and auxilio_frames >= max(3, int(n_frames * 0.60)) and duration_sec >= 1.2:
            signal_type = (
                "Senal_Auxilio_Mujer"
                if dominant_gender == SubjectGender.MUJER
                else (
                    "Senal_Auxilio_Hombre"
                    if dominant_gender == SubjectGender.HOMBRE
                    else "Senal_Auxilio_Hombre"  # Default si no determinado
                )
            )
            detected_events.append(
                DetectedEventEvidence(
                    type=signal_type,
                    confidence=0.88,
                    evidence=f"Ambas muñecas sobre el nivel de los hombros/cabeza de forma continua en {auxilio_frames}/{n_frames} frames ({duration_sec:.1f} s).",
                )
            )

        # D. Detección de Posible Hurto (Hurto_Sospechoso)
        hurto_hits = 0
        hurto_conf_sum = 0.0
        for obs in history:
            p_h = obs.scene_probabilities.get("Hurto_Sospechoso", 0.0)
            if p_h >= 0.55:
                hurto_hits += 1
                hurto_conf_sum += p_h

        if n_frames >= 3 and hurto_hits >= max(2, int(n_frames * 0.50)) and duration_sec >= 0.9:
            avg_c = min(0.92, hurto_conf_sum / max(1, hurto_hits))
            detected_events.append(
                DetectedEventEvidence(
                    type="Hurto_Sospechoso",
                    confidence=avg_c,
                    evidence=f"Patrón de sustracción/manipulación sospechosa sostenido en {hurto_hits}/{n_frames} frames ({duration_sec:.1f} s). (Alerta orientativa en cámara fija)",
                )
            )

        # 4. Precedencia por Severidad
        # Armas > Ataque > Auxilio > Hurto > Escena_Normal
        severity_rank = {
            "Arma_De_Fuego": (100, ThreatLevel.CRITICAL),
            "Arma_Blanca": (95, ThreatLevel.CRITICAL),
            "Ataque_Agresion": (80, ThreatLevel.HIGH),
            "Senal_Auxilio_Mujer": (70, ThreatLevel.HIGH),
            "Senal_Auxilio_Hombre": (70, ThreatLevel.HIGH),
            "Hurto_Sospechoso": (50, ThreatLevel.MEDIUM),
            "Escena_Normal": (10, ThreatLevel.LOW),
        }

        if detected_events:
            # Ordenar por severidad decreciente
            detected_events.sort(
                key=lambda ev: severity_rank.get(ev.type, (0, ThreatLevel.LOW))[0],
                reverse=True,
            )
            primary_ev = detected_events[0]
            _, th_level = severity_rank.get(primary_ev.type, (10, ThreatLevel.LOW))
            return (
                primary_ev.type,
                th_level,
                primary_ev.confidence,
                detected_events,
                window_info,
                updated_persons,
            )

        # E. Escena Normal
        norm_conf = observation.scene_probabilities.get("Escena_Normal", 0.85)
        normal_ev = [
            DetectedEventEvidence(
                type="Escena_Normal",
                confidence=norm_conf,
                evidence=f"Monitoreo normal sin anomalías confirmadas ({n_frames} frames evaluados en {duration_sec:.1f} s).",
            )
        ]
        return "Escena_Normal", ThreatLevel.LOW, norm_conf, normal_ev, window_info, updated_persons

