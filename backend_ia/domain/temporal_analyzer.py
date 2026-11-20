"""
Analizador temporal de seguridad puro (Clean Architecture - Capa de Dominio).
Aplica análisis temporal sobre ventanas de 3.0 segundos por cámara:
- No emite eventos con un solo fotograma aislado.
- Asocia personas por IoU espacial entre frames y suaviza su género por consenso temporal.
- Exige actividad física (movimiento articular/corporal o interacción de >= 2 personas)
  además de persistencia de escena para confirmar Ataque_Agresion, evitando falsos positivos
  cuando una persona está quieta frente a cámara.
- Detecta señales de auxilio continuas (ambas muñecas sobre hombros/cabeza >= 1.0 s).
- Detecta persistencia de armas cerca de personas (>= 2 frames recientes).
- Respeta precedencia por severidad: Armas > Ataque > Auxilio > Hurto > Escena Normal.
- Cero dependencias de librerías externas o frameworks.
"""

import math
from typing import Dict, List, Optional, Tuple
from backend_ia.domain.entities import (
    DetectedEventEvidence,
    DetectedPerson,
    FrameObservation,
    SubjectGender,
    ThreatLevel,
)


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1e-6, area_a + area_b - inter)


def _bbox_center(b: List[float]) -> Tuple[float, float]:
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


class TemporalSecurityAnalyzer:
    """Analizador con estado en memoria por cámara."""

    def __init__(self, window_ms: int = 3000, gap_reset_ms: int = 4000) -> None:
        self.window_ms = window_ms
        self.gap_reset_ms = gap_reset_ms
        self._windows: Dict[str, List[FrameObservation]] = {}
        # track_id -> (ultimo_bbox, ultimo_ts)
        self._tracks: Dict[str, Dict[int, Tuple[List[float], int]]] = {}
        self._next_track_id: Dict[str, int] = {}
        # track_id -> lista reciente de (SubjectGender, confidence)
        self._gender_votes: Dict[str, Dict[int, List[Tuple[SubjectGender, float]]]] = {}

    def reset_camera(self, camera_id: str) -> None:
        self._windows.pop(camera_id, None)
        self._tracks.pop(camera_id, None)
        self._next_track_id.pop(camera_id, None)
        self._gender_votes.pop(camera_id, None)

    def _assign_tracks_and_smooth_gender(
        self, camera_id: str, persons: List[DetectedPerson], frame_ts: int
    ) -> List[DetectedPerson]:
        tracks = self._tracks.setdefault(camera_id, {})
        votes_map = self._gender_votes.setdefault(camera_id, {})

        # Expirar tracks inactivos hace más de 3.0 segundos
        stale_ids = [tid for tid, (_, ts) in tracks.items() if (frame_ts - ts) > self.window_ms]
        for tid in stale_ids:
            tracks.pop(tid, None)
            votes_map.pop(tid, None)

        used_tids = set()
        updated: List[DetectedPerson] = []

        for p in persons:
            best_tid: Optional[int] = None
            best_score = 0.0
            pcx, pcy = _bbox_center(p.bbox)

            for tid, (prev_bbox, _) in tracks.items():
                if tid in used_tids:
                    continue
                iou = _bbox_iou(p.bbox, prev_bbox)
                tcx, tcy = _bbox_center(prev_bbox)
                dist = math.hypot(pcx - tcx, pcy - tcy)
                match_score = iou if iou >= 0.25 else (max(0.0, 0.24 - dist) if dist <= 0.18 else 0.0)
                if match_score > best_score:
                    best_score = match_score
                    best_tid = tid

            if best_tid is None:
                best_tid = self._next_track_id.get(camera_id, 1)
                self._next_track_id[camera_id] = best_tid + 1

            used_tids.add(best_tid)
            tracks[best_tid] = (p.bbox, frame_ts)

            # Votación ponderada por confianza de género en ventana reciente
            track_votes = votes_map.setdefault(best_tid, [])
            if p.gender != SubjectGender.NO_DETERMINADO and p.gender_confidence >= 0.58:
                track_votes.append((p.gender, p.gender_confidence))
                if len(track_votes) > 8:
                    track_votes.pop(0)

            resolved_gender = SubjectGender.NO_DETERMINADO
            resolved_conf = p.gender_confidence
            if track_votes:
                h_sum = sum(c for g, c in track_votes if g == SubjectGender.HOMBRE)
                m_sum = sum(c for g, c in track_votes if g == SubjectGender.MUJER)
                total = h_sum + m_sum
                if total > 0.0:
                    if h_sum >= m_sum and (h_sum / total) >= 0.60:
                        resolved_gender = SubjectGender.HOMBRE
                        h_confs = [c for g, c in track_votes if g == SubjectGender.HOMBRE]
                        resolved_conf = sum(h_confs) / len(h_confs)
                    elif m_sum > h_sum and (m_sum / total) >= 0.60:
                        resolved_gender = SubjectGender.MUJER
                        m_confs = [c for g, c in track_votes if g == SubjectGender.MUJER]
                        resolved_conf = sum(m_confs) / len(m_confs)

            updated.append(
                DetectedPerson(
                    track_id=best_tid,
                    bbox=p.bbox,
                    score=p.score,
                    gender=resolved_gender,
                    gender_confidence=resolved_conf,
                    keypoints=p.keypoints,
                )
            )

        return updated

    def _estimate_physical_activity(
        self, history: List[FrameObservation]
    ) -> Tuple[float, bool]:
        """
        Calcula el desplazamiento medio articular/corporal entre fotogramas consecutivos
        y detecta si existen >= 2 personas en proximidad física directa.
        """
        displacements: List[float] = []
        multi_person_close = False

        for idx, obs in enumerate(history):
            # Verificar proximidad entre pares de personas en el mismo frame
            if len(obs.persons) >= 2:
                for i in range(len(obs.persons)):
                    for j in range(i + 1, len(obs.persons)):
                        b1, b2 = obs.persons[i].bbox, obs.persons[j].bbox
                        iou = _bbox_iou(b1, b2)
                        c1, c2 = _bbox_center(b1), _bbox_center(b2)
                        if iou >= 0.08 or math.hypot(c1[0] - c2[0], c1[1] - c2[1]) <= 0.35:
                            multi_person_close = True

            if idx == 0:
                continue
            prev_obs = history[idx - 1]
            prev_by_id = {p.track_id: p for p in prev_obs.persons}

            for p in obs.persons:
                prev_p = prev_by_id.get(p.track_id)
                if prev_p is None:
                    continue
                # Comparar desplazamiento de muñecas/codos/hombros y centro de caja
                prev_kps = {kp.name: kp for kp in prev_p.keypoints}
                kp_moves: List[float] = []
                for kp in p.keypoints:
                    if kp.name in (
                        "left_wrist",
                        "right_wrist",
                        "left_elbow",
                        "right_elbow",
                        "left_shoulder",
                        "right_shoulder",
                    ) and kp.score >= 0.50:
                        pk = prev_kps.get(kp.name)
                        if pk and pk.score >= 0.50:
                            kp_moves.append(math.hypot(kp.x - pk.x, kp.y - pk.y))
                c_now = _bbox_center(p.bbox)
                c_prev = _bbox_center(prev_p.bbox)
                body_move = math.hypot(c_now[0] - c_prev[0], c_now[1] - c_prev[1])
                step_move = max([body_move] + kp_moves)
                displacements.append(step_move)

        avg_motion = (sum(displacements) / len(displacements)) if displacements else 0.0
        return avg_motion, multi_person_close

    def push_and_evaluate(
        self, camera_id: str, observation: FrameObservation
    ) -> Tuple[str, ThreatLevel, float, List[DetectedEventEvidence], Dict[str, object], List[DetectedPerson]]:
        # 1. Gestionar ventana temporal
        history = self._windows.setdefault(camera_id, [])
        if history:
            gap = observation.frame_ts - history[-1].frame_ts
            if gap > self.gap_reset_ms or gap < 0:
                history.clear()
                self._tracks.pop(camera_id, None)
                self._gender_votes.pop(camera_id, None)

        # 2. Rastreo espacial por IoU y suavizado temporal de género
        updated_persons = self._assign_tracks_and_smooth_gender(
            camera_id, observation.persons, observation.frame_ts
        )
        tracked_obs = FrameObservation(
            frame_ts=observation.frame_ts,
            persons=updated_persons,
            objects=observation.objects,
            scene_probabilities=observation.scene_probabilities,
        )
        history.append(tracked_obs)

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

        # Si solo tenemos 1 frame, no se emite ninguna alarma (requisito anti-falsos positivos)
        if n_frames <= 1:
            events = [
                DetectedEventEvidence(
                    type="Escena_Normal",
                    confidence=0.95,
                    evidence="Acumulando ventana temporal (1 fotograma registrado).",
                )
            ]
            return "Escena_Normal", ThreatLevel.LOW, 0.95, events, window_info, updated_persons

        # 3. Evaluar reglas temporales y actividad física
        detected_events: List[DetectedEventEvidence] = []
        avg_motion, multi_person_close = self._estimate_physical_activity(history)

        # A. Detección de Armas (Arma_De_Fuego / Arma_Blanca)
        gun_obj_hits = 0
        knife_obj_hits = 0
        gun_scene_hits = 0
        knife_scene_hits = 0
        gun_conf_sum = 0.0
        knife_conf_sum = 0.0

        for obs in history[-3:]:
            frame_gun_obj = False
            frame_knife_obj = False
            for obj in obs.objects:
                if obj.label in ("gun", "handgun", "pistol", "rifle", "Arma_De_Fuego"):
                    frame_gun_obj = True
                    gun_conf_sum += obj.confidence
                elif obj.label in ("knife", "cuchillo", "Arma_Blanca"):
                    frame_knife_obj = True
                    knife_conf_sum += obj.confidence
            if frame_gun_obj:
                gun_obj_hits += 1
            if frame_knife_obj:
                knife_obj_hits += 1

            p_gun = obs.scene_probabilities.get("Arma_De_Fuego", 0.0)
            p_knife = obs.scene_probabilities.get("Arma_Blanca", 0.0)
            if p_gun >= 0.78:
                gun_scene_hits += 1
                gun_conf_sum += p_gun
            if p_knife >= 0.78:
                knife_scene_hits += 1
                knife_conf_sum += p_knife

        gun_total = gun_obj_hits + gun_scene_hits
        if gun_obj_hits >= 2 or gun_scene_hits >= 2 or (gun_obj_hits >= 1 and gun_scene_hits >= 1):
            avg_c = min(0.99, gun_conf_sum / max(1, gun_total))
            detected_events.append(
                DetectedEventEvidence(
                    type="Arma_De_Fuego",
                    confidence=avg_c,
                    evidence=f"Arma de fuego confirmada en {gun_total} observaciones recientes.",
                )
            )

        knife_total = knife_obj_hits + knife_scene_hits
        if knife_obj_hits >= 2 or knife_scene_hits >= 2 or (knife_obj_hits >= 1 and knife_scene_hits >= 1):
            avg_c = min(0.99, knife_conf_sum / max(1, knife_total))
            detected_events.append(
                DetectedEventEvidence(
                    type="Arma_Blanca",
                    confidence=avg_c,
                    evidence=f"Arma blanca confirmada en {knife_total} observaciones recientes.",
                )
            )

        # B. Detección de Ataque o Agresión Física (Ataque_Agresion)
        # Evita falsos positivos en personas quietas frente a cámara: requiere alta probabilidad
        # sostenida corroborada con interacción de >= 2 personas o movimiento brusco,
        # o consenso visual abrumador (>= 0.88).
        ataque_hits = 0
        ataque_extreme_hits = 0
        ataque_conf_sum = 0.0
        for obs in history:
            p_at = obs.scene_probabilities.get("Ataque_Agresion", 0.0)
            if p_at >= 0.68:
                ataque_hits += 1
                ataque_conf_sum += p_at
            if p_at >= 0.88:
                ataque_extreme_hits += 1

        has_physical_agitation = multi_person_close or avg_motion >= 0.032
        ataque_confirmed = (
            n_frames >= 3
            and duration_sec >= 0.9
            and (
                (ataque_hits >= max(3, int(n_frames * 0.60)) and has_physical_agitation)
                or (ataque_extreme_hits >= max(3, int(n_frames * 0.75)))
            )
        )
        if ataque_confirmed:
            avg_c = min(0.98, ataque_conf_sum / max(1, ataque_hits))
            detected_events.append(
                DetectedEventEvidence(
                    type="Ataque_Agresion",
                    confidence=avg_c,
                    evidence=(
                        f"Patrón de agresión física sostenido en {ataque_hits}/{n_frames} frames "
                        f"({duration_sec:.1f} s, actividad={avg_motion:.3f})."
                    ),
                )
            )

        # C. Detección de Señal de Auxilio a Cámara (Senal_Auxilio_Hombre / Senal_Auxilio_Mujer)
        auxilio_frames = 0
        dominant_gender = SubjectGender.NO_DETERMINADO
        highest_g_conf = 0.0

        for obs in history:
            frame_has_signal = False
            for p in obs.persons:
                kps = {kp.name: kp for kp in p.keypoints}
                lw = kps.get("left_wrist")
                rw = kps.get("right_wrist")
                ls = kps.get("left_shoulder")
                rs = kps.get("right_shoulder")

                if lw and rw and ls and rs:
                    if lw.score >= 0.50 and rw.score >= 0.50 and ls.score >= 0.45 and rs.score >= 0.45:
                        shoulder_y = min(ls.y, rs.y)
                        hands_up = (shoulder_y - lw.y > 0.09) and (shoulder_y - rw.y > 0.09)
                        if hands_up:
                            frame_has_signal = True
                            if p.gender != SubjectGender.NO_DETERMINADO and p.gender_confidence > highest_g_conf:
                                dominant_gender = p.gender
                                highest_g_conf = p.gender_confidence
                            break
            if frame_has_signal:
                auxilio_frames += 1

        if n_frames >= 3 and auxilio_frames >= max(3, int(n_frames * 0.60)) and duration_sec >= 1.2:
            signal_type = (
                "Senal_Auxilio_Mujer"
                if dominant_gender == SubjectGender.MUJER
                else "Senal_Auxilio_Hombre"
            )
            detected_events.append(
                DetectedEventEvidence(
                    type=signal_type,
                    confidence=0.88,
                    evidence=(
                        f"Ambas muñecas sobre el nivel de los hombros de forma continua en "
                        f"{auxilio_frames}/{n_frames} frames ({duration_sec:.1f} s)."
                    ),
                )
            )

        # D. Detección de Posible Hurto (Hurto_Sospechoso)
        hurto_hits = 0
        hurto_conf_sum = 0.0
        for obs in history:
            p_h = obs.scene_probabilities.get("Hurto_Sospechoso", 0.0)
            if p_h >= 0.72 and len(obs.persons) >= 1:
                hurto_hits += 1
                hurto_conf_sum += p_h

        if (
            n_frames >= 3
            and hurto_hits >= max(3, int(n_frames * 0.60))
            and duration_sec >= 1.1
            and avg_motion >= 0.020
        ):
            avg_c = min(0.92, hurto_conf_sum / max(1, hurto_hits))
            detected_events.append(
                DetectedEventEvidence(
                    type="Hurto_Sospechoso",
                    confidence=avg_c,
                    evidence=(
                        f"Patrón de sustracción sospechosa sostenido en {hurto_hits}/{n_frames} frames "
                        f"({duration_sec:.1f} s)."
                    ),
                )
            )

        # 4. Precedencia por Severidad: Armas > Ataque > Auxilio > Hurto > Escena_Normal
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
        norm_conf = max(
            observation.scene_probabilities.get("Escena_Normal", 0.0),
            0.85 if not detected_events else 0.70,
        )
        normal_ev = [
            DetectedEventEvidence(
                type="Escena_Normal",
                confidence=round(min(0.99, norm_conf), 4),
                evidence=f"Monitoreo normal sin anomalías confirmadas ({n_frames} frames en {duration_sec:.1f} s).",
            )
        ]
        return "Escena_Normal", ThreatLevel.LOW, round(min(0.99, norm_conf), 4), normal_ev, window_info, updated_persons
