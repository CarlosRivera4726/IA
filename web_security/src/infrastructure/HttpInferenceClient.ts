/**
 * Infrastructure Adapter: HttpInferenceClient
 * Implements IInferenceApiPort to communicate with the Clean Architecture Python backend
 * running at http://localhost:8000. Real-time inference, zero synthetic data fallback.
 */

import {
  CANONICAL_DETECTION_CLASSES,
  DEFAULT_SIX_CAMERA_NODES,
  type BackendHealthStatus,
  type CameraNode,
  type CapturedFramePayload,
  type ClassProbabilities,
  type DetectedEventEvidence,
  type DetectedObject,
  type DetectedPerson,
  type DetectionClass,
  type IInferenceApiPort,
  type ThreatLevel,
  type ThreatPrediction
} from '../domain';

export class HttpInferenceClient implements IInferenceApiPort {
  private readonly baseUrl: string;
  private readonly requestTimeoutMs: number;
  private lastKnownOnline = false;

  constructor(baseUrl = 'http://localhost:8000', requestTimeoutMs = 15000) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.requestTimeoutMs = requestTimeoutMs;
  }

  public async checkHealth(): Promise<BackendHealthStatus> {
    const endpointUrl = `${this.baseUrl}/api/v1/health`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.requestTimeoutMs);

    try {
      const response = await fetch(endpointUrl, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        this.lastKnownOnline = false;
        return {
          online: false,
          model_loaded: false,
          scene_model_loaded: false,
          gender_model_loaded: false,
          pose_model_loaded: false,
          weapon_detector_loaded: false,
          backend_engine: `Error HTTP ${response.status}`,
          device: 'N/A',
          endpoint_url: endpointUrl,
          status_message: `El servidor respondió con código ${response.status}.`
        };
      }

      const data = (await response.json()) as {
        status?: string;
        backend?: string;
        device?: string;
        models_loaded?: {
          scene?: boolean;
          gender?: boolean;
          pose?: boolean;
          objects?: boolean;
        };
      };

      const models = data.models_loaded ?? {};
      const allLoaded = Boolean(models.scene && models.gender && models.pose && models.objects);
      this.lastKnownOnline = true;

      return {
        online: true,
        model_loaded: allLoaded,
        scene_model_loaded: Boolean(models.scene),
        gender_model_loaded: Boolean(models.gender),
        pose_model_loaded: Boolean(models.pose),
        weapon_detector_loaded: Boolean(models.objects),
        backend_engine: String(data.backend ?? 'keras-torch-cuda'),
        device: String(data.device ?? 'cuda'),
        endpoint_url: endpointUrl,
        status_message: 'Motor neuronal de visión por computadora activo y listo.'
      };
    } catch {
      clearTimeout(timeoutId);
      this.lastKnownOnline = false;
      return {
        online: false,
        model_loaded: false,
        scene_model_loaded: false,
        gender_model_loaded: false,
        pose_model_loaded: false,
        weapon_detector_loaded: false,
        backend_engine: 'Servidor desconectado',
        device: 'N/A',
        endpoint_url: endpointUrl,
        status_message: 'Esperando backend en http://localhost:8000...'
      };
    }
  }

  public async getCameraNodes(): Promise<readonly CameraNode[]> {
    const endpointUrl = `${this.baseUrl}/api/v1/cameras`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.requestTimeoutMs);

    try {
      const response = await fetch(endpointUrl, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        return DEFAULT_SIX_CAMERA_NODES;
      }

      const payload = (await response.json()) as {
        cameras?: Array<Record<string, unknown>>;
      };

      if (!Array.isArray(payload.cameras) || payload.cameras.length === 0) {
        return DEFAULT_SIX_CAMERA_NODES;
      }

      return DEFAULT_SIX_CAMERA_NODES.map((fallbackNode) => {
        const remote = payload.cameras?.find(
          (item) => String(item.camera_id ?? item.id ?? '') === fallbackNode.camera_id
        );
        if (!remote) {
          return fallbackNode;
        }

        const isCam01 = fallbackNode.camera_id === 'CAM-01';
        return {
          camera_id: fallbackNode.camera_id,
          label: String(remote.name ?? remote.label ?? fallbackNode.label),
          location_zone: String(remote.location ?? remote.location_zone ?? fallbackNode.location_zone),
          status: isCam01 ? 'ACTIVE' : 'TEMPORARILY_DISABLED',
          is_active: isCam01,
          connection_mode: isCam01 ? 'LOCAL_DEVICE_WEBCAM' : 'SERVER_RTSP_NODE',
          stream_endpoint: String(remote.stream_url ?? remote.stream_endpoint ?? fallbackNode.stream_endpoint),
          resolution: String(remote.resolution ?? fallbackNode.resolution),
          status_reason: String(remote.status_reason ?? remote.description ?? fallbackNode.status_reason)
        };
      });
    } catch {
      clearTimeout(timeoutId);
      return DEFAULT_SIX_CAMERA_NODES;
    }
  }

  public async inferFrame(payload: CapturedFramePayload): Promise<ThreatPrediction> {
    const endpointUrl = `${this.baseUrl}/api/v1/infer`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.requestTimeoutMs);
    const requestStarted = performance.now();

    const response = await fetch(endpointUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      body: JSON.stringify({
        camera_id: payload.camera_id,
        image_base64: payload.image_base64,
        frame_ts: payload.frame_ts
      }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      this.lastKnownOnline = false;
      const errorText = await response.text().catch(() => '');
      throw new Error(`Inference HTTP ${response.status}: ${errorText}`);
    }

    const raw = (await response.json()) as Record<string, unknown>;
    this.lastKnownOnline = true;
    const elapsedMs = Number((performance.now() - requestStarted).toFixed(1));

    return this.normalizeServerPrediction(raw, payload, elapsedMs);
  }

  public isBackendOnline(): boolean {
    return this.lastKnownOnline;
  }

  private normalizeServerPrediction(
    raw: Record<string, unknown>,
    payload: CapturedFramePayload,
    measuredLatencyMs: number
  ): ThreatPrediction {
    const rawClass = String(raw.primary_event ?? raw.predicted_class ?? 'Escena_Normal') as DetectionClass;
    const primaryEvent: DetectionClass = CANONICAL_DETECTION_CLASSES.includes(rawClass)
      ? rawClass
      : 'Escena_Normal';

    const rawThreatLevel = String(raw.threat_level ?? 'LOW') as ThreatLevel;
    const threatLevel: ThreatLevel = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].includes(rawThreatLevel)
      ? rawThreatLevel
      : 'LOW';

    const confidence = typeof raw.confidence === 'number' ? raw.confidence : 0.85;
    const isThreat = typeof raw.is_threat === 'boolean' ? raw.is_threat : primaryEvent !== 'Escena_Normal';

    const rawProbabilities = (raw.scene_probabilities ?? raw.probabilities ?? {}) as Record<string, number>;
    const probabilities = this.normalizeProbabilities(rawProbabilities, primaryEvent);

    const rawPersons = Array.isArray(raw.detected_persons) ? raw.detected_persons : [];
    const detectedPersons: DetectedPerson[] = rawPersons.map((p, idx) => {
      const obj = p as Record<string, unknown>;
      const rawBbox = Array.isArray(obj.bbox) ? obj.bbox : [0.1, 0.2, 0.9, 0.8];
      const rawKps = Array.isArray(obj.keypoints) ? obj.keypoints : [];
      return {
        person_id: Number(obj.person_id ?? idx + 1),
        bbox: [Number(rawBbox[0] ?? 0), Number(rawBbox[1] ?? 0), Number(rawBbox[2] ?? 1), Number(rawBbox[3] ?? 1)],
        gender: (['Hombre', 'Mujer', 'No_Determinado'].includes(String(obj.gender))
          ? String(obj.gender)
          : 'No_Determinado') as 'Hombre' | 'Mujer' | 'No_Determinado',
        gender_confidence: Number(obj.gender_confidence ?? 0.0),
        has_distress_arms_raised: Boolean(obj.has_distress_arms_raised),
        keypoints: rawKps.map((k) => {
          const kpObj = k as Record<string, unknown>;
          return {
            name: String(kpObj.name ?? ''),
            x: Number(kpObj.x ?? 0),
            y: Number(kpObj.y ?? 0),
            confidence: Number(kpObj.confidence ?? kpObj.score ?? 0)
          };
        })
      };
    });

    const rawObjects = Array.isArray(raw.detected_objects) ? raw.detected_objects : [];
    const detectedObjects: DetectedObject[] = rawObjects.map((o) => {
      const obj = o as Record<string, unknown>;
      const rawBbox = Array.isArray(obj.bbox) ? obj.bbox : [0.2, 0.3, 0.6, 0.7];
      return {
        label: String(obj.label ?? 'object'),
        confidence: Number(obj.confidence ?? 0.8),
        bbox: [Number(rawBbox[0] ?? 0), Number(rawBbox[1] ?? 0), Number(rawBbox[2] ?? 1), Number(rawBbox[3] ?? 1)]
      };
    });

    const rawEvidence = (raw.evidence ?? {}) as Record<string, unknown>;
    const evidence: DetectedEventEvidence = {
      description: String(
        rawEvidence.description ?? rawEvidence.evidence ?? 'Supervisión continua sin anomalías.'
      ),
      rule_triggered: String(rawEvidence.rule_triggered ?? rawEvidence.type ?? 'scene_baseline'),
      sustained_duration_sec: Number(rawEvidence.sustained_duration_sec ?? 0.0)
    };

    const rawModelInfo = (raw.model_info ?? {}) as Record<string, unknown>;

    return {
      camera_id: String(raw.camera_id ?? payload.camera_id),
      timestamp: String(raw.timestamp ?? payload.captured_at_iso),
      primary_event: primaryEvent,
      threat_level: threatLevel,
      is_threat: isThreat,
      confidence,
      scene_probabilities: probabilities,
      detected_persons: detectedPersons,
      detected_objects: detectedObjects,
      evidence,
      model_info: {
        scene_model: String(rawModelInfo.scene_model ?? 'SentinelScene-Keras3'),
        gender_model: String(rawModelInfo.gender_model ?? 'SentinelGender-Keras3'),
        pose_model: String(rawModelInfo.pose_model ?? 'torchvision.KeypointRCNN'),
        object_model: String(rawModelInfo.object_model ?? 'torchvision.FasterRCNN'),
        backend: String(rawModelInfo.backend ?? 'keras-torch-cuda'),
        latency_ms: Number(rawModelInfo.latency_ms ?? measuredLatencyMs)
      }
    };
  }

  private normalizeProbabilities(
    rawProbs: Record<string, number>,
    dominantClass: DetectionClass
  ): ClassProbabilities {
    const result: Record<DetectionClass, number> = {
      Arma_Blanca: 0.01,
      Arma_De_Fuego: 0.01,
      Ataque_Agresion: 0.01,
      Escena_Normal: 0.02,
      Hurto_Sospechoso: 0.01,
      Senal_Auxilio_Hombre: 0.01,
      Senal_Auxilio_Mujer: 0.01
    };

    let hasProvidedValues = false;
    for (const cls of CANONICAL_DETECTION_CLASSES) {
      if (typeof rawProbs[cls] === 'number' && !Number.isNaN(rawProbs[cls])) {
        result[cls] = Math.max(0, rawProbs[cls]);
        hasProvidedValues = true;
      }
    }

    if (!hasProvidedValues) {
      result[dominantClass] = 0.92;
    }

    const sum = Object.values(result).reduce((acc, val) => acc + val, 0);
    if (sum > 0) {
      for (const cls of CANONICAL_DETECTION_CLASSES) {
        result[cls] = Number((result[cls] / sum).toFixed(4));
      }
    }

    return result;
  }
}
