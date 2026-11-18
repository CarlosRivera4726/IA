/**
 * Core Domain Entities and Value Objects for SentinelCV Security System.
 * Pure TypeScript module with zero external or runtime platform dependencies.
 * Clean Architecture compliant.
 */

export type DetectionClass =
  | 'Arma_Blanca'
  | 'Arma_De_Fuego'
  | 'Ataque_Agresion'
  | 'Escena_Normal'
  | 'Hurto_Sospechoso'
  | 'Senal_Auxilio_Hombre'
  | 'Senal_Auxilio_Mujer';

export const CANONICAL_DETECTION_CLASSES: readonly DetectionClass[] = [
  'Arma_Blanca',
  'Arma_De_Fuego',
  'Ataque_Agresion',
  'Escena_Normal',
  'Hurto_Sospechoso',
  'Senal_Auxilio_Hombre',
  'Senal_Auxilio_Mujer'
] as const;

export type ThreatLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export type SubjectGender = 'Hombre' | 'Mujer' | 'No_Determinado';

export type CameraStatus = 'ACTIVE' | 'TEMPORARILY_DISABLED' | 'ERROR';

export interface DetectionClassDescriptor {
  readonly key: DetectionClass;
  readonly readableLabel: string;
  readonly defaultThreatLevel: ThreatLevel;
  readonly defaultGender: SubjectGender;
  readonly categoryGroup: string;
  readonly operationalProtocol: string;
}

export const DETECTION_CLASS_CATALOG: Record<DetectionClass, DetectionClassDescriptor> = {
  Arma_Blanca: {
    key: 'Arma_Blanca',
    readableLabel: 'Agresión o amenaza con arma blanca',
    defaultThreatLevel: 'CRITICAL',
    defaultGender: 'No_Determinado',
    categoryGroup: 'Armas y amenazas letales',
    operationalProtocol: 'Alerta prioritaria inmediata. Notificar al cuadrante de vigilancia táctica.'
  },
  Arma_De_Fuego: {
    key: 'Arma_De_Fuego',
    readableLabel: 'Porte o encañonamiento con arma de fuego',
    defaultThreatLevel: 'CRITICAL',
    defaultGender: 'No_Determinado',
    categoryGroup: 'Armas y amenazas letales',
    operationalProtocol: 'Código rojo de seguridad. Activar protocolo balístico y avisar a autoridades policiales.'
  },
  Ataque_Agresion: {
    key: 'Ataque_Agresion',
    readableLabel: 'Ataque o agresión física activa',
    defaultThreatLevel: 'HIGH',
    defaultGender: 'No_Determinado',
    categoryGroup: 'Integridad personal',
    operationalProtocol: 'Despachar equipo de contención y registrar segmento forense de video.'
  },
  Escena_Normal: {
    key: 'Escena_Normal',
    readableLabel: 'Escena normal sin incidentes',
    defaultThreatLevel: 'LOW',
    defaultGender: 'No_Determinado',
    categoryGroup: 'Supervisión de rutina',
    operationalProtocol: 'Supervisión estándar continua en circuito cerrado.'
  },
  Hurto_Sospechoso: {
    key: 'Hurto_Sospechoso',
    readableLabel: 'Hurto o sustracción sospechosa',
    defaultThreatLevel: 'HIGH',
    defaultGender: 'No_Determinado',
    categoryGroup: 'Custodia de bienes',
    operationalProtocol: 'Fijar seguimiento visual continuo del sujeto en la grilla y notificar a seguridad.'
  },
  Senal_Auxilio_Hombre: {
    key: 'Senal_Auxilio_Hombre',
    readableLabel: 'Señal de auxilio a la cámara (Hombre)',
    defaultThreatLevel: 'HIGH',
    defaultGender: 'Hombre',
    categoryGroup: 'Auxilio y rescate',
    operationalProtocol: 'Verificar estado del sujeto masculino en el perímetro y enviar asistencia inmediata.'
  },
  Senal_Auxilio_Mujer: {
    key: 'Senal_Auxilio_Mujer',
    readableLabel: 'Señal de auxilio a la cámara (Mujer)',
    defaultThreatLevel: 'HIGH',
    defaultGender: 'Mujer',
    categoryGroup: 'Auxilio y rescate',
    operationalProtocol: 'Activar código violeta de atención prioritaria a mujer en situación de riesgo.'
  }
};

export interface DetectedKeypoint {
  readonly name: string;
  readonly x: number;
  readonly y: number;
  readonly confidence: number;
}

export interface DetectedPerson {
  readonly person_id: number;
  readonly bbox: readonly [number, number, number, number];
  readonly gender: SubjectGender;
  readonly gender_confidence: number;
  readonly has_distress_arms_raised: boolean;
  readonly keypoints: readonly DetectedKeypoint[];
}

export interface DetectedObject {
  readonly label: string;
  readonly confidence: number;
  readonly bbox: readonly [number, number, number, number];
}

export interface DetectedEventEvidence {
  readonly description: string;
  readonly rule_triggered: string;
  readonly sustained_duration_sec: number;
}

export type ClassProbabilities = Readonly<Record<DetectionClass, number>>;

export interface ThreatPrediction {
  readonly camera_id: string;
  readonly timestamp: string;
  readonly primary_event: DetectionClass;
  readonly threat_level: ThreatLevel;
  readonly is_threat: boolean;
  readonly confidence: number;
  readonly scene_probabilities: ClassProbabilities;
  readonly detected_persons: readonly DetectedPerson[];
  readonly detected_objects: readonly DetectedObject[];
  readonly evidence: DetectedEventEvidence;
  readonly model_info: {
    readonly scene_model: string;
    readonly gender_model: string;
    readonly pose_model: string;
    readonly object_model: string;
    readonly backend: string;
    readonly latency_ms: number;
  };
}

export interface CameraNode {
  readonly camera_id: string;
  readonly label: string;
  readonly name?: string;
  readonly location_zone: string;
  readonly location?: string;
  readonly status: CameraStatus;
  readonly is_active: boolean;
  readonly connection_mode: string;
  readonly stream_endpoint: string;
  readonly stream_url?: string;
  readonly resolution: string;
  readonly status_reason: string;
  readonly description?: string;
}

export interface SecurityAlert {
  readonly id: string;
  readonly camera_id: string;
  readonly timestamp: string;
  readonly predicted_class: DetectionClass;
  readonly readable_label: string;
  readonly confidence: number;
  readonly threat_level: ThreatLevel;
  readonly subject_gender: SubjectGender;
  readonly summary: string;
}

export const DEFAULT_SIX_CAMERA_NODES: readonly CameraNode[] = [
  {
    camera_id: 'CAM-01',
    label: 'Cámara del dispositivo operador',
    name: 'Cámara del dispositivo operador',
    location_zone: 'Consola principal de monitoreo',
    location: 'Consola principal de monitoreo',
    status: 'ACTIVE',
    is_active: true,
    connection_mode: 'LOCAL_DEVICE_WEBCAM',
    stream_endpoint: 'webcam://localhost:user_media',
    stream_url: 'webcam://localhost:user_media',
    resolution: '640x480 (30 FPS)',
    status_reason: 'Transmisión activa analizada autónomamente en tiempo real.',
    description: 'Transmisión activa analizada autónomamente en tiempo real.'
  },
  {
    camera_id: 'CAM-02',
    label: 'Acceso peatonal norte',
    name: 'Acceso peatonal norte',
    location_zone: 'Torniquetes de ingreso exterior',
    location: 'Torniquetes de ingreso exterior',
    status: 'TEMPORARILY_DISABLED',
    is_active: false,
    connection_mode: 'SERVER_RTSP_NODE',
    stream_endpoint: 'rtsp://10.24.0.102:554/stream1',
    stream_url: 'rtsp://10.24.0.102:554/stream1',
    resolution: '1920x1080 (30 FPS)',
    status_reason: 'Canal de servidor central temporalmente deshabilitado en espera de enlace.',
    description: 'Canal de servidor central temporalmente deshabilitado en espera de enlace.'
  },
  {
    camera_id: 'CAM-03',
    label: 'Bóveda y caja principal',
    name: 'Bóveda y caja principal',
    location_zone: 'Sector de custodia de valores',
    location: 'Sector de custodia de valores',
    status: 'TEMPORARILY_DISABLED',
    is_active: false,
    connection_mode: 'SERVER_RTSP_NODE',
    stream_endpoint: 'rtsp://10.24.0.103:554/stream1',
    stream_url: 'rtsp://10.24.0.103:554/stream1',
    resolution: '1920x1080 (30 FPS)',
    status_reason: 'Canal de servidor central temporalmente deshabilitado en espera de enlace.',
    description: 'Canal de servidor central temporalmente deshabilitado en espera de enlace.'
  },
  {
    camera_id: 'CAM-04',
    label: 'Pasillo de carga oriental',
    name: 'Pasillo de carga oriental',
    location_zone: 'Muelle logístico 02',
    location: 'Muelle logístico 02',
    status: 'TEMPORARILY_DISABLED',
    is_active: false,
    connection_mode: 'SERVER_RTSP_NODE',
    stream_endpoint: 'rtsp://10.24.0.104:554/stream1',
    stream_url: 'rtsp://10.24.0.104:554/stream1',
    resolution: '1280x720 (25 FPS)',
    status_reason: 'Canal de servidor central temporalmente deshabilitado en espera de enlace.',
    description: 'Canal de servidor central temporalmente deshabilitado en espera de enlace.'
  },
  {
    camera_id: 'CAM-05',
    label: 'Estacionamiento subterráneo',
    name: 'Estacionamiento subterráneo',
    location_zone: 'Nivel -1 bahías 12 a 24',
    location: 'Nivel -1 bahías 12 a 24',
    status: 'TEMPORARILY_DISABLED',
    is_active: false,
    connection_mode: 'SERVER_RTSP_NODE',
    stream_endpoint: 'rtsp://10.24.0.105:554/stream1',
    stream_url: 'rtsp://10.24.0.105:554/stream1',
    resolution: '1280x720 (20 FPS)',
    status_reason: 'Canal de servidor central temporalmente deshabilitado en espera de enlace.',
    description: 'Canal de servidor central temporalmente deshabilitado en espera de enlace.'
  },
  {
    camera_id: 'CAM-06',
    label: 'Galería comercial nivel 2',
    name: 'Galería comercial nivel 2',
    location_zone: 'Corredor central de locales',
    location: 'Corredor central de locales',
    status: 'TEMPORARILY_DISABLED',
    is_active: false,
    connection_mode: 'SERVER_RTSP_NODE',
    stream_endpoint: 'rtsp://10.24.0.106:554/stream1',
    stream_url: 'rtsp://10.24.0.106:554/stream1',
    resolution: '1920x1080 (30 FPS)',
    status_reason: 'Canal de servidor central temporalmente deshabilitado en espera de enlace.',
    description: 'Canal de servidor central temporalmente deshabilitado en espera de enlace.'
  }
];
