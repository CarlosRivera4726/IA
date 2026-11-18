/**
 * Domain Ports (Interfaces) defining contracts between Application Use Cases
 * and Infrastructure Adapters. Zero platform/browser coupling.
 * Clean Architecture compliant.
 */

import type {
  CameraNode,
  ThreatPrediction
} from './entities';

export interface CapturedFramePayload {
  readonly camera_id: string;
  readonly image_base64: string;
  readonly frame_ts: number;
  readonly captured_at_iso: string;
  readonly width: number;
  readonly height: number;
}

export interface BackendHealthStatus {
  readonly online: boolean;
  readonly model_loaded: boolean;
  readonly scene_model_loaded: boolean;
  readonly gender_model_loaded: boolean;
  readonly pose_model_loaded: boolean;
  readonly weapon_detector_loaded: boolean;
  readonly backend_engine: string;
  readonly device: string;
  readonly endpoint_url: string;
  readonly status_message: string;
}

export interface ICameraStreamPort {
  startStream(): Promise<{
    readonly active: boolean;
    readonly hardware_available: boolean;
    readonly description: string;
  }>;
  stopStream(): void;
  captureFrameForInference(cameraId: string): CapturedFramePayload | null;
  renderPredictionOverlay(prediction: ThreatPrediction): void;
}

export interface IInferenceApiPort {
  checkHealth(): Promise<BackendHealthStatus>;
  getCameraNodes(): Promise<readonly CameraNode[]>;
  inferFrame(payload: CapturedFramePayload): Promise<ThreatPrediction>;
}
