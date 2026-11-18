/**
 * Application Use Case: MonitorActiveCameraUseCase
 * Orchestrates real frame capture from Camera 01, inference via the neural network port,
 * overlay rendering, and security alert generation.
 * Depends exclusively on Domain layer contracts. Zero scenario forcing.
 */

import {
  DETECTION_CLASS_CATALOG,
  type ICameraStreamPort,
  type IInferenceApiPort,
  type SecurityAlert,
  type ThreatPrediction
} from '../domain';

export interface ActiveInferenceResult {
  readonly prediction: ThreatPrediction;
  readonly generatedAlert: SecurityAlert | null;
  readonly alertHistory: readonly SecurityAlert[];
}

export class MonitorActiveCameraUseCase {
  private readonly cameraPort: ICameraStreamPort;
  private readonly inferencePort: IInferenceApiPort;
  private readonly activeCameraId: string;
  private readonly maxAlertHistory: number;
  private alertHistory: SecurityAlert[] = [];

  constructor(
    cameraPort: ICameraStreamPort,
    inferencePort: IInferenceApiPort,
    activeCameraId = 'CAM-01',
    maxAlertHistory = 25
  ) {
    this.cameraPort = cameraPort;
    this.inferencePort = inferencePort;
    this.activeCameraId = activeCameraId;
    this.maxAlertHistory = maxAlertHistory;
  }

  public async initializeCamera(): Promise<{
    readonly active: boolean;
    readonly hardware_available: boolean;
    readonly description: string;
  }> {
    return this.cameraPort.startStream();
  }

  public stopCamera(): void {
    this.cameraPort.stopStream();
  }

  public async executeFrameAnalysis(): Promise<ActiveInferenceResult | null> {
    const framePayload = this.cameraPort.captureFrameForInference(this.activeCameraId);
    if (!framePayload) {
      return null;
    }

    const prediction = await this.inferencePort.inferFrame(framePayload);
    this.cameraPort.renderPredictionOverlay(prediction);

    const generatedAlert = this.evaluateAlertRule(prediction);
    if (generatedAlert) {
      this.alertHistory = [generatedAlert, ...this.alertHistory].slice(0, this.maxAlertHistory);
    }

    return {
      prediction,
      generatedAlert,
      alertHistory: this.getAlertHistory()
    };
  }

  public getAlertHistory(): readonly SecurityAlert[] {
    return [...this.alertHistory];
  }

  public clearAlertHistory(): void {
    this.alertHistory = [];
  }

  private evaluateAlertRule(prediction: ThreatPrediction): SecurityAlert | null {
    if (!prediction.is_threat && prediction.primary_event === 'Escena_Normal') {
      return null;
    }

    const descriptor =
      DETECTION_CLASS_CATALOG[prediction.primary_event] ??
      DETECTION_CLASS_CATALOG.Escena_Normal;

    const confidencePercent = (prediction.confidence * 100).toFixed(1);
    const dominantGender = prediction.detected_persons.find(
      (p) => p.gender !== 'No_Determinado'
    )?.gender ?? 'No_Determinado';

    const genderSuffix = dominantGender !== 'No_Determinado'
      ? ` Sujeto detectado: ${dominantGender}.`
      : '';

    return {
      id: `ALT-${Date.now()}-${Math.floor(prediction.confidence * 1000)}`,
      camera_id: prediction.camera_id,
      timestamp: prediction.timestamp,
      predicted_class: prediction.primary_event,
      readable_label: descriptor.readableLabel,
      confidence: prediction.confidence,
      threat_level: prediction.threat_level,
      subject_gender: dominantGender,
      summary: `${descriptor.readableLabel} (${confidencePercent}%).${genderSuffix} ${prediction.evidence.description}`
    };
  }
}
