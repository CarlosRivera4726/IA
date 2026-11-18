/**
 * Infrastructure Adapter: WebRtcCameraAdapter
 * Implements ICameraStreamPort for Camera 01 (local device webcam via getUserMedia).
 * Captures real physical camera frames for real-time inference and overlays
 * neural network bounding boxes and pose keypoints directly on the viewport canvas.
 * Zero synthetic images, zero forced scenarios.
 */

import type {
  CapturedFramePayload,
  ICameraStreamPort,
  ThreatPrediction
} from '../domain';

export class WebRtcCameraAdapter implements ICameraStreamPort {
  private readonly videoElement: HTMLVideoElement | null;
  private readonly overlayCanvas: HTMLCanvasElement | null;
  private readonly captureCanvas: HTMLCanvasElement;
  private mediaStream: MediaStream | null = null;
  private isStreaming = false;
  private hardwareAvailable = false;
  private animationFrameId: number | null = null;
  private latestPrediction: ThreatPrediction | null = null;

  constructor(
    videoElement: HTMLVideoElement | null,
    overlayCanvas: HTMLCanvasElement | null
  ) {
    this.videoElement = videoElement;
    this.overlayCanvas = overlayCanvas;
    this.captureCanvas = document.createElement('canvas');
    this.captureCanvas.width = 480;
    this.captureCanvas.height = 360;
  }

  public async startStream(): Promise<{
    readonly active: boolean;
    readonly hardware_available: boolean;
    readonly description: string;
  }> {
    this.stopRenderLoop();

    if (
      typeof navigator !== 'undefined' &&
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === 'function'
    ) {
      try {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          },
          audio: false
        });

        if (this.videoElement) {
          this.videoElement.srcObject = this.mediaStream;
          await this.videoElement.play();
        }

        this.hardwareAvailable = true;
        this.isStreaming = true;
        this.startRenderLoop();

        return {
          active: true,
          hardware_available: true,
          description: 'Cámara física local conectada en vivo mediante WebRTC (getUserMedia).'
        };
      } catch (err) {
        this.hardwareAvailable = false;
        this.isStreaming = false;
        const msg = err instanceof Error ? err.message : String(err);
        return {
          active: false,
          hardware_available: false,
          description: `Dispositivo de video no accesible o permiso denegado: ${msg}`
        };
      }
    }

    this.hardwareAvailable = false;
    this.isStreaming = false;
    return {
      active: false,
      hardware_available: false,
      description: 'El navegador no dispone de soporte para la API MediaDevices/getUserMedia.'
    };
  }

  public stopStream(): void {
    this.isStreaming = false;
    this.stopRenderLoop();

    if (this.mediaStream) {
      for (const track of this.mediaStream.getTracks()) {
        track.stop();
      }
      this.mediaStream = null;
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }
  }

  public captureFrameForInference(cameraId: string): CapturedFramePayload | null {
    if (!this.isStreaming || !this.videoElement || this.videoElement.readyState < 2) {
      return null;
    }

    const ctx = this.captureCanvas.getContext('2d');
    if (!ctx) {
      return null;
    }

    const width = this.captureCanvas.width;
    const height = this.captureCanvas.height;

    ctx.drawImage(this.videoElement, 0, 0, width, height);

    const imageBase64 = this.captureCanvas.toDataURL('image/jpeg', 0.85);

    return {
      camera_id: cameraId,
      image_base64: imageBase64,
      frame_ts: Date.now(),
      captured_at_iso: new Date().toISOString(),
      width,
      height
    };
  }

  public renderPredictionOverlay(prediction: ThreatPrediction): void {
    this.latestPrediction = prediction;
  }

  private startRenderLoop(): void {
    const renderStep = () => {
      if (!this.isStreaming) {
        return;
      }
      this.paintViewportOverlay();
      this.animationFrameId = window.requestAnimationFrame(renderStep);
    };
    this.animationFrameId = window.requestAnimationFrame(renderStep);
  }

  private stopRenderLoop(): void {
    if (this.animationFrameId !== null) {
      window.cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  private paintViewportOverlay(): void {
    if (!this.overlayCanvas) {
      return;
    }
    const ctx = this.overlayCanvas.getContext('2d');
    if (!ctx) {
      return;
    }

    const width = this.overlayCanvas.width;
    const height = this.overlayCanvas.height;
    ctx.clearRect(0, 0, width, height);

    this.drawReticleGrid(ctx, width, height);

    if (this.latestPrediction) {
      this.drawDetections(ctx, width, height, this.latestPrediction);
    }
  }

  private drawReticleGrid(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number
  ): void {
    ctx.strokeStyle = 'rgba(236, 234, 228, 0.15)';
    ctx.lineWidth = 1;
    const corner = 18;

    // Corner calibration brackets
    ctx.beginPath();
    ctx.moveTo(10, 10 + corner);
    ctx.lineTo(10, 10);
    ctx.lineTo(10 + corner, 10);

    ctx.moveTo(width - 10 - corner, 10);
    ctx.lineTo(width - 10, 10);
    ctx.lineTo(width - 10, 10 + corner);

    ctx.moveTo(10, height - 10 - corner);
    ctx.lineTo(10, height - 10);
    ctx.lineTo(10 + corner, height - 10);

    ctx.moveTo(width - 10 - corner, height - 10);
    ctx.lineTo(width - 10, height - 10);
    ctx.lineTo(width - 10, height - 10 - corner);
    ctx.stroke();
  }

  private drawDetections(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    prediction: ThreatPrediction
  ): void {
    const isCritical = prediction.threat_level === 'CRITICAL';
    const isHigh = prediction.threat_level === 'HIGH';
    const threatColor = isCritical
      ? '#E03E2D'
      : isHigh
        ? '#D99B26'
        : '#289E68';

    // 1. Draw detected persons
    for (const person of prediction.detected_persons) {
      const [yMin, xMin, yMax, xMax] = person.bbox;
      const x = Math.max(8, xMin * width);
      const y = Math.max(26, yMin * height);
      const boxW = Math.min(width - x - 8, (xMax - xMin) * width);
      const boxH = Math.min(height - y - 8, (yMax - yMin) * height);

      const personColor = person.has_distress_arms_raised ? '#D99B26' : threatColor;
      ctx.strokeStyle = personColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, boxW, boxH);

      const genderText = person.gender === 'No_Determinado'
        ? 'Persona'
        : `Persona (${person.gender} ${(person.gender_confidence * 100).toFixed(0)}%)`;
      const flagText = person.has_distress_arms_raised ? ' [Brazos arriba]' : '';
      const caption = `${genderText}${flagText}`;

      ctx.font = '500 11px "IBM Plex Mono", monospace';
      const textWidth = ctx.measureText(caption).width + 12;

      ctx.fillStyle = personColor;
      ctx.fillRect(x, Math.max(4, y - 20), textWidth, 20);

      ctx.fillStyle = '#111417';
      ctx.fillText(caption, x + 6, Math.max(18, y - 6));

      // Draw keypoints if available
      for (const kp of person.keypoints) {
        if (kp.confidence >= 0.4) {
          const kpX = kp.x * width;
          const kpY = kp.y * height;
          ctx.fillStyle = '#ECEAE4';
          ctx.beginPath();
          ctx.arc(kpX, kpY, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // 2. Draw detected objects (only actual weapons: knives, firearms)
    for (const obj of prediction.detected_objects) {
      const isActualWeapon = ['knife', 'gun', 'pistol', 'arma', 'cuchillo', 'scissors'].some((w) =>
        obj.label.toLowerCase().includes(w)
      );
      if (!isActualWeapon) {
        continue;
      }

      const [yMin, xMin, yMax, xMax] = obj.bbox;
      const x = Math.max(8, xMin * width);
      const y = Math.max(26, yMin * height);
      const boxW = Math.min(width - x - 8, (xMax - xMin) * width);
      const boxH = Math.min(height - y - 8, (yMax - yMin) * height);

      ctx.strokeStyle = '#E03E2D';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, boxW, boxH);

      const caption = `Arma: ${obj.label} ${(obj.confidence * 100).toFixed(0)}%`;
      ctx.font = '500 11px "IBM Plex Mono", monospace';
      const textWidth = ctx.measureText(caption).width + 12;

      ctx.fillStyle = '#E03E2D';
      ctx.fillRect(x, Math.max(4, y - 20), textWidth, 20);

      ctx.fillStyle = '#111417';
      ctx.fillText(caption, x + 6, Math.max(18, y - 6));
    }
  }
}
