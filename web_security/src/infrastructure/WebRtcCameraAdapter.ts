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

    if (this.latestPrediction) {
      this.drawDetections(ctx, width, height, this.latestPrediction);
    }
  }

  private drawLabeledBox(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    boxW: number,
    boxH: number,
    color: string,
    caption: string
  ): void {
    // Marco principal fino y nítido
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, boxW, boxH);

    // Refuerzo sutil en vértices para lectura rápida sobre video en movimiento
    const tick = Math.min(10, boxW * 0.2, boxH * 0.2);
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(x, y + tick);
    ctx.lineTo(x, y);
    ctx.lineTo(x + tick, y);

    ctx.moveTo(x + boxW - tick, y);
    ctx.lineTo(x + boxW, y);
    ctx.lineTo(x + boxW, y + tick);

    ctx.moveTo(x, y + boxH - tick);
    ctx.lineTo(x, y + boxH);
    ctx.lineTo(x + tick, y + boxH);

    ctx.moveTo(x + boxW - tick, y + boxH);
    ctx.lineTo(x + boxW, y + boxH);
    ctx.lineTo(x + boxW, y + boxH - tick);
    ctx.stroke();

    // Etiqueta oscura de alto contraste con barra lateral de color semántico
    ctx.font = '500 11px "IBM Plex Sans", "IBM Plex Mono", sans-serif';
    const padX = 8;
    const tagH = 20;
    const textWidth = ctx.measureText(caption).width + padX * 2 + 3;
    const tagY = y >= tagH + 4 ? y - tagH - 2 : y + 4;
    const tagX = Math.max(4, Math.min(ctx.canvas.width - textWidth - 4, x));

    ctx.fillStyle = 'rgba(17, 20, 23, 0.88)';
    ctx.fillRect(tagX, tagY, textWidth, tagH);

    ctx.fillStyle = color;
    ctx.fillRect(tagX, tagY, 3, tagH);

    ctx.fillStyle = '#ECEAE4';
    ctx.fillText(caption, tagX + padX + 2, tagY + 13.5);
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

    // 1. Personas detectadas (backend entrega [xmin, ymin, xmax, ymax] normalizado)
    for (const person of prediction.detected_persons) {
      const [xMin, yMin, xMax, yMax] = person.bbox;
      const x = Math.max(4, xMin * width);
      const y = Math.max(4, yMin * height);
      const boxW = Math.max(12, Math.min(width - x - 4, (xMax - xMin) * width));
      const boxH = Math.max(12, Math.min(height - y - 4, (yMax - yMin) * height));

      const personColor = person.has_distress_arms_raised ? '#D99B26' : threatColor;
      const genderText = person.gender === 'No_Determinado'
        ? 'Persona'
        : `${person.gender} ${(person.gender_confidence * 100).toFixed(0)}%`;
      const flagText = person.has_distress_arms_raised ? ' (Auxilio)' : '';
      this.drawLabeledBox(ctx, x, y, boxW, boxH, personColor, `${genderText}${flagText}`);

      // Puntos articulares visibles con halo oscuro para contraste
      for (const kp of person.keypoints) {
        if (kp.confidence >= 0.5) {
          const kpX = kp.x * width;
          const kpY = kp.y * height;
          ctx.fillStyle = '#111417';
          ctx.beginPath();
          ctx.arc(kpX, kpY, 3.2, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = '#ECEAE4';
          ctx.beginPath();
          ctx.arc(kpX, kpY, 2.0, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // 2. Objetos detectados (solo armas: cuchillos, armas de fuego)
    for (const obj of prediction.detected_objects) {
      const isActualWeapon = ['knife', 'gun', 'pistol', 'arma', 'cuchillo', 'scissors'].some((w) =>
        obj.label.toLowerCase().includes(w)
      );
      if (!isActualWeapon) {
        continue;
      }

      const [xMin, yMin, xMax, yMax] = obj.bbox;
      const x = Math.max(4, xMin * width);
      const y = Math.max(4, yMin * height);
      const boxW = Math.max(12, Math.min(width - x - 4, (xMax - xMin) * width));
      const boxH = Math.max(12, Math.min(height - y - 4, (yMax - yMin) * height));

      const caption = `Arma (${obj.label}) ${(obj.confidence * 100).toFixed(0)}%`;
      this.drawLabeledBox(ctx, x, y, boxW, boxH, '#E03E2D', caption);
    }
  }
}
