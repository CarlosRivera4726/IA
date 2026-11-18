/**
 * Application Controller: InferenceLoopController
 * Implements Single-Flight Lock concurrency control to prevent network backpressure
 * when sending real-time video frames from Camera 01 to the inference backend.
 * Depends exclusively on Domain and Application contracts.
 */

import type { ActiveInferenceResult, MonitorActiveCameraUseCase } from './MonitorActiveCameraUseCase';

export interface LoopTelemetryMetrics {
  readonly isRunning: boolean;
  readonly isRequestInFlight: boolean;
  readonly completedFrames: number;
  readonly skippedBackpressureFrames: number;
  readonly lastLatencyMs: number;
}

export type InferenceResultListener = (
  result: ActiveInferenceResult,
  telemetry: LoopTelemetryMetrics
) => void;

export class InferenceLoopController {
  private readonly monitorUseCase: MonitorActiveCameraUseCase;
  private readonly intervalMs: number;
  private isRunning = false;
  private isRequestInFlight = false;
  private completedFrames = 0;
  private skippedBackpressureFrames = 0;
  private lastLatencyMs = 0;
  private timerHandle: ReturnType<typeof setInterval> | null = null;
  private readonly listeners: Set<InferenceResultListener> = new Set();

  constructor(monitorUseCase: MonitorActiveCameraUseCase, intervalMs = 420) {
    this.monitorUseCase = monitorUseCase;
    this.intervalMs = intervalMs;
  }

  public subscribe(listener: InferenceResultListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  public start(): void {
    if (this.isRunning) {
      return;
    }
    this.isRunning = true;
    void this.tick();
    this.timerHandle = setInterval(() => {
      void this.tick();
    }, this.intervalMs);
  }

  public stop(): void {
    this.isRunning = false;
    if (this.timerHandle !== null) {
      clearInterval(this.timerHandle);
      this.timerHandle = null;
    }
  }

  public async triggerSingleInference(): Promise<ActiveInferenceResult | null> {
    return this.tick();
  }

  public getTelemetry(): LoopTelemetryMetrics {
    return {
      isRunning: this.isRunning,
      isRequestInFlight: this.isRequestInFlight,
      completedFrames: this.completedFrames,
      skippedBackpressureFrames: this.skippedBackpressureFrames,
      lastLatencyMs: this.lastLatencyMs
    };
  }

  private async tick(): Promise<ActiveInferenceResult | null> {
    if (this.isRequestInFlight) {
      this.skippedBackpressureFrames += 1;
      return null;
    }

    this.isRequestInFlight = true;
    const startedAt = Date.now();

    try {
      const result = await this.monitorUseCase.executeFrameAnalysis();
      const elapsed = Date.now() - startedAt;
      this.lastLatencyMs = result?.prediction.model_info.latency_ms ?? elapsed;
      if (result) {
        this.completedFrames += 1;
        const snapshot = this.getTelemetry();
        for (const listener of this.listeners) {
          listener(result, snapshot);
        }
      }
      return result;
    } finally {
      this.isRequestInFlight = false;
    }
  }
}
