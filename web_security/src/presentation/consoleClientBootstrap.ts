/**
 * Presentation Controller: consoleClientBootstrap.ts
 * Wires Clean Architecture Use Cases and Infrastructure Adapters to the Astro DOM.
 * 100% autonomous detection from real continuous video frames. Zero scenario forcing.
 */

import {
  CANONICAL_DETECTION_CLASSES,
  DETECTION_CLASS_CATALOG,
  type DetectionClass,
  type SecurityAlert
} from '../domain';
import {
  GetCameraGridUseCase,
  InferenceLoopController,
  MonitorActiveCameraUseCase,
  type ActiveInferenceResult,
  type LoopTelemetryMetrics
} from '../application';
import { HttpInferenceClient, WebRtcCameraAdapter } from '../infrastructure';

export function bootstrapSecurityConsole(): void {
  const videoEl = document.getElementById('cam01-video') as HTMLVideoElement | null;
  const canvasEl = document.getElementById('cam01-overlay-canvas') as HTMLCanvasElement | null;

  const inferenceClient = new HttpInferenceClient('http://localhost:8000', 15000);
  const cameraAdapter = new WebRtcCameraAdapter(videoEl, canvasEl);
  const monitorUseCase = new MonitorActiveCameraUseCase(cameraAdapter, inferenceClient, 'CAM-01');
  const gridUseCase = new GetCameraGridUseCase(inferenceClient);
  const loopController = new InferenceLoopController(monitorUseCase, 450);

  const headerDot = document.getElementById('header-backend-dot');
  const headerText = document.getElementById('header-backend-text');
  const hudSourceMode = document.getElementById('hud-source-mode');
  const hudPredictedClass = document.getElementById('hud-predicted-class');
  const hudSubjectGender = document.getElementById('hud-subject-gender');
  const hudFrameCount = document.getElementById('hud-frame-count');
  const hudEvidenceText = document.getElementById('hud-evidence-text');
  const cam01SourceDesc = document.getElementById('cam01-source-description');
  const cam01FpsCounter = document.getElementById('cam01-fps-counter');

  const readoutCard = document.getElementById('prediction-readout-card');
  const readoutReadableClass = document.getElementById('readout-readable-class');
  const readoutCanonicalKey = document.getElementById('readout-canonical-key');
  const readoutConfidencePct = document.getElementById('readout-confidence-pct');
  const readoutSubjectGender = document.getElementById('readout-subject-gender');
  const readoutThreatLevel = document.getElementById('readout-threat-level');
  const readoutLatencyMs = document.getElementById('readout-latency-ms');
  const readoutBackpressure = document.getElementById('readout-backpressure-stat');
  const readoutProtocolText = document.getElementById('readout-protocol-text');
  const telemetryModelName = document.getElementById('telemetry-model-name');

  const alertLogList = document.getElementById('security-alert-log');

  const updateTopologyAndHealthUI = async (targetCameraId?: string): Promise<void> => {
    const snapshot = await gridUseCase.execute();

    if (headerDot && headerText) {
      if (snapshot.backendHealth.online) {
        headerDot.className = 'engine-dot engine-dot-online';
        headerText.textContent = `Motor activo: ${snapshot.backendHealth.backend_engine} (${snapshot.backendHealth.device})`;
      } else {
        headerDot.className = 'engine-dot engine-dot-standby';
        headerText.textContent = snapshot.backendHealth.status_message;
      }
    }

    const totalEl = document.getElementById('summary-total-cameras');
    const activeEl = document.getElementById('summary-active-cameras');
    const disabledEl = document.getElementById('summary-disabled-cameras');
    if (totalEl) totalEl.textContent = String(snapshot.totalCount);
    if (activeEl) activeEl.textContent = String(snapshot.activeCount);
    if (disabledEl) disabledEl.textContent = String(snapshot.disabledCount);

    const nowTime = new Date().toLocaleTimeString('es-CO', { hour12: false });

    for (const node of snapshot.disabledServerNodes) {
      if (targetCameraId && node.camera_id !== targetCameraId) {
        continue;
      }
      const reasonEl = document.querySelector(`[data-reason-for="${node.camera_id}"]`);
      const endpointEl = document.querySelector(`[data-endpoint-for="${node.camera_id}"]`);
      const syncEl = document.querySelector(`[data-last-sync-for="${node.camera_id}"]`);

      if (reasonEl) {
        reasonEl.textContent = node.status_reason;
      }
      if (endpointEl) {
        endpointEl.textContent = node.stream_endpoint;
      }
      if (syncEl) {
        syncEl.textContent = `Verificado a las ${nowTime} (En espera)`;
      }
    }
  };

  const renderAlertHistory = (alerts: readonly SecurityAlert[]): void => {
    if (!alertLogList) {
      return;
    }

    if (alerts.length === 0) {
      alertLogList.innerHTML = `
        <li class="alert-empty-state">
          No se registran incidentes críticos. El sistema analiza continuamente el video de la cámara web física en busca de armas, agresiones o gestos de auxilio.
        </li>
      `;
      return;
    }

    alertLogList.innerHTML = alerts
      .slice(0, 8)
      .map((alert) => {
        const timeFormatted = new Date(alert.timestamp).toLocaleTimeString('es-CO', {
          hour12: false
        });
        const badgeClass =
          alert.threat_level === 'CRITICAL'
            ? 'alert-item-critical'
            : 'alert-item-high';
        return `
          <li class="alert-log-item ${badgeClass}">
            <div class="alert-item-top">
              <span class="alert-item-title">${alert.readable_label}</span>
              <span class="alert-item-time">${timeFormatted}</span>
            </div>
            <div class="alert-item-meta">
              <span>Nodo: ${alert.camera_id}</span>
              <span>Sujeto: ${alert.subject_gender}</span>
              <span>Confianza: ${(alert.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="alert-item-summary">
              ${alert.summary}
            </div>
          </li>
        `;
      })
      .join('');
  };

  const updateInferenceUI = (
    result: ActiveInferenceResult,
    telemetry: LoopTelemetryMetrics
  ): void => {
    const { prediction, alertHistory } = result;
    const descriptor =
      DETECTION_CLASS_CATALOG[prediction.primary_event] ??
      DETECTION_CLASS_CATALOG.Escena_Normal;

    // Dominant gender string
    const dominantPerson = prediction.detected_persons.find(
      (p) => p.gender !== 'No_Determinado'
    );
    const genderDisplay = dominantPerson
      ? `${dominantPerson.gender} (${(dominantPerson.gender_confidence * 100).toFixed(0)}%)`
      : 'No_Determinado';

    if (hudPredictedClass) {
      hudPredictedClass.textContent = prediction.primary_event;
    }
    if (hudSubjectGender) {
      hudSubjectGender.textContent = genderDisplay;
    }
    if (hudFrameCount) {
      hudFrameCount.textContent = String(telemetry.completedFrames);
    }
    if (hudEvidenceText) {
      hudEvidenceText.textContent = `${prediction.evidence.description} [Regla: ${prediction.evidence.rule_triggered}]`;
    }
    if (cam01FpsCounter) {
      cam01FpsCounter.textContent = `Latencia: ${prediction.model_info.latency_ms.toFixed(0)} ms`;
    }

    if (readoutCard) {
      readoutCard.setAttribute('data-threat-level', prediction.threat_level);
    }
    if (readoutReadableClass) {
      readoutReadableClass.textContent = descriptor.readableLabel;
    }
    if (readoutCanonicalKey) {
      readoutCanonicalKey.textContent = prediction.primary_event;
    }
    if (readoutConfidencePct) {
      readoutConfidencePct.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
    }
    if (readoutSubjectGender) {
      readoutSubjectGender.textContent = genderDisplay;
    }
    if (readoutThreatLevel) {
      const threatMap: Record<string, string> = {
        LOW: 'Bajo (Escena segura)',
        MEDIUM: 'Medio (Precaución)',
        HIGH: 'Alto (Intervención requerida)',
        CRITICAL: 'Crítico (Respuesta inmediata)'
      };
      readoutThreatLevel.textContent = threatMap[prediction.threat_level] ?? prediction.threat_level;
    }
    if (readoutLatencyMs) {
      readoutLatencyMs.textContent = `${prediction.model_info.latency_ms.toFixed(1)} ms`;
    }
    if (readoutBackpressure) {
      readoutBackpressure.textContent = `${telemetry.skippedBackpressureFrames} omitidos`;
    }
    if (readoutProtocolText) {
      readoutProtocolText.textContent = `${descriptor.operationalProtocol} ${prediction.evidence.description}`;
    }
    if (telemetryModelName) {
      telemetryModelName.textContent = `${prediction.model_info.scene_model} | ${prediction.model_info.backend}`;
    }

    for (const clsKey of CANONICAL_DETECTION_CLASSES) {
      const rawProb = prediction.scene_probabilities[clsKey] ?? 0;
      const pct = Math.min(100, Math.max(0, rawProb * 100));
      const valueEl = document.querySelector(`[data-prob-value="${clsKey}"]`);
      const fillEl = document.querySelector(`[data-prob-fill="${clsKey}"]`) as HTMLElement | null;
      const rowEl = document.querySelector(`[data-prob-row="${clsKey}"]`);

      if (valueEl) {
        valueEl.textContent = `${pct.toFixed(1)}%`;
      }
      if (fillEl) {
        fillEl.style.width = `${pct.toFixed(1)}%`;
      }
      if (rowEl) {
        if (clsKey === prediction.primary_event) {
          rowEl.classList.add('prob-row-dominant');
        } else {
          rowEl.classList.remove('prob-row-dominant');
        }
      }
    }

    renderAlertHistory(alertHistory);
  };

  loopController.subscribe(updateInferenceUI);

  const startLocalCameraFlow = async (): Promise<void> => {
    const status = await monitorUseCase.initializeCamera();
    if (hudSourceMode) {
      hudSourceMode.textContent = status.hardware_available
        ? 'Cámara física WebRTC activa'
        : 'Dispositivo no disponible';
    }
    if (cam01SourceDesc) {
      cam01SourceDesc.textContent = status.description;
    }
    loopController.start();
  };

  // Wire buttons
  const startWebcamBtn = document.getElementById('btn-start-webcam');
  if (startWebcamBtn) {
    startWebcamBtn.addEventListener('click', () => {
      void startLocalCameraFlow();
    });
  }

  const heroActivateBtn = document.getElementById('hero-activate-cam-btn');
  if (heroActivateBtn) {
    heroActivateBtn.addEventListener('click', () => {
      const matrixSection = document.getElementById('matriz-camaras');
      matrixSection?.scrollIntoView({ behavior: 'smooth' });
      void startLocalCameraFlow();
    });
  }

  const inferOnceBtn = document.getElementById('btn-infer-once');
  if (inferOnceBtn) {
    inferOnceBtn.addEventListener('click', () => {
      void loopController.triggerSingleInference();
    });
  }

  const toggleLoopBtn = document.getElementById('btn-toggle-loop');
  if (toggleLoopBtn) {
    toggleLoopBtn.addEventListener('click', () => {
      const isCurrentlyRunning = toggleLoopBtn.getAttribute('data-running') === 'true';
      if (isCurrentlyRunning) {
        loopController.stop();
        toggleLoopBtn.setAttribute('data-running', 'false');
        toggleLoopBtn.textContent = 'Reanudar inferencia continua';
      } else {
        loopController.start();
        toggleLoopBtn.setAttribute('data-running', 'true');
        toggleLoopBtn.textContent = 'Pausar inferencia continua';
      }
    });
  }

  const verifyButtons = document.querySelectorAll<HTMLButtonElement>('[data-verify-camera]');
  verifyButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const camId = btn.getAttribute('data-verify-camera') ?? undefined;
      void updateTopologyAndHealthUI(camId);
    });
  });

  const syncAllBtn = document.getElementById('btn-sync-cameras');
  if (syncAllBtn) {
    syncAllBtn.addEventListener('click', () => {
      void updateTopologyAndHealthUI();
    });
  }

  const clearAlertsBtn = document.getElementById('btn-clear-alerts');
  if (clearAlertsBtn) {
    clearAlertsBtn.addEventListener('click', () => {
      monitorUseCase.clearAlertHistory();
      renderAlertHistory([]);
    });
  }

  // Initial boot sequence
  void updateTopologyAndHealthUI();
  void startLocalCameraFlow();
}
