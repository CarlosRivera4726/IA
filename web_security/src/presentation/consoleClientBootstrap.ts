/**
 * Presentation Controller: consoleClientBootstrap.ts
 * Wires Clean Architecture Use Cases and Infrastructure Adapters to the Astro DOM.
 * 100% autonomous detection from real continuous video frames. Auto-connects on boot.
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
  const globalThreatBadge = document.getElementById('soc-global-threat-badge');
  const globalThreatText = document.getElementById('soc-threat-text');
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
    try {
      const snapshot = await gridUseCase.execute();

      if (headerDot && headerText) {
        if (snapshot.backendHealth.online) {
          headerDot.className = 'engine-dot engine-dot-online';
          headerText.textContent = `IA CUDA: ONLINE (${snapshot.backendHealth.device || 'RTX 5050'})`;
        } else {
          headerDot.className = 'engine-dot engine-dot-standby';
          headerText.textContent = 'IA :8000 EN ESPERA';
        }
      }

      const activeEl = document.getElementById('summary-active-cameras');
      if (activeEl) activeEl.textContent = String(snapshot.activeCount);

      const nowTime = new Date().toLocaleTimeString('es-CO', { hour12: false });

      for (const node of snapshot.disabledServerNodes) {
        if (targetCameraId && node.camera_id !== targetCameraId) {
          continue;
        }
        const reasonEl = document.querySelector(`[data-reason-for="${node.camera_id}"]`);
        const endpointEl = document.querySelector(`[data-endpoint-for="${node.camera_id}"]`);

        if (reasonEl) {
          reasonEl.textContent = `${node.status_reason} (${nowTime})`;
        }
        if (endpointEl) {
          endpointEl.textContent = node.stream_endpoint;
        }
      }
    } catch {
      if (headerDot && headerText) {
        headerDot.className = 'engine-dot engine-dot-standby';
        headerText.textContent = 'RECONECTANDO :8000...';
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
          Supervisión activa sin incidentes críticos. Las alertas aparecerán aquí automáticamente.
        </li>
      `;
      return;
    }

    alertLogList.innerHTML = alerts
      .slice(0, 10)
      .map((alert) => {
        const timeFormatted = new Date(alert.timestamp).toLocaleTimeString('es-CO', {
          hour12: false
        });
        const badgeClass =
          alert.threat_level === 'CRITICAL'
            ? 'alert-item-critical'
            : alert.threat_level === 'HIGH'
              ? 'alert-item-high'
              : 'alert-item-low';
        return `
          <li class="alert-log-item ${badgeClass}">
            <div class="alert-item-top">
              <span class="alert-item-title">${alert.readable_label}</span>
              <span class="alert-item-time mono-stat">${timeFormatted}</span>
            </div>
            <div class="alert-item-meta mono-stat">
              <span>${alert.camera_id}</span>
              <span>${alert.subject_gender}</span>
              <span>${(alert.confidence * 100).toFixed(0)}%</span>
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

    // Top Bar Global Threat Badge
    if (globalThreatBadge && globalThreatText) {
      globalThreatBadge.setAttribute('data-threat', prediction.threat_level);
      if (prediction.threat_level === 'CRITICAL') {
        globalThreatText.textContent = 'AMENAZA CRÍTICA';
      } else if (prediction.threat_level === 'HIGH') {
        globalThreatText.textContent = 'ALERTA DE SEGURIDAD';
      } else {
        globalThreatText.textContent = 'SISTEMA NORMAL';
      }
    }

    if (hudPredictedClass) {
      hudPredictedClass.textContent = descriptor.readableLabel;
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
      cam01FpsCounter.textContent = `${prediction.model_info.latency_ms.toFixed(0)} ms`;
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
        LOW: 'Bajo (Normal)',
        MEDIUM: 'Medio (Precaución)',
        HIGH: 'Alto (Intervención)',
        CRITICAL: 'Crítico (Emergencia)'
      };
      readoutThreatLevel.textContent = threatMap[prediction.threat_level] ?? prediction.threat_level;
    }
    if (readoutLatencyMs) {
      readoutLatencyMs.textContent = `${prediction.model_info.latency_ms.toFixed(0)} ms`;
    }
    if (readoutBackpressure) {
      readoutBackpressure.textContent = `${telemetry.skippedBackpressureFrames} omitidos`;
    }
    if (readoutProtocolText) {
      readoutProtocolText.textContent = `${descriptor.operationalProtocol} ${prediction.evidence.description}`;
    }
    if (telemetryModelName) {
      telemetryModelName.textContent = `${prediction.model_info.scene_model} + Torchvision`;
    }

    // Probability bars
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

  // Conexión automática de cámara física
  const startLocalCameraFlow = async (): Promise<void> => {
    try {
      const status = await monitorUseCase.initializeCamera();
      if (cam01SourceDesc) {
        cam01SourceDesc.textContent = status.description;
      }
      if (status.active) {
        loopController.start();
      }
    } catch {
      if (hudPredictedClass) {
        hudPredictedClass.textContent = 'Acceso a cámara denegado o no disponible';
      }
    }
  };

  // Botones auxiliares
  const startWebcamBtn = document.getElementById('btn-start-webcam');
  if (startWebcamBtn) {
    startWebcamBtn.addEventListener('click', () => {
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
        toggleLoopBtn.textContent = 'Reanudar';
      } else {
        loopController.start();
        toggleLoopBtn.setAttribute('data-running', 'true');
        toggleLoopBtn.textContent = 'Pausar';
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

  // Auto-boot sequence
  void updateTopologyAndHealthUI();
  void startLocalCameraFlow();

  // Sondeo periódico automático de salud del servidor (cada 6 segundos)
  setInterval(() => {
    void updateTopologyAndHealthUI();
  }, 6000);
}
