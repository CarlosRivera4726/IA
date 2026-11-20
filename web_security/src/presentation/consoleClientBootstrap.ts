/**
 * Presentation Controller: consoleClientBootstrap.ts
 * Wires Clean Architecture Use Cases and Infrastructure Adapters to the Astro DOM.
 * 100% autonomous detection from real continuous video frames. Auto-connects on boot.
 */

import {
  CANONICAL_DETECTION_CLASSES,
  DETECTION_CLASS_CATALOG,
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

function formatGenderLabel(gender: string, confidence?: number): string {
  if (gender === 'No_Determinado' || !gender) {
    return 'Sin determinar';
  }
  if (typeof confidence === 'number' && confidence > 0) {
    return `${gender} (${(confidence * 100).toFixed(0)}%)`;
  }
  return gender;
}

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
          const shortDevice = (snapshot.backendHealth.device || 'GPU local')
            .replace(/^NVIDIA\s+GeForce\s+/i, '')
            .replace(/\s+Laptop\s+GPU$/i, '');
          headerText.textContent = `Motor en línea (${shortDevice})`;
        } else {
          headerDot.className = 'engine-dot engine-dot-standby';
          headerText.textContent = 'Motor en espera (:8000)';
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
          const text = `${node.status_reason} (${nowTime})`;
          reasonEl.textContent = text;
          reasonEl.setAttribute('title', text);
        }
        if (endpointEl) {
          endpointEl.textContent = node.stream_endpoint;
        }
      }
    } catch {
      if (headerDot && headerText) {
        headerDot.className = 'engine-dot engine-dot-standby';
        headerText.textContent = 'Reconectando servicio local';
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
          Sin incidentes registrados en esta sesión. Las alertas detectadas aparecerán aquí en orden cronológico.
        </li>
      `;
      return;
    }

    alertLogList.innerHTML = alerts
      .slice(0, 12)
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
        const genderText = formatGenderLabel(alert.subject_gender);
        return `
          <li class="alert-log-item ${badgeClass}">
            <div class="alert-item-top">
              <span class="alert-item-title">${alert.readable_label}</span>
              <span class="alert-item-time mono-stat">${timeFormatted}</span>
            </div>
            <div class="alert-item-meta">
              <span class="mono-stat">${alert.camera_id}</span>
              <span>${genderText}</span>
              <span class="mono-stat">${(alert.confidence * 100).toFixed(0)}%</span>
            </div>
            <p class="alert-item-summary">${alert.summary}</p>
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

    const dominantPerson = prediction.detected_persons.find(
      (p) => p.gender !== 'No_Determinado'
    );
    const genderDisplay = dominantPerson
      ? formatGenderLabel(dominantPerson.gender, dominantPerson.gender_confidence)
      : prediction.detected_persons.length > 0
        ? 'Persona detectada'
        : 'Sin detectar';

    if (globalThreatBadge && globalThreatText) {
      globalThreatBadge.setAttribute('data-threat', prediction.threat_level);
      if (prediction.threat_level === 'CRITICAL') {
        globalThreatText.textContent = 'Amenaza crítica';
      } else if (prediction.threat_level === 'HIGH') {
        globalThreatText.textContent = 'Alerta activa';
      } else if (prediction.threat_level === 'MEDIUM') {
        globalThreatText.textContent = 'Precaución';
      } else {
        globalThreatText.textContent = 'Normal';
      }
    }

    if (hudPredictedClass) {
      hudPredictedClass.textContent = descriptor.readableLabel;
      hudPredictedClass.setAttribute('data-threat', prediction.threat_level);
    }
    if (hudSubjectGender) {
      hudSubjectGender.textContent = genderDisplay;
    }
    if (hudFrameCount) {
      hudFrameCount.textContent = String(telemetry.completedFrames);
    }
    if (hudEvidenceText) {
      hudEvidenceText.textContent = prediction.evidence.description;
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
        LOW: 'Bajo',
        MEDIUM: 'Medio',
        HIGH: 'Alto',
        CRITICAL: 'Crítico'
      };
      readoutThreatLevel.textContent = threatMap[prediction.threat_level] ?? prediction.threat_level;
      readoutThreatLevel.setAttribute('data-threat', prediction.threat_level);
    }
    if (readoutLatencyMs) {
      readoutLatencyMs.textContent = `${prediction.model_info.latency_ms.toFixed(0)} ms`;
    }
    if (readoutBackpressure) {
      readoutBackpressure.textContent = String(telemetry.skippedBackpressureFrames);
    }
    if (readoutProtocolText) {
      readoutProtocolText.textContent = `${descriptor.operationalProtocol} ${prediction.evidence.description}`;
    }
    if (telemetryModelName) {
      telemetryModelName.textContent = `CAM-01 (${prediction.model_info.backend})`;
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
        rowEl.classList.toggle('prob-row-dominant', clsKey === prediction.primary_event);
      }
    }

    renderAlertHistory(alertHistory);
  };

  loopController.subscribe(updateInferenceUI);

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
        hudPredictedClass.textContent = 'Cámara no disponible o permiso denegado';
      }
      if (hudEvidenceText) {
        hudEvidenceText.textContent = 'Autorice el acceso a la cámara en el navegador y pulse Reiniciar.';
      }
    }
  };

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
    btn.addEventListener('click', async () => {
      const camId = btn.getAttribute('data-verify-camera') ?? undefined;
      const original = btn.textContent;
      btn.disabled = true;
      btn.textContent = 'Verificando...';
      try {
        await updateTopologyAndHealthUI(camId);
      } finally {
        btn.disabled = false;
        btn.textContent = original || 'Verificar';
      }
    });
  });

  const syncAllBtn = document.getElementById('btn-sync-cameras') as HTMLButtonElement | null;
  if (syncAllBtn) {
    syncAllBtn.addEventListener('click', async () => {
      const labelEl = syncAllBtn.querySelector('span');
      const original = labelEl?.textContent;
      syncAllBtn.disabled = true;
      if (labelEl) labelEl.textContent = 'Sincronizando...';
      try {
        await updateTopologyAndHealthUI();
      } finally {
        syncAllBtn.disabled = false;
        if (labelEl) labelEl.textContent = original || 'Sincronizar';
      }
    });
  }

  const clearAlertsBtn = document.getElementById('btn-clear-alerts');
  if (clearAlertsBtn) {
    clearAlertsBtn.addEventListener('click', () => {
      monitorUseCase.clearAlertHistory();
      renderAlertHistory([]);
    });
  }

  void updateTopologyAndHealthUI();
  void startLocalCameraFlow();

  setInterval(() => {
    void updateTopologyAndHealthUI();
  }, 6000);
}
