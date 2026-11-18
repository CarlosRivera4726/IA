/**
 * Application Use Case: GetCameraGridUseCase
 * Retrieves and normalizes the 6-camera surveillance topology:
 * - CAM-01: Active local device camera
 * - CAM-02 to CAM-06: Server nodes temporarily disabled
 * Depends exclusively on Domain layer contracts.
 */

import {
  DEFAULT_SIX_CAMERA_NODES,
  type BackendHealthStatus,
  type CameraNode,
  type IInferenceApiPort
} from '../domain';

export interface CameraGridSnapshot {
  readonly nodes: readonly CameraNode[];
  readonly activeNode: CameraNode;
  readonly disabledServerNodes: readonly CameraNode[];
  readonly totalCount: number;
  readonly activeCount: number;
  readonly disabledCount: number;
  readonly backendHealth: BackendHealthStatus;
}

export class GetCameraGridUseCase {
  private readonly inferencePort: IInferenceApiPort;

  constructor(inferencePort: IInferenceApiPort) {
    this.inferencePort = inferencePort;
  }

  public async execute(): Promise<CameraGridSnapshot> {
    const [backendHealth, remoteNodes] = await Promise.all([
      this.inferencePort.checkHealth(),
      this.inferencePort.getCameraNodes()
    ]);

    const normalizedNodes = this.normalizeSixCameraTopology(remoteNodes);
    const activeNode =
      normalizedNodes.find((node) => node.camera_id === 'CAM-01') ??
      DEFAULT_SIX_CAMERA_NODES[0];
    const disabledServerNodes = normalizedNodes.filter(
      (node) => node.camera_id !== 'CAM-01'
    );

    return {
      nodes: normalizedNodes,
      activeNode,
      disabledServerNodes,
      totalCount: normalizedNodes.length,
      activeCount: normalizedNodes.filter((n) => n.is_active).length,
      disabledCount: normalizedNodes.filter((n) => !n.is_active).length,
      backendHealth
    };
  }

  private normalizeSixCameraTopology(
    candidates: readonly CameraNode[]
  ): readonly CameraNode[] {
    if (!candidates || candidates.length === 0) {
      return DEFAULT_SIX_CAMERA_NODES;
    }

    return DEFAULT_SIX_CAMERA_NODES.map((defaultNode) => {
      const matched = candidates.find(
        (candidate) => candidate.camera_id === defaultNode.camera_id
      );
      if (!matched) {
        return defaultNode;
      }

      if (defaultNode.camera_id === 'CAM-01') {
        return {
          ...defaultNode,
          ...matched,
          status: 'ACTIVE',
          is_active: true,
          connection_mode: 'LOCAL_DEVICE_WEBCAM'
        };
      }

      return {
        ...defaultNode,
        ...matched,
        status: 'TEMPORARILY_DISABLED',
        is_active: false,
        connection_mode: 'SERVER_RTSP_NODE'
      };
    });
  }
}
