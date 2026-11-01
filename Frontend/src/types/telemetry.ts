export type RiskLevel = 'NORMAL' | 'WARNING' | 'CRITICAL'

export interface Alert {
  id: string
  timestamp: string
  camera_id: string
  alert_type: string
  message: string
  risk_level: RiskLevel
  metadata?: Record<string, any>
}

export interface Track {
  track_id: number
  class_name: string
  bbox: [number, number, number, number]
  confidence: number
  speed?: number
  risk_score: number
}

export interface CameraTelemetry {
  camera_id: string
  camera_name: string
  timestamp: string
  frame_number: number
  scene_status: RiskLevel
  tracks: Track[]
  alerts: Alert[]
}

export interface TelemetryFrame {
  timestamp: string
  frame_id: number
  overall_risk: RiskLevel
  cameras: Record<string, CameraTelemetry>
  global_alerts: Alert[]
}

export type ConnectionStatus = 'CONNECTING' | 'CONNECTED' | 'DISCONNECTED' | 'RECONNECTING' | 'ERROR'

export interface CameraStream {
  camera_id: string
  camera_name: string
  status: 'ACTIVE' | 'OFFLINE' | 'CONNECTING' | 'CONFIGURED'
  width?: number
  height?: number
}

export interface ServerStatus {
  status: string
  processing_fps?: number
  latency_ms?: number
  server_time?: string
  telemetry_mode?: 'simulation' | 'live' | string
}
