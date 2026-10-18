import { useCallback, useEffect, useRef, useState } from 'react'
import type { Alert, CameraTelemetry, ConnectionStatus, RiskLevel, TelemetryFrame, Track } from '../types/telemetry'

const MAX_ALERTS = 50

function riskFrom(value: unknown): RiskLevel {
  const text = String(value ?? '').toUpperCase()
  if (['CRITICAL', 'INTRUSION', 'BREACH_DETECTED', 'KNOWN_INTRUDER', 'UNAUTHORIZED_PERSONNEL'].includes(text)) return 'CRITICAL'
  if (['WARNING', 'HIGH_ACTIVITY', 'ACTIVE', 'SUSPICIOUS_MOTION', 'INTERACTION', 'ISOLATED', 'HIGH_DENSITY'].includes(text)) return 'WARNING'
  return 'NORMAL'
}

function normalizeAlert(raw: any, index: number, fallbackTime: string, cameraId = 'system'): Alert {
  const rawType = String(raw?.alert_type ?? raw?.type ?? 'SECURITY_EVENT')
  const severity = riskFrom(raw?.risk_level ?? raw?.threat ?? rawType)
  const message = String(raw?.message ?? raw?.threat ?? raw?.plate ?? rawType.replace(/_/g, ' '))
  return {
    id: String(raw?.id ?? `${cameraId}-${rawType}-${raw?.timestamp ?? fallbackTime}-${index}`),
    timestamp: String(raw?.timestamp ?? fallbackTime),
    camera_id: String(raw?.camera_id ?? raw?.camera ?? cameraId),
    alert_type: rawType,
    message,
    risk_level: severity,
    metadata: raw?.metadata ?? raw,
  }
}

function normalizeTrack(raw: any, index: number): Track | null {
  const bbox = raw?.bbox
  if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every((v: unknown) => Number.isFinite(Number(v)))) return null
  const parsedId = Number(raw?.track_id ?? raw?.id ?? index)
  const riskScore = Number(raw?.risk_score ?? raw?.risk ?? (Number(raw?.confidence ?? 0) * 100))
  return {
    track_id: Number.isFinite(parsedId) ? parsedId : index,
    class_name: String(raw?.class_name ?? raw?.class ?? 'person'),
    bbox: bbox.map(Number) as Track['bbox'],
    confidence: Number(raw?.confidence ?? 0),
    speed: raw?.speed == null ? undefined : Number(raw.speed),
    risk_score: Number.isFinite(riskScore) ? riskScore : 0,
  }
}

export function normalizeTelemetry(raw: any): TelemetryFrame | null {
  if (!raw || typeof raw !== 'object') return null
  const timestamp = typeof raw.timestamp === 'string' ? raw.timestamp : new Date(Number(raw.timestamp) * 1000 || Date.now()).toISOString()
  const rawCameras = raw.cameras ?? raw.streams ?? (raw.camera_id ? { [raw.camera_id]: raw } : {})
  if (!rawCameras || typeof rawCameras !== 'object') return null
  const globalAlerts: Alert[] = (Array.isArray(raw.global_alerts) ? raw.global_alerts : Array.isArray(raw.alerts) ? raw.alerts : [])
    .map((a: any, i: number) => normalizeAlert(a, i, timestamp))
  const cameras: Record<string, CameraTelemetry> = {}
  for (const [id, value] of Object.entries(rawCameras as Record<string, any>)) {
    const camera = value ?? {}
    const cameraAlerts = (Array.isArray(camera.alerts) ? camera.alerts : []).map((a: any, i: number) => normalizeAlert(a, i, timestamp, id))
    const tracks = (Array.isArray(camera.tracks) ? camera.tracks : []).map(normalizeTrack).filter(Boolean) as Track[]
    const sceneStatus = riskFrom(camera.scene_status)
    cameras[id] = {
      camera_id: String(camera.camera_id ?? id),
      camera_name: String(camera.camera_name ?? camera.name ?? id.toUpperCase()),
      timestamp: typeof camera.timestamp === 'string' ? camera.timestamp : timestamp,
      frame_number: Number(camera.frame_number ?? raw.frame_id ?? raw.frame ?? 0),
      scene_status: sceneStatus,
      tracks,
      alerts: cameraAlerts,
    }
  }
  const maxSeverity: RiskLevel = Object.values(cameras).some(c => c.scene_status === 'CRITICAL') || globalAlerts.some(a => a.risk_level === 'CRITICAL')
    ? 'CRITICAL'
    : Object.values(cameras).some(c => c.scene_status === 'WARNING') || globalAlerts.some(a => a.risk_level === 'WARNING')
      ? 'WARNING'
      : 'NORMAL'
  return { timestamp, frame_id: Number(raw.frame_id ?? raw.frame ?? 0), overall_risk: riskFrom(raw.overall_risk ?? maxSeverity), cameras, global_alerts: globalAlerts }
}

export function useTelemetry() {
  const [status, setStatus] = useState<ConnectionStatus>('CONNECTING')
  const [cameras, setCameras] = useState<Record<string, CameraTelemetry>>({})
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [latestFrame, setLatestFrame] = useState<TelemetryFrame | null>(null)
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null)
  const [criticalFlash, setCriticalFlash] = useState(0)
  const [clockTick, setClockTick] = useState(Date.now())
  const retryRef = useRef(0)
  const socketRef = useRef<WebSocket | null>(null)
  const socketAttempt = useRef(0)
  const reconnectTimer = useRef<number | undefined>(undefined)
  const closedByUser = useRef(false)
  const knownAlertIds = useRef(new Set<string>())
  const previousRisk = useRef<RiskLevel>('NORMAL')

  const connect = useCallback(() => {
    if (closedByUser.current) return
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const endpoint = `${scheme}//${window.location.host}/ws/telemetry`
    setStatus(retryRef.current === 0 ? 'CONNECTING' : 'RECONNECTING')
    let socket: WebSocket
    try { socket = new WebSocket(endpoint) } catch { setStatus('ERROR'); return }
    const attempt = ++socketAttempt.current
    socketRef.current = socket
    socket.onopen = () => { if (attempt === socketAttempt.current) { retryRef.current = 0; setStatus('CONNECTED') } }
    socket.onmessage = event => {
      if (attempt !== socketAttempt.current) return
      try {
        const frame = normalizeTelemetry(JSON.parse(event.data))
        if (!frame) return
        setLatestFrame(frame)
        setCameras(previous => ({ ...previous, ...frame.cameras }))
        setLastMessageAt(Date.now())
        const incoming = [...frame.global_alerts, ...Object.values(frame.cameras).flatMap(c => c.alerts)]
        const hasNewCritical = incoming.some(alert => alert.risk_level === 'CRITICAL' && !knownAlertIds.current.has(alert.id))
        for (const alert of incoming) knownAlertIds.current.add(alert.id)
        if (knownAlertIds.current.size > 200) knownAlertIds.current = new Set([...knownAlertIds.current].slice(-100))
        if (incoming.length) {
          setAlerts(previous => [...incoming, ...previous.filter(old => !incoming.some(item => item.id === old.id))]
            .sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)).slice(0, MAX_ALERTS))
        }
        if (hasNewCritical || (frame.overall_risk === 'CRITICAL' && previousRisk.current !== 'CRITICAL')) setCriticalFlash(Date.now())
        previousRisk.current = frame.overall_risk
      } catch { /* Ignore malformed or partial messages; retain last good frame. */ }
    }
    socket.onerror = () => { if (attempt === socketAttempt.current) setStatus('ERROR') }
    socket.onclose = () => {
      if (closedByUser.current || attempt !== socketAttempt.current) return
      setStatus('RECONNECTING')
      const delay = Math.min(1000 * (2 ** retryRef.current), 10_000)
      retryRef.current += 1
      reconnectTimer.current = window.setTimeout(connect, delay)
    }
  }, [])

  useEffect(() => {
    closedByUser.current = false
    connect()
    return () => {
      closedByUser.current = true
      socketAttempt.current += 1
      window.clearTimeout(reconnectTimer.current)
      socketRef.current?.close()
    }
  }, [connect])

  useEffect(() => {
    const timer = window.setInterval(() => setClockTick(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  const clearAlerts = useCallback(() => setAlerts([]), [])
  const stale = lastMessageAt == null || clockTick - lastMessageAt > 5000
  return { status, cameras, alerts, latestFrame, lastMessageAt, stale, criticalFlash, clearAlerts }
}
