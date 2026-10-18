import { useEffect, useState } from 'react'
import type { CameraStream, ServerStatus } from '../types/telemetry'

const fallback: CameraStream[] = [
  { camera_id: 'cam1', camera_name: 'Highway Checkpost', status: 'CONNECTING' },
  { camera_id: 'cam2', camera_name: 'Border Patrol (Crowd)', status: 'CONNECTING' },
  { camera_id: 'cam3', camera_name: 'Night Vision Sector', status: 'CONNECTING' },
]
const streamStatus: Record<string, CameraStream['status']> = {
  active: 'ACTIVE', online: 'ACTIVE', offline: 'OFFLINE', connecting: 'CONNECTING', configured: 'CONFIGURED',
}

export function useStreams() {
  const [streams, setStreams] = useState<CameraStream[]>(fallback)
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null)
  useEffect(() => {
    let mounted = true
    const load = async () => {
      try {
        let response = await fetch('/api/v1/streams')
        if (response.status === 404) response = await fetch('/api/v1/streams/active')
        if (response.ok) {
          const payload = await response.json()
          const items = Array.isArray(payload) ? payload : payload.cameras ?? payload.streams ?? []
          if (mounted && items.length) setStreams(items.map((item: any) => ({
            camera_id: String(item.camera_id ?? item.id), camera_name: String(item.camera_name ?? item.name ?? item.id),
            status: streamStatus[String(item.status).toLowerCase()] ?? 'CONFIGURED',
          })))
        }
      } catch { /* Keep configured display fallbacks if the endpoint is unavailable. */ }
      try {
        const response = await fetch('/api/v1/status')
        if (response.ok && mounted) setServerStatus(await response.json())
      } catch { /* Status is optional; websocket state remains authoritative. */ }
    }
    void load()
    const timer = window.setInterval(load, 15_000)
    return () => { mounted = false; window.clearInterval(timer) }
  }, [])
  return { streams, serverStatus }
}
