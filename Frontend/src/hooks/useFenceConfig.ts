import { useState } from 'react'

export type FencePoint = [number, number]
export type FenceType = 'line' | 'zone'
export interface FenceShape { type: FenceType; points: FencePoint[]; ignore: FencePoint[][]; frame?: { width: number; height: number } }

export function useFenceConfig() {
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const saveFence = async (camera_id: string, fence_points: FencePoint[], type: FenceType = 'line') => {
    setSaving(true); setError(null)
    try {
      const response = await fetch('/api/v1/fence/config', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id, type, points: fence_points }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail ?? `Fence API returned ${response.status}`)
      }
      return true
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Could not save fence configuration'
      setError(message); return false
    } finally { setSaving(false) }
  }

  /** Replace a camera's ignore areas (posters, screens...). [] removes them all. */
  const saveIgnore = async (camera_id: string, zones: FencePoint[][]) => {
    setSaving(true); setError(null)
    try {
      const response = await fetch('/api/v1/fence/ignore', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id, zones }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail ?? `Ignore-area API returned ${response.status}`)
      }
      return true
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not save ignore areas'); return false
    } finally { setSaving(false) }
  }

  /** Current fence of a camera, or null if the backend is unreachable. */
  const loadFence = async (camera_id: string): Promise<FenceShape | null> => {
    try {
      const response = await fetch(`/api/v1/fence/${encodeURIComponent(camera_id)}`)
      if (!response.ok) return null
      const body = await response.json()
      return { type: body.type === 'zone' ? 'zone' : 'line', points: body.points ?? [], ignore: body.ignore ?? [], frame: body.frame }
    } catch { return null }
  }

  return { saveFence, saveIgnore, loadFence, saving, error }
}
