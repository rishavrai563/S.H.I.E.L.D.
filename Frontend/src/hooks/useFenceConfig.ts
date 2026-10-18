import { useState } from 'react'

export type FencePoint = [number, number]
export function useFenceConfig() {
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const saveFence = async (camera_id: string, fence_points: FencePoint[]) => {
    setSaving(true); setError(null)
    try {
      const response = await fetch('/api/v1/config/fence', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id, fence_points }),
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
  return { saveFence, saving, error }
}
