import { useCallback, useState } from 'react'

export interface FaceIdentity {
  identity_id: string
  name: string
  category: string
  notes: string
  enrolled_at: number
  num_references: number
  thumbnail: string
}

export interface PlateEntry { plate: string; tag: string; notes: string; added_at: number }

export interface PlateRead {
  plate: string
  confidence: number
  camera_id: string
  vehicle_class: string
  timestamp: number
  watchlist: { tag: string; watch_plate: string; exact: boolean } | null
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...init })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail ?? `Request failed (${response.status})`)
  return body as T
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new Error('Could not read file'))
    reader.readAsDataURL(file)
  })
}

export function useWatchlist() {
  const [faces, setFaces] = useState<FaceIdentity[]>([])
  const [plates, setPlates] = useState<PlateEntry[]>([])
  const [reads, setReads] = useState<PlateRead[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = useCallback(async <T,>(fn: () => Promise<T>): Promise<T | undefined> => {
    setBusy(true); setError(null)
    try { return await fn() } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Request failed'); return undefined
    } finally { setBusy(false) }
  }, [])

  const refresh = useCallback(async () => {
    await run(async () => {
      const [g, w, r] = await Promise.allSettled([
        request<{ identities: FaceIdentity[] }>('/api/v1/faces/gallery'),
        request<{ watchlist: PlateEntry[] }>('/api/v1/anpr/watchlist'),
        request<{ plates: PlateRead[] }>('/api/v1/anpr/plates?limit=20'),
      ])
      if (g.status === 'fulfilled') setFaces(g.value.identities)
      if (w.status === 'fulfilled') setPlates(w.value.watchlist)
      if (r.status === 'fulfilled') setReads(r.value.plates)
      const failed = [g, w].find(x => x.status === 'rejected') as PromiseRejectedResult | undefined
      if (failed) throw failed.reason
    })
  }, [run])

  const enrollFace = useCallback(async (name: string, category: string, notes: string, files: File[]) => {
    const images = await Promise.all(files.map(fileToBase64))
    const ok = await run(() => request('/api/v1/faces/enroll', { method: 'POST', body: JSON.stringify({ name, category, notes, images }) }))
    if (ok) await refresh()
    return Boolean(ok)
  }, [run, refresh])

  const removeFace = useCallback(async (id: string) => {
    if (await run(() => request(`/api/v1/faces/${encodeURIComponent(id)}`, { method: 'DELETE' }))) await refresh()
  }, [run, refresh])

  const addPlate = useCallback(async (plate: string, tag: string, notes: string) => {
    const ok = await run(() => request('/api/v1/anpr/watchlist', { method: 'POST', body: JSON.stringify({ plate, tag, notes }) }))
    if (ok) await refresh()
    return Boolean(ok)
  }, [run, refresh])

  const removePlate = useCallback(async (plate: string) => {
    if (await run(() => request(`/api/v1/anpr/watchlist/${encodeURIComponent(plate)}`, { method: 'DELETE' }))) await refresh()
  }, [run, refresh])

  return { faces, plates, reads, busy, error, refresh, enrollFace, removeFace, addPlate, removePlate }
}
