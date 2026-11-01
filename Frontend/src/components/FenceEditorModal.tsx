import { useEffect, useState } from 'react'
import { AlertCircle, Crosshair, LoaderCircle, Save, X } from 'lucide-react'
import { useFenceConfig, type FencePoint } from '../hooks/useFenceConfig'
import type { CameraStream } from '../types/telemetry'

interface Props { open: boolean; cameraId: string; streams: CameraStream[]; onClose: () => void }

const defaultTripwire = (width: number, height: number): FencePoint[] => [
  [Math.round(width * 50 / 1920), Math.round(height * 1030 / 1080)],
  [Math.round(width * 1870 / 1920), Math.round(height * 1030 / 1080)],
]

export default function FenceEditorModal({ open, cameraId, streams, onClose }: Props) {
  const [selectedCamera, setSelectedCamera] = useState(cameraId)
  const [points, setPoints] = useState<FencePoint[]>(defaultTripwire(1920, 1080))
  const { saveFence, saving, error } = useFenceConfig()
  const selectedStream = streams.find(stream => stream.camera_id === selectedCamera)
  const frameWidth = selectedStream?.width ?? 1920
  const frameHeight = selectedStream?.height ?? 1080

  useEffect(() => {
    if (!open) return
    setSelectedCamera(cameraId)
    const stream = streams.find(item => item.camera_id === cameraId)
    setPoints(defaultTripwire(stream?.width ?? 1920, stream?.height ?? 1080))
  }, [cameraId, open])

  useEffect(() => {
    if (open) setPoints(defaultTripwire(frameWidth, frameHeight))
  }, [selectedCamera, open, frameWidth, frameHeight])

  if (!open) return null

  const setPoint = (index: number, axis: 0 | 1, value: number) => {
    setPoints(previous => previous.map((point, i) => i === index
      ? axis === 0 ? [value, point[1]] : [point[0], value]
      : point))
  }

  const save = async () => {
    if (points.length === 2 && await saveFence(selectedCamera, points)) onClose()
  }

  return <div className="modal-backdrop" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <section role="dialog" aria-modal="true" aria-labelledby="fence-title" className="w-full max-w-xl rounded-2xl border border-slate-700 bg-[#111c30] shadow-2xl shadow-black/50">
      <div className="flex items-start justify-between border-b border-white/5 px-5 py-4">
        <div className="flex gap-3"><div className="rounded-lg bg-emerald-400/10 p-2 text-emerald-300"><Crosshair size={18} /></div><div><h2 id="fence-title" className="font-semibold text-white">Virtual tripwire</h2><p className="mt-1 text-xs text-slate-500">Set two frame-space endpoints for the selected camera.</p></div></div>
        <button className="icon-button" onClick={onClose} aria-label="Close"><X size={16} /></button>
      </div>

      <div className="space-y-4 p-5">
        <label className="block text-xs font-medium text-slate-400">Camera
          <select value={selectedCamera} onChange={event => setSelectedCamera(event.target.value)} className="mt-1.5 w-full rounded-lg border border-shield-line bg-slate-900 px-3 py-2.5 text-sm text-slate-200 outline-none focus:border-emerald-400/50">
            {streams.map(stream => <option key={stream.camera_id} value={stream.camera_id}>{stream.camera_name} ({stream.camera_id})</option>)}
          </select>
        </label>

        <div className="flex items-center justify-between"><p className="text-xs font-medium text-slate-400">Tripwire endpoints <span className="text-slate-600">(pixels, {frameWidth} x {frameHeight})</span></p><span className="text-[10px] text-slate-500">Exactly two points</span></div>
        <div className="space-y-2">
          {points.map((point, index) => <div key={index} className="grid grid-cols-2 items-end gap-2 rounded-lg bg-slate-900/60 p-3">
            <span className="col-span-2 text-[10px] uppercase tracking-wider text-slate-600">Endpoint {index + 1}</span>
            <label className="text-[10px] text-slate-500">X<input type="number" min={0} max={frameWidth} value={point[0]} onChange={event => setPoint(index, 0, Number(event.target.value))} className="number-input" /></label>
            <label className="text-[10px] text-slate-500">Y<input type="number" min={0} max={frameHeight} value={point[1]} onChange={event => setPoint(index, 1, Number(event.target.value))} className="number-input" /></label>
          </div>)}
        </div>

        <div className="rounded-lg border border-cyan-400/10 bg-cyan-400/[0.04] p-3 text-[11px] leading-relaxed text-cyan-100/60"><AlertCircle size={13} className="mr-1.5 inline text-cyan-300" />The backend applies this two-point tripwire to the selected camera's live risk analyzer.</div>
        {error && <p role="alert" className="text-xs text-rose-300">{error}. Check that the backend exposes <code>/api/v1/fence/config</code>.</p>}
      </div>

      <div className="flex justify-end gap-2 border-t border-white/5 px-5 py-4">
        <button onClick={onClose} className="rounded-lg px-4 py-2 text-xs text-slate-400 hover:bg-white/5">Cancel</button>
        <button disabled={saving || points.length !== 2} onClick={() => void save()} className="inline-flex items-center gap-2 rounded-lg bg-emerald-400 px-4 py-2 text-xs font-bold text-slate-950 hover:bg-emerald-300 disabled:opacity-50">{saving ? <LoaderCircle className="animate-spin" size={14} /> : <Save size={14} />}{saving ? 'Saving...' : 'Save tripwire'}</button>
      </div>
    </section>
  </div>
}
