import { useEffect, useState } from 'react'
import { AlertCircle, Crosshair, LoaderCircle, Plus, Save, Trash2, X } from 'lucide-react'
import { useFenceConfig, type FencePoint } from '../hooks/useFenceConfig'
import type { CameraStream } from '../types/telemetry'

interface Props { open: boolean; cameraId: string; streams: CameraStream[]; onClose: () => void }
const defaults: FencePoint[] = [[50, 1030], [1870, 1030]]
export default function FenceEditorModal({ open, cameraId, streams, onClose }: Props) {
  const [selectedCamera, setSelectedCamera] = useState(cameraId)
  const [points, setPoints] = useState<FencePoint[]>(defaults)
  const { saveFence, saving, error } = useFenceConfig()
  useEffect(() => { if (open) setSelectedCamera(cameraId) }, [cameraId, open])
  if (!open) return null
  const setPoint = (index: number, axis: 0 | 1, value: number) => setPoints(previous => previous.map((point, i) => i === index ? axis === 0 ? [value, point[1]] : [point[0], value] : point))
  const save = async () => { if (points.length < 2) return; if (await saveFence(selectedCamera, points)) onClose() }
  return <div className="modal-backdrop" role="presentation" onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}><section role="dialog" aria-modal="true" aria-labelledby="fence-title" className="w-full max-w-xl rounded-2xl border border-slate-700 bg-[#111c30] shadow-2xl shadow-black/50">
    <div className="flex items-start justify-between border-b border-white/5 px-5 py-4"><div className="flex gap-3"><div className="rounded-lg bg-emerald-400/10 p-2 text-emerald-300"><Crosshair size={18} /></div><div><h2 id="fence-title" className="font-semibold text-white">Virtual fence</h2><p className="mt-1 text-xs text-slate-500">Define frame-space coordinates for a camera.</p></div></div><button className="icon-button" onClick={onClose} aria-label="Close"><X size={16} /></button></div>
    <div className="space-y-4 p-5"><label className="block text-xs font-medium text-slate-400">Camera<select value={selectedCamera} onChange={e => setSelectedCamera(e.target.value)} className="mt-1.5 w-full rounded-lg border border-shield-line bg-slate-900 px-3 py-2.5 text-sm text-slate-200 outline-none focus:border-emerald-400/50">{streams.map(s => <option key={s.camera_id} value={s.camera_id}>{s.camera_name} ({s.camera_id})</option>)}</select></label>
      <div className="flex items-center justify-between"><p className="text-xs font-medium text-slate-400">Fence points <span className="text-slate-600">(pixels, 1920 × 1080)</span></p><button onClick={() => setPoints(p => [...p, [960, 540]])} className="inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] text-emerald-300 hover:bg-emerald-400/10"><Plus size={13} /> Add point</button></div>
      <div className="max-h-60 space-y-2 overflow-auto">{points.map((point, index) => <div key={index} className="grid grid-cols-[1fr_1fr_auto] items-end gap-2 rounded-lg bg-slate-900/60 p-3"><span className="col-span-3 text-[10px] uppercase tracking-wider text-slate-600">Point {index + 1}</span><label className="text-[10px] text-slate-500">X<input type="number" value={point[0]} onChange={e => setPoint(index, 0, Number(e.target.value))} className="number-input" /></label><label className="text-[10px] text-slate-500">Y<input type="number" value={point[1]} onChange={e => setPoint(index, 1, Number(e.target.value))} className="number-input" /></label><button disabled={points.length <= 2} aria-label={`Remove point ${index + 1}`} onClick={() => setPoints(p => p.filter((_, i) => i !== index))} className="icon-button disabled:opacity-30"><Trash2 size={14} /></button></div>)}</div>
      <div className="rounded-lg border border-amber-400/10 bg-amber-400/[0.04] p-3 text-[11px] leading-relaxed text-amber-100/60"><AlertCircle size={13} className="mr-1.5 inline text-amber-300" />Fence updates are not connected to a running risk engine. This backend returns an error instead of claiming the changes were applied. The current risk engine supports a two-point tripwire, not a polygon.</div>
      {error && <p role="alert" className="text-xs text-rose-300">{error}. Check that the backend exposes <code>/api/v1/config/fence</code>.</p>}
    </div>
    <div className="flex justify-end gap-2 border-t border-white/5 px-5 py-4"><button onClick={onClose} className="rounded-lg px-4 py-2 text-xs text-slate-400 hover:bg-white/5">Cancel</button><button disabled={saving || points.length < 2} onClick={() => void save()} className="inline-flex items-center gap-2 rounded-lg bg-emerald-400 px-4 py-2 text-xs font-bold text-slate-950 hover:bg-emerald-300 disabled:opacity-50">{saving ? <LoaderCircle className="animate-spin" size={14} /> : <Save size={14} />}{saving ? 'Saving…' : 'Save fence'}</button></div>
  </section></div>
}
