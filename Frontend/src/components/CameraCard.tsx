import { memo } from 'react'
import { Camera, Maximize2, PencilRuler, Radio, Users } from 'lucide-react'
import type { CameraStream, CameraTelemetry, RiskLevel } from '../types/telemetry'

interface Props { stream: CameraStream; data?: CameraTelemetry; stale: boolean; onConfigure: (cameraId: string) => void }
const riskBorder: Record<RiskLevel, string> = { NORMAL: 'border-emerald-400/40', WARNING: 'border-amber-400/60', CRITICAL: 'border-rose-400/80' }
const riskPill: Record<RiskLevel, string> = { NORMAL: 'bg-emerald-400/10 text-emerald-300', WARNING: 'bg-amber-400/10 text-amber-300', CRITICAL: 'bg-rose-400/10 text-rose-300' }
const streamStyle = { ACTIVE: 'text-emerald-300', OFFLINE: 'text-rose-300', CONNECTING: 'text-amber-300', CONFIGURED: 'text-slate-400' }
const clamp = (value: number, max: number) => Math.max(0, Math.min(max, value))

function CameraCard({ stream, data, stale, onConfigure }: Props) {
  const risk = data?.scene_status ?? 'NORMAL'
  const frameDate = data?.timestamp ? new Date(data.timestamp) : null
  return <article className={`overflow-hidden rounded-xl border bg-shield-panel/90 shadow-lg shadow-black/10 transition-colors ${riskBorder[risk]}`}>
    <div className="flex items-center justify-between gap-3 border-b border-white/5 px-4 py-3">
      <div className="flex min-w-0 items-center gap-3"><div className="rounded-lg bg-slate-800 p-2 text-slate-300"><Camera size={16} /></div><div className="min-w-0"><div className="flex items-center gap-2"><h3 className="truncate text-sm font-semibold text-slate-100">{stream.camera_name || data?.camera_name}</h3><span className="font-mono text-[10px] text-slate-500">{stream.camera_id}</span></div><div className={`mt-1 flex items-center gap-1.5 text-[10px] uppercase tracking-wider ${streamStyle[stream.status]}`}><Radio size={11} />{stream.status}</div></div></div>
      <div className="flex items-center gap-2"><span className={`rounded px-2 py-1 text-[10px] font-bold tracking-wider ${riskPill[risk]}`}>{risk}</span><button aria-label={`Configure ${stream.camera_id} virtual fence`} title="Virtual fence" onClick={() => onConfigure(stream.camera_id)} className="icon-button"><PencilRuler size={14} /></button></div>
    </div>
    <div className="camera-view relative aspect-video overflow-hidden bg-[#080f1b]">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_115%,rgba(16,185,129,0.11),transparent_50%),linear-gradient(180deg,rgba(148,163,184,0.025)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.025)_1px,transparent_1px)] bg-[length:auto,24px_24px,24px_24px]" />
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-600"><Camera size={28} strokeWidth={1.25} /><span className="text-[10px] uppercase tracking-[0.2em]">Live video unavailable</span><span className="text-[9px] text-slate-700">Telemetry overlay preview</span></div>
      <div className="absolute left-3 top-3 flex items-center gap-2"><span className="rounded bg-black/60 px-2 py-1 font-mono text-[10px] text-slate-300">FRAME {data?.frame_number ?? '—'}</span>{stale && <span className="rounded bg-amber-500/15 px-2 py-1 text-[9px] font-bold text-amber-300">STALE</span>}</div>
      <div className="absolute right-3 top-3 rounded bg-black/60 px-2 py-1 font-mono text-[10px] text-slate-400">{frameDate ? frameDate.toLocaleTimeString() : 'NO SIGNAL'}</div>
      {data?.tracks.map(track => {
        // The pipeline currently processes frames at a nominal 1920x1080; clamp to avoid invalid CSS on outliers.
        const [x1, y1, x2, y2] = track.bbox
        const left = clamp(x1 / 1920 * 100, 100), top = clamp(y1 / 1080 * 100, 100)
        const width = clamp((x2 - x1) / 1920 * 100, 100 - left), height = clamp((y2 - y1) / 1080 * 100, 100 - top)
        const color = track.risk_score >= 80 ? 'border-rose-400 text-rose-200' : track.risk_score >= 50 ? 'border-amber-300 text-amber-100' : 'border-emerald-400 text-emerald-200'
        return <div key={track.track_id} className={`absolute z-[1] border ${color}`} style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}>
          <span className="absolute -top-4 left-[-1px] whitespace-nowrap bg-black/75 px-1 py-0.5 font-mono text-[9px]">ID {track.track_id} · {track.class_name} · {Math.round(track.risk_score)}</span>
        </div>
      })}
      <button aria-label="Camera details" className="absolute bottom-3 right-3 rounded bg-black/50 p-1.5 text-slate-400"><Maximize2 size={13} /></button>
    </div>
    <div className="flex items-center justify-between border-t border-white/5 px-4 py-3 text-xs"><div className="flex items-center gap-2 text-slate-400"><Users size={14} className="text-cyan-300" /><span>{data?.tracks.length ?? 0} active tracks</span></div><span className="font-mono text-[10px] text-slate-500">{data?.tracks.filter(t => t.risk_score >= 80).length ?? 0} high risk</span></div>
  </article>
}
export default memo(CameraCard)
