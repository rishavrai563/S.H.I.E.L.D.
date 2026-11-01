import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Bell, CheckCheck, Filter, Siren, Volume2, VolumeX } from 'lucide-react'
import type { Alert, RiskLevel } from '../types/telemetry'

interface Props { alerts: Alert[]; cameraIds: string[]; criticalFlash: number; onClear: () => void }
const severityStyle: Record<RiskLevel, string> = { NORMAL: 'text-slate-400 bg-slate-400/10', WARNING: 'text-amber-300 bg-amber-400/10', CRITICAL: 'text-rose-300 bg-rose-400/10' }
export default function AlertFeed({ alerts, cameraIds, criticalFlash, onClear }: Props) {
  const [camera, setCamera] = useState('ALL')
  const [severity, setSeverity] = useState('ALL')
  const [sound, setSound] = useState(false)
  const filtered = useMemo(() => alerts.filter(a => (camera === 'ALL' || a.camera_id === camera) && (severity === 'ALL' || a.risk_level === severity)), [alerts, camera, severity])
  const cameras = useMemo(() => [...new Set([...cameraIds, ...alerts.map(a => a.camera_id)])], [alerts, cameraIds])
  const audioContext = useRef<AudioContext | null>(null)
  const lastSound = useRef(0)
  const playAlert = useCallback(async () => {
    const ctx = audioContext.current
    if (!sound || !ctx) return
    if (ctx.state === 'suspended') await ctx.resume()
    const osc = ctx.createOscillator(), gain = ctx.createGain()
    osc.type = 'sine'; osc.frequency.value = 880; gain.gain.value = 0.06
    osc.connect(gain); gain.connect(ctx.destination); osc.start(); osc.stop(ctx.currentTime + 0.16)
  }, [sound])
  const toggleSound = () => {
    if (!sound) {
      audioContext.current ??= new AudioContext()
      lastSound.current = criticalFlash
      void audioContext.current.resume()
    }
    setSound(!sound)
  }
  useEffect(() => () => { void audioContext.current?.close() }, [])
  // Audio is user-enabled and played only in response to new critical events.
  useEffect(() => {
    if (sound && criticalFlash && criticalFlash !== lastSound.current) { lastSound.current = criticalFlash; void playAlert() }
  }, [sound, criticalFlash, playAlert])
  return <section className="rounded-xl border border-shield-line bg-shield-panel/80">
    <div className="flex items-center justify-between border-b border-white/5 px-4 py-4"><div><p className="eyebrow">LIVE MONITOR</p><h2 className="section-title">Event feed <span className="ml-2 font-mono text-sm font-normal text-slate-500">{alerts.length}</span></h2></div><button title="Clear alert history" onClick={onClear} className="icon-button"><CheckCheck size={15} /></button></div>
    <div className="flex flex-wrap items-center gap-2 border-b border-white/5 px-4 py-3"><Filter size={13} className="text-slate-500" /><select aria-label="Filter by camera" value={camera} onChange={e => setCamera(e.target.value)} className="filter-select"><option value="ALL">All cameras</option>{cameras.map(id => <option key={id}>{id}</option>)}</select><select aria-label="Filter by severity" value={severity} onChange={e => setSeverity(e.target.value)} className="filter-select"><option value="ALL">All severity</option><option>NORMAL</option><option>WARNING</option><option>CRITICAL</option></select><button onClick={toggleSound} className={`ml-auto inline-flex items-center gap-1.5 rounded-md px-2 py-1.5 text-[10px] font-medium ${sound ? 'bg-emerald-400/10 text-emerald-300' : 'bg-slate-800 text-slate-400'}`}>{sound ? <Volume2 size={13} /> : <VolumeX size={13} />}{sound ? 'Sound on' : 'Sound off'}</button></div>
    <div className="max-h-[430px] overflow-auto p-2">
      {filtered.length ? <ul className="space-y-1">{filtered.map(alert => { const Icon = alert.risk_level === 'CRITICAL' ? Siren : alert.risk_level === 'WARNING' ? AlertTriangle : Bell; return <li key={`${alert.id}-${alert.risk_level === 'CRITICAL' ? criticalFlash : 0}`} className={`rounded-lg border border-transparent p-3 hover:border-white/5 ${alert.risk_level === 'CRITICAL' && criticalFlash ? 'critical-flash' : ''}`}>
        <div className="flex items-start gap-3"><div className={`mt-0.5 rounded-md p-1.5 ${severityStyle[alert.risk_level]}`}><Icon size={14} /></div><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-x-2 gap-y-1"><span className="text-xs font-semibold text-slate-200">{alert.message}</span><span className={`rounded px-1.5 py-0.5 text-[9px] font-bold ${severityStyle[alert.risk_level]}`}>{alert.risk_level}</span></div><div className="mt-1.5 flex items-center justify-between gap-2 text-[10px] text-slate-500"><span className="truncate">{alert.camera_id} <span className="mx-1 text-slate-700">/</span> {alert.alert_type.replace(/_/g, ' ')}</span><time className="shrink-0 font-mono">{new Date(alert.timestamp).toLocaleTimeString()}</time></div></div></div>
      </li> })}</ul> : <div className="flex min-h-36 flex-col items-center justify-center gap-2 text-slate-600"><Bell size={20} /><p className="text-xs">No events match these filters</p></div>}
    </div>
  </section>
}
