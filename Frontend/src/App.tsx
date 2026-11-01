import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, ArrowDownRight, ArrowUpRight, Clock3, ShieldCheck, Users } from 'lucide-react'
import Header from './components/Header'
import CameraGrid from './components/CameraGrid'
import AlertFeed from './components/AlertFeed'
import FenceEditorModal from './components/FenceEditorModal'
import ErrorBoundary from './components/ErrorBoundary'
import { useStreams } from './hooks/useStreams'
import { useTelemetry } from './hooks/useTelemetry'
import type { RiskLevel } from './types/telemetry'

export default function App() {
  const { status, cameras, alerts, latestFrame, lastMessageAt, stale, criticalFlash, clearAlerts } = useTelemetry()
  const { streams, serverStatus } = useStreams()
  const [fenceCamera, setFenceCamera] = useState('cam1')
  const [fenceOpen, setFenceOpen] = useState(false)
  const [fps, setFps] = useState<number | undefined>()
  const [clockTick, setClockTick] = useState(Date.now())
  const frameSample = useRef({ id: 0, at: Date.now() })
  const latestFrameRef = useRef(latestFrame)
  latestFrameRef.current = latestFrame
  useEffect(() => {
    const timer = window.setInterval(() => {
      const frame = latestFrameRef.current
      if (!frame) return
      const current = { id: frame.frame_id, at: Date.now() }
      const elapsed = (current.at - frameSample.current.at) / 1000
      if (elapsed > 0 && current.id >= frameSample.current.id) setFps(Number(((current.id - frameSample.current.id) / elapsed).toFixed(1)))
      frameSample.current = current
    }, 1000)
    return () => window.clearInterval(timer)
  }, [])
  useEffect(() => { const timer = window.setInterval(() => setClockTick(Date.now()), 1000); return () => window.clearInterval(timer) }, [])
  const overallRisk = useMemo<RiskLevel>(() => latestFrame?.overall_risk ?? 'NORMAL', [latestFrame])
  const trackCount = useMemo(() => Object.values(cameras).reduce((sum, camera) => sum + camera.tracks.length, 0), [cameras])
  const criticalCount = useMemo(() => alerts.filter(alert => alert.risk_level === 'CRITICAL').length, [alerts])
  const latency = serverStatus?.latency_ms
  const openFence = useCallback((id: string) => { setFenceCamera(id); setFenceOpen(true) }, [])
  const incidents = alerts.filter(a => a.risk_level !== 'NORMAL').length

  return <div className="min-h-screen bg-shield-base text-slate-200">
    <Header status={status} fps={fps} latency={latency} serverStatus={serverStatus} criticalCount={criticalCount} overallRisk={overallRisk} />
    <main className="mx-auto max-w-[1800px] px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-3"><div><p className="eyebrow">COMMAND OVERVIEW / {new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric', year: 'numeric' })}</p><h2 className="mt-1 text-2xl font-semibold tracking-tight text-white">Operations overview</h2><p className="mt-1 text-xs text-slate-500">Live video analytics · multi-camera situational awareness</p></div><div className="flex flex-wrap items-center gap-2">{serverStatus?.telemetry_mode === 'simulation' && <span className="rounded-lg border border-violet-400/20 bg-violet-400/5 px-3 py-2 text-[10px] font-bold tracking-wider text-violet-200">SIMULATED SOURCE</span>}<div className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-[11px] ${stale ? 'border-amber-400/20 bg-amber-400/5 text-amber-200' : 'border-emerald-400/20 bg-emerald-400/5 text-emerald-200'}`}><span className={`h-1.5 w-1.5 rounded-full ${stale ? 'bg-amber-300' : 'animate-pulse bg-emerald-300'}`} />{stale ? 'TELEMETRY PAUSED · LAST DATA MAY BE STALE' : 'LIVE TELEMETRY'}</div></div></div>
      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <SummaryCard title="Telemetry cameras" value={`${Object.keys(cameras).length}/${streams.length}`} icon={<ShieldCheck size={17} />} tone="emerald" foot="Seen in the current session" />
        <SummaryCard title="Active tracks" value={String(trackCount)} icon={<Users size={17} />} tone="cyan" foot="Across all camera feeds" />
        <SummaryCard title="Priority events" value={String(incidents)} icon={<AlertTriangle size={17} />} tone="amber" foot="Warnings + critical alerts" />
        <SummaryCard title="Last telemetry" value={lastMessageAt ? `${Math.max(0, Math.floor((clockTick - lastMessageAt) / 1000))}s` : '—'} icon={<Clock3 size={17} />} tone="slate" foot={latestFrame ? `Frame ${latestFrame.frame_id}` : 'Awaiting first frame'} />
      </div>
      <div className="grid grid-cols-1 items-start gap-5 2xl:grid-cols-[minmax(0,1fr)_370px]">
        <ErrorBoundary><CameraGrid streams={streams} telemetry={cameras} stale={stale} onConfigure={openFence} /></ErrorBoundary>
        <ErrorBoundary><AlertFeed alerts={alerts} cameraIds={streams.map(stream => stream.camera_id)} criticalFlash={criticalFlash} onClear={clearAlerts} /></ErrorBoundary>
      </div>
      <footer className="mt-8 flex flex-wrap items-center justify-between gap-2 border-t border-shield-line/70 py-4 text-[10px] uppercase tracking-wider text-slate-600"><span>SHIELD analytics console <span className="mx-1 text-slate-700">·</span> telemetry schema v1</span><span>{latestFrame ? <><ArrowUpRight className="mr-1 inline text-emerald-500" size={12} />Frame {latestFrame.frame_id} received</> : <><ArrowDownRight className="mr-1 inline text-slate-600" size={12} />Waiting for telemetry</>}</span></footer>
    </main>
    <FenceEditorModal open={fenceOpen} cameraId={fenceCamera} streams={streams} onClose={() => setFenceOpen(false)} />
  </div>
}

function SummaryCard({ title, value, icon, tone, foot }: { title: string; value: string; icon: ReactNode; tone: string; foot: string }) {
  return <div className="rounded-xl border border-shield-line bg-shield-panel/70 p-4"><div className="flex items-center justify-between"><span className="text-[11px] text-slate-500">{title}</span><span className={`summary-icon tone-${tone}`}>{icon}</span></div><div className="mt-3 text-2xl font-semibold tabular-nums text-white">{value}</div><p className="mt-1 text-[10px] text-slate-600">{foot}</p></div>
}
