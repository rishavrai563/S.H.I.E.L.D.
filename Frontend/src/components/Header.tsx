import { useEffect, useState } from 'react'
import { Activity, Bell, Clock3, Shield, Wifi, WifiOff } from 'lucide-react'
import type { ConnectionStatus, RiskLevel, ServerStatus } from '../types/telemetry'

interface Props { status: ConnectionStatus; fps?: number; latency?: number; serverStatus: ServerStatus | null; criticalCount: number; overallRisk: RiskLevel }
const statusStyles: Record<ConnectionStatus, string> = {
  CONNECTING: 'bg-amber-400/10 text-amber-300 ring-amber-400/20', CONNECTED: 'bg-emerald-400/10 text-emerald-300 ring-emerald-400/20',
  DISCONNECTED: 'bg-slate-400/10 text-slate-300 ring-slate-400/20', RECONNECTING: 'bg-amber-400/10 text-amber-300 ring-amber-400/20', ERROR: 'bg-rose-400/10 text-rose-300 ring-rose-400/20',
}

export default function Header({ status, fps, latency, serverStatus, criticalCount, overallRisk }: Props) {
  const [clock, setClock] = useState(new Date())
  const [serverOffset, setServerOffset] = useState(0)
  useEffect(() => { const timer = window.setInterval(() => setClock(new Date()), 1000); return () => window.clearInterval(timer) }, [])
  useEffect(() => {
    const serverMillis = serverStatus?.server_time ? Date.parse(serverStatus.server_time) : NaN
    if (Number.isFinite(serverMillis)) setServerOffset(serverMillis - Date.now())
  }, [serverStatus?.server_time])
  const Icon = status === 'CONNECTED' ? Wifi : WifiOff
  return <header className="sticky top-0 z-20 border-b border-shield-line bg-shield-base/90 backdrop-blur-xl">
    <div className="mx-auto flex max-w-[1800px] flex-wrap items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-emerald-400/20 bg-emerald-400/10 text-emerald-300"><Shield size={22} /></div>
        <div><div className="flex items-center gap-2"><h1 className="text-lg font-bold tracking-[0.16em] text-white">SHIELD</h1><span className="rounded bg-slate-700/70 px-1.5 py-0.5 text-[9px] font-bold tracking-widest text-slate-300">SOC</span></div><p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Security operations center</p></div>
      </div>
      <div className="flex flex-wrap items-center gap-2 sm:gap-3">
        <div className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold ring-1 ${statusStyles[status]}`}><span className={`h-1.5 w-1.5 rounded-full ${status === 'CONNECTED' ? 'animate-pulse bg-emerald-300' : status === 'ERROR' ? 'bg-rose-300' : 'bg-amber-300'}`} /><Icon size={14} />{status}</div>
        <div className="metric-chip"><Activity size={14} className="text-cyan-300" /><span>{fps ?? serverStatus?.processing_fps ?? '—'} <small>FPS</small></span></div>
        <div className="metric-chip"><span className="text-cyan-300">◷</span><span>{latency ?? serverStatus?.latency_ms ?? '—'} <small>ms</small></span></div>
        <div className="metric-chip hidden md:flex" title="Server time when the API reports it"><Clock3 size={14} className="text-slate-400" /><span className="tabular-nums">{new Date(clock.getTime() + serverOffset).toLocaleTimeString()}</span></div>
        <div className={`metric-chip ${criticalCount ? 'text-rose-300' : 'text-slate-400'}`}><Bell size={14} /><span>{criticalCount} <small>CRITICAL</small></span></div>
        <div className={`hidden rounded-md px-2 py-1 text-[10px] font-bold tracking-wider sm:block risk-text-${overallRisk.toLowerCase()}`}>{overallRisk}</div>
      </div>
    </div>
  </header>
}
