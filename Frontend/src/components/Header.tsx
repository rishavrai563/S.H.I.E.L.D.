import { useEffect, useState } from 'react'
import { ScanFace, Shield } from 'lucide-react'
import type { ConnectionStatus } from '../types/telemetry'

interface Props {
  status: ConnectionStatus
  fps?: number
  latency?: number
  serverTime?: string
  criticalCount: number
  onOpenWatchlists: () => void
  view: 'console' | 'about'
  onNavigate: (view: 'console' | 'about') => void
}

const statusText: Record<ConnectionStatus, string> = {
  CONNECTING: 'Connecting', CONNECTED: 'Live', DISCONNECTED: 'Offline', RECONNECTING: 'Reconnecting', ERROR: 'Connection error',
}
const statusColor: Record<ConnectionStatus, string> = {
  CONNECTING: 'text-caution', CONNECTED: 'text-tide', DISCONNECTED: 'text-fog', RECONNECTING: 'text-caution', ERROR: 'text-breach',
}

export default function Header({ status, fps, latency, serverTime, criticalCount, onOpenWatchlists, view, onNavigate }: Props) {
  const [clock, setClock] = useState(new Date())
  const [offset, setOffset] = useState(0)
  useEffect(() => { const t = window.setInterval(() => setClock(new Date()), 1000); return () => window.clearInterval(t) }, [])
  useEffect(() => {
    const server = serverTime ? Date.parse(serverTime) : NaN
    if (Number.isFinite(server)) setOffset(server - Date.now())
  }, [serverTime])

  return <header className="navbar">
    <div className="nav-meta">
      <span className={`flex items-center gap-2 ${statusColor[status]}`} role="status">
        <span className={`dot ${status === 'CONNECTED' ? 'motion-safe:animate-pulse' : ''}`} />
        <span className="text-[12px] font-medium">{statusText[status]}</span>
      </span>
      <span className="hidden md:inline">FPS <b>{fps != null ? fps.toFixed(1) : '—'}</b></span>
      {latency != null && <span className="hidden md:inline">Latency <b>{latency} ms</b></span>}
      <span className="hidden lg:inline" title="Server time"><b>{new Date(clock.getTime() + offset).toLocaleTimeString()}</b></span>
    </div>

    <div className="wordmark"><Shield size={18} strokeWidth={2.2} className="text-tide" />shield</div>

    <div className="flex items-center justify-end gap-2">
      <nav className="seg" aria-label="Sections">
        <button aria-current={view === 'console' ? 'page' : undefined} onClick={() => onNavigate('console')}>Console</button>
        <button aria-current={view === 'about' ? 'page' : undefined} onClick={() => onNavigate('about')}>About</button>
      </nav>
      {criticalCount > 0 && <span className="tag tag-critical hidden sm:inline-flex">{criticalCount} critical</span>}
      {view === 'console' && <button onClick={onOpenWatchlists} className="pill pill-tide"><ScanFace size={14} /><span className="hidden sm:inline">Watchlists</span></button>}
    </div>
  </header>
}
