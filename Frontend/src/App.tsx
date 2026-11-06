import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react'
import Header from './components/Header'
import CameraGrid from './components/CameraGrid'
import AlertFeed from './components/AlertFeed'
import FenceEditorModal from './components/FenceEditorModal'
import WatchlistModal from './components/WatchlistModal'
import ErrorBoundary from './components/ErrorBoundary'
import PixelArrow from './components/PixelArrow'
import Landing from './components/Landing'
import type { SectorCamera } from './components/SectorMap'
import { startSmoothScroll } from './lib/smoothScroll'
import { useStreams } from './hooks/useStreams'
import { useTelemetry } from './hooks/useTelemetry'
import type { CameraStream, CameraTelemetry, ConnectionStatus, RiskLevel } from './types/telemetry'

// three.js is only downloaded once the console renders the sector view.
const SectorMap = lazy(() => import('./components/SectorMap'))

interface Situation { first: string; second: string; tone: '' | 'tone-caution' | 'tone-breach' }

/** The headline states the situation in two words an operator can read from across the room. */
function situationFor(status: ConnectionStatus, stale: boolean, risk: RiskLevel, cameras: Record<string, CameraTelemetry>): Situation {
  if (status !== 'CONNECTED' || stale) return { first: 'Awaiting', second: 'Telemetry', tone: '' }
  const worst = (level: RiskLevel) => Object.values(cameras).find(c => c.scene_status === level)?.camera_id
  if (risk === 'CRITICAL') return { first: 'Respond', second: (worst('CRITICAL') ?? 'Alert').toUpperCase(), tone: 'tone-breach' }
  if (risk === 'WARNING') return { first: 'Watch', second: (worst('WARNING') ?? 'Activity').toUpperCase(), tone: 'tone-caution' }
  return { first: 'All sectors', second: 'Clear', tone: '' }
}

/** Survey-grid coordinates: real camera IDs and positions pinned to grid lines. */
function GridTags({ streams }: { streams: CameraStream[] }) {
  const tags = streams.map((s, i) => ({
    text: `${s.camera_id.toUpperCase()}${s.gps ? `  ${s.gps.lat.toFixed(4)}N ${s.gps.lng.toFixed(4)}E` : ''}`,
    col: 1 + i * 4, row: 1,
  }))
  return <div aria-hidden className="grid-layer pointer-events-none absolute inset-x-0 top-0 hidden h-[600px] overflow-hidden lg:block">
    {tags.map(t => <span key={`${t.text}-${t.col}`} className="grid-tag" style={{ left: `calc(var(--grid) * ${t.col} + 6px)`, top: `calc(var(--grid) * ${t.row} - 7px)` }}>{t.text}</span>)}
  </div>
}

export default function App() {
  const { status, cameras, alerts, latestFrame, lastMessageAt, stale, criticalFlash, clearAlerts } = useTelemetry()
  const { streams, serverStatus } = useStreams()
  const [fenceCamera, setFenceCamera] = useState('cam1')
  const [fenceOpen, setFenceOpen] = useState(false)
  const [watchlistOpen, setWatchlistOpen] = useState(false)
  const [clockTick, setClockTick] = useState(Date.now())
  // Measured by the backend over a rolling window (not inferred from message timing).
  // Shown as 0 when telemetry goes stale so a stalled pipeline is visible.
  const fps = stale ? 0 : (latestFrame?.processing_fps ?? serverStatus?.processing_fps)
  const latency = stale ? undefined : (latestFrame?.latency_ms ?? serverStatus?.latency_ms)
  useEffect(() => { const timer = window.setInterval(() => setClockTick(Date.now()), 1000); return () => window.clearInterval(timer) }, [])
  useEffect(() => startSmoothScroll(), [])

  const overallRisk = useMemo<RiskLevel>(() => latestFrame?.overall_risk ?? 'NORMAL', [latestFrame])
  const people = useMemo(() => Object.values(cameras).reduce((sum, c) => sum + c.tracks.length, 0), [cameras])
  const vehicles = useMemo(() => Object.values(cameras).reduce((sum, c) => sum + c.vehicles.length, 0), [cameras])
  const criticalCount = useMemo(() => alerts.filter(a => a.risk_level === 'CRITICAL').length, [alerts])
  const warningCount = useMemo(() => alerts.filter(a => a.risk_level === 'WARNING').length, [alerts])
  const nightCount = useMemo(() => Object.values(cameras).filter(c => c.night_mode).length, [cameras])
  const openFence = useCallback((id: string) => { setFenceCamera(id); setFenceOpen(true) }, [])
  const situation = situationFor(status, stale, overallRisk, cameras)
  const liveCameras = Object.keys(cameras).length
  const sectorCameras = useMemo<SectorCamera[]>(() => streams.map(s => {
    const c = cameras[s.camera_id]
    return { id: s.camera_id, name: s.camera_name, online: Boolean(c) || s.status === 'ACTIVE', risk: c?.scene_status ?? 'NORMAL',
      people: c?.tracks.length ?? 0, vehicles: c?.vehicles.length ?? 0, night: Boolean(c?.night_mode), lng: s.gps?.lng }
  }), [streams, cameras])
  const lastSeen = lastMessageAt ? Math.max(0, Math.floor((clockTick - lastMessageAt) / 1000)) : null

  const note = status !== 'CONNECTED'
    ? 'Connecting to the analytics server. Live video and alerts appear here as soon as the first frame is processed.'
    : `${liveCameras} of ${streams.length} cameras reporting. ${people} ${people === 1 ? 'person' : 'people'} and ${vehicles} ${vehicles === 1 ? 'vehicle' : 'vehicles'} in view.`

  return <div className="canvas text-bone">
    <GridTags streams={streams} />
    <Header status={status} fps={fps} latency={latency} serverTime={serverStatus?.server_time}
      criticalCount={criticalCount} onOpenWatchlists={() => setWatchlistOpen(true)} />

    <Landing cameraCount={streams.length} liveCount={liveCameras} />

    <main id="console" className="mx-auto max-w-[1440px] px-4 pb-16 sm:px-8">
      <section className="situation" aria-live="polite" aria-label="Current situation">
        <h1 className="sr-only">SHIELD border surveillance console</h1>
        <p className={`situation-line ${situation.tone}`}>{situation.first}</p>
        <p className={`situation-line offset ${situation.tone}`}><PixelArrow className="arrow" />{situation.second}</p>
        <p className="situation-note">{note}</p>
      </section>

      <div className="readouts mb-10">
        <Readout label="Cameras reporting" value={`${liveCameras}/${streams.length}`} foot={nightCount ? `${nightCount} in night mode` : 'All in day mode'} />
        <Readout label="People and vehicles" value={String(people + vehicles)} foot={`${people} people, ${vehicles} vehicles`} />
        <Readout label="Open alerts" value={String(criticalCount + warningCount)} foot={criticalCount ? `${criticalCount} need a response now` : 'None need a response'} tone={criticalCount ? 'text-breach' : warningCount ? 'text-caution' : ''} />
        <Readout label="Last frame" value={lastSeen == null ? '—' : `${lastSeen}s`} foot={latestFrame ? `Frame ${latestFrame.frame_id}` : 'Nothing received yet'} />
      </div>

      <section className="panel mb-14" aria-labelledby="sector-title">
        <div className="panel-head flex-wrap">
          <h2 id="sector-title" className="panel-title">Sector view</h2>
          <span className="text-[12.5px] faint">Drag to look around. Rings pulse faster as a camera's state escalates.</span>
        </div>
        <div className="h-[380px] sm:h-[460px]">
          <Suspense fallback={<div className="grid h-full place-items-center text-[13px] faint">Loading the 3D sector view</div>}>
            <SectorMap cameras={sectorCameras} />
          </Suspense>
        </div>
        <ul className="sector-legend">
          <li><i style={{ background: 'var(--color-tide)' }} />Clear</li>
          <li><i style={{ background: 'var(--color-caution)' }} />Watch</li>
          <li><i style={{ background: 'var(--color-breach)' }} />Respond</li>
          <li><i style={{ background: '#a59bff' }} />Night view</li>
          <li><i className="line" style={{ borderColor: 'var(--color-bone)' }} />Border</li>
          <li><i style={{ background: '#e879f9' }} />Fence</li>
          <li><i className="dot-sm" style={{ background: 'var(--color-bone)' }} />Person</li>
          <li><i className="dot-sm" style={{ background: 'var(--color-tide)' }} />Vehicle</li>
        </ul>
      </section>

      <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <ErrorBoundary><CameraGrid streams={streams} telemetry={cameras} stale={stale} onConfigure={openFence} /></ErrorBoundary>
        <ErrorBoundary><AlertFeed alerts={alerts} cameraIds={streams.map(stream => stream.camera_id)} criticalFlash={criticalFlash} onClear={clearAlerts} /></ErrorBoundary>
      </div>
    </main>

    <WatchlistModal open={watchlistOpen} onClose={() => setWatchlistOpen(false)} />
    <FenceEditorModal open={fenceOpen} cameraId={fenceCamera} streams={streams} onClose={() => setFenceOpen(false)} />
  </div>
}

function Readout({ label, value, foot, tone = '' }: { label: string; value: string; foot: string; tone?: string }) {
  return <div className="readout"><span className="readout-label">{label}</span><div><div className={`readout-value ${tone}`}>{value}</div><p className="readout-foot">{foot}</p></div></div>
}
