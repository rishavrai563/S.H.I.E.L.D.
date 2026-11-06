import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Bell, Trash2, Volume2, VolumeX } from 'lucide-react'
import type { Alert, RiskLevel } from '../types/telemetry'

interface Props { alerts: Alert[]; cameraIds: string[]; criticalFlash: number; onClear: () => void }

const levelText: Record<RiskLevel, string> = { NORMAL: 'Info', WARNING: 'Watch', CRITICAL: 'Respond' }
const levelTag: Record<RiskLevel, string> = { NORMAL: 'tag-normal', WARNING: 'tag-warning', CRITICAL: 'tag-critical' }
const typeText = (type: string) => type.toLowerCase().replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase())

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

  return <section className="panel xl:sticky xl:top-24" aria-labelledby="events-title">
    <div className="panel-head">
      <h2 id="events-title" className="panel-title">Events <span className="mono ml-1 text-[12px] font-normal faint">{alerts.length}</span></h2>
      <div className="flex items-center gap-1">
        <button onClick={toggleSound} aria-pressed={sound} title={sound ? 'Mute critical alarm' : 'Sound an alarm on critical events'} className={`icon-button ${sound ? 'text-tide' : ''}`}>{sound ? <Volume2 size={15} /> : <VolumeX size={15} />}</button>
        <button title="Clear the event list" aria-label="Clear the event list" onClick={onClear} className="icon-button"><Trash2 size={15} /></button>
      </div>
    </div>
    <div className="flex gap-2 border-b border-line px-4 py-3">
      <select aria-label="Show events from camera" value={camera} onChange={e => setCamera(e.target.value)} className="filter-select flex-1">
        <option value="ALL">All cameras</option>{cameras.map(id => <option key={id} value={id}>{id}</option>)}
      </select>
      <select aria-label="Show events of severity" value={severity} onChange={e => setSeverity(e.target.value)} className="filter-select flex-1">
        <option value="ALL">Any severity</option><option value="CRITICAL">Respond</option><option value="WARNING">Watch</option><option value="NORMAL">Info</option>
      </select>
    </div>
    <div className="max-h-[640px] overflow-auto" data-lenis-prevent>
      {filtered.length
        ? <ul>{filtered.map(alert => <li key={`${alert.id}-${alert.risk_level === 'CRITICAL' ? criticalFlash : 0}`}
            className={`event ${alert.risk_level === 'CRITICAL' ? 'event-critical' : ''} ${alert.risk_level === 'CRITICAL' && criticalFlash ? 'critical-flash' : ''}`}>
            <div className="flex items-start justify-between gap-3">
              <p className="text-[14px] font-medium leading-snug">{alert.message}</p>
              {alert.risk_level !== 'CRITICAL' && <span className={`tag ${levelTag[alert.risk_level]} shrink-0`}>{levelText[alert.risk_level]}</span>}
            </div>
            <div className="event-meta mt-1.5 flex justify-between gap-2">
              <span className="truncate"><span className="mono">{alert.camera_id}</span>, {typeText(alert.alert_type)}</span>
              <time className="mono shrink-0">{new Date(alert.timestamp).toLocaleTimeString()}</time>
            </div>
          </li>)}</ul>
        : <div className="flex min-h-48 flex-col items-center justify-center gap-2 px-6 text-center faint">
            <Bell size={18} />
            <p className="text-[13px]">{alerts.length ? 'No events match these filters.' : 'No events yet. Intrusions, watchlist matches and night movement appear here.'}</p>
          </div>}
    </div>
  </section>
}
