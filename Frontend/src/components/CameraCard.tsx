import { memo } from 'react'
import { Car, Moon, PencilRuler, Users } from 'lucide-react'
import type { CameraStream, CameraTelemetry, RiskLevel } from '../types/telemetry'
import { useCameraStream } from '../hooks/useCameraStream'

interface Props { stream: CameraStream; data?: CameraTelemetry; stale: boolean; onConfigure: (cameraId: string) => void }

const riskTag: Record<RiskLevel, string> = { NORMAL: 'tag-normal', WARNING: 'tag-warning', CRITICAL: 'tag-critical' }
const riskText: Record<RiskLevel, string> = { NORMAL: 'Clear', WARNING: 'Watch', CRITICAL: 'Respond' }
const streamText = { ACTIVE: 'Streaming', OFFLINE: 'Offline', CONNECTING: 'Connecting', CONFIGURED: 'Configured' }
const clamp = (value: number, max: number) => Math.max(0, Math.min(max, value))

function CameraCard({ stream, data, stale, onConfigure }: Props) {
  const risk = data?.scene_status ?? 'NORMAL'
  const frameDate = data?.timestamp ? new Date(data.timestamp) : null
  const { imageRef, hasFrame } = useCameraStream(stream.camera_id)
  const frameWidth = stream.width ?? 1920
  const frameHeight = stream.height ?? 1080
  const watchlisted = data?.vehicles.filter(v => v.watchlist_tag) ?? []
  const highRisk = data?.tracks.filter(t => t.risk_score >= 80).length ?? 0

  return <article className="panel" aria-label={`${stream.camera_name} camera`}>
    <div className="panel-head">
      <div className="min-w-0">
        <h3 className="panel-title truncate">{stream.camera_name || data?.camera_name}</h3>
        <p className="mt-0.5 text-[12px] faint"><span className="mono">{stream.camera_id}</span>, {streamText[stream.status].toLowerCase()}</p>
      </div>
      <div className="flex shrink-0 items-center gap-1.5">
        {data?.night_mode && <span className="tag tag-night"><Moon size={11} />Night</span>}
        <span className={`tag ${riskTag[risk]}`}>{riskText[risk]}</span>
        <button aria-label={`Edit the tripwire for ${stream.camera_name}`} title="Edit tripwire" onClick={() => onConfigure(stream.camera_id)} className="icon-button"><PencilRuler size={15} /></button>
      </div>
    </div>

    <div className={`feed ${risk === 'CRITICAL' ? 'feed-state-critical' : risk === 'WARNING' ? 'feed-state-warning' : ''}`}>
      <img ref={imageRef} alt={`Live annotated video from ${stream.camera_name}`} className={`absolute inset-0 h-full w-full object-contain ${hasFrame ? '' : 'hidden'}`} />
      {!hasFrame && <>
        <div className="dither" />
        <div className="absolute inset-x-0 bottom-0 p-4">
          <p className="text-[15px] font-medium text-bone">{stream.status === 'ACTIVE' ? 'Waiting for the first frame' : 'No video from this camera'}</p>
          <p className="mt-1 text-[12.5px] muted">{stream.status === 'OFFLINE' ? 'Check the camera source in cameras.yaml.' : 'Video appears once the analytics server processes it.'}</p>
        </div>
      </>}
      <div className="absolute left-3 top-3 flex items-center gap-1.5">
        <span className="overlay-chip">#{data?.frame_number ?? '—'}</span>
        {data?.analysis_fps != null && <span className="overlay-chip text-tide" title="Frames of this camera analysed per second, and capture-to-result latency">{data.analysis_fps.toFixed(1)} fps{data.latency_ms != null ? `, ${data.latency_ms} ms` : ''}</span>}
        {stale && <span className="overlay-chip text-caution">Stale</span>}
      </div>
      <span className="overlay-chip absolute right-3 top-3">{frameDate ? frameDate.toLocaleTimeString() : '--:--:--'}</span>
      {!hasFrame && data?.tracks.map(track => {
        const [x1, y1, x2, y2] = track.bbox
        const left = clamp(x1 / frameWidth * 100, 100), top = clamp(y1 / frameHeight * 100, 100)
        const width = clamp((x2 - x1) / frameWidth * 100, 100 - left), height = clamp((y2 - y1) / frameHeight * 100, 100 - top)
        const color = track.risk_score >= 80 ? 'border-breach' : track.risk_score >= 50 ? 'border-caution' : 'border-tide'
        return <div key={track.track_id} className={`absolute z-[1] border ${color}`} style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}>
          <span className="overlay-chip absolute -top-5 left-[-1px] whitespace-nowrap">{track.track_id} {track.class_name} {Math.round(track.risk_score)}</span>
        </div>
      })}
    </div>

    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3 text-[13px]">
      <span className="flex items-center gap-2 muted"><Users size={15} className="text-tide" /><span className="mono text-bone">{data?.tracks.length ?? 0}</span> people</span>
      <span className="flex items-center gap-2 muted"><Car size={15} className="text-tide" /><span className="mono text-bone">{data?.vehicles.length ?? 0}</span> vehicles</span>
      {highRisk > 0 && <span className="text-breach">{highRisk} high risk</span>}
      {watchlisted.length > 0 && <span className="tag tag-critical ml-auto">Watchlist vehicle {watchlisted[0].plate}</span>}
    </div>
  </article>
}
export default memo(CameraCard)
