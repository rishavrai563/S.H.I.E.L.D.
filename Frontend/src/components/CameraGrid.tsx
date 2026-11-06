import { useCallback } from 'react'
import { Video } from 'lucide-react'
import CameraCard from './CameraCard'
import type { CameraStream, CameraTelemetry } from '../types/telemetry'

interface Props { streams: CameraStream[]; telemetry: Record<string, CameraTelemetry>; stale: boolean; onConfigure: (cameraId: string) => void }

export default function CameraGrid({ streams, telemetry, stale, onConfigure }: Props) {
  const configure = useCallback((id: string) => onConfigure(id), [onConfigure])
  return <section className="min-w-0" aria-labelledby="feeds-title">
    <div className="mb-4 flex items-baseline justify-between gap-4">
      <h2 id="feeds-title" className="section-title">Camera feeds</h2>
      <span className="text-[13px] faint">Annotated live video</span>
    </div>
    {streams.length
      ? <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">{streams.map(stream => <CameraCard key={stream.camera_id} stream={stream} data={telemetry[stream.camera_id]} stale={stale} onConfigure={configure} />)}</div>
      : <div className="empty-state"><Video size={22} /><span>No cameras configured. Add them in config/cameras.yaml.</span></div>}
  </section>
}
