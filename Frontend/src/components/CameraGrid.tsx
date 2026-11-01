import { useCallback } from 'react'
import { Video } from 'lucide-react'
import CameraCard from './CameraCard'
import type { CameraStream, CameraTelemetry } from '../types/telemetry'

interface Props { streams: CameraStream[]; telemetry: Record<string, CameraTelemetry>; stale: boolean; onConfigure: (cameraId: string) => void }
export default function CameraGrid({ streams, telemetry, stale, onConfigure }: Props) {
  const configure = useCallback((id: string) => onConfigure(id), [onConfigure])
  return <section className="min-w-0">
    <div className="section-heading"><div><p className="eyebrow">MONITORING</p><h2 className="section-title">Camera streams <span className="ml-2 font-mono text-sm font-normal text-slate-500">{streams.length.toString().padStart(2, '0')}</span></h2></div><div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-slate-500"><Video size={14} /> Multi-camera view</div></div>
    {streams.length ? <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">{streams.map(stream => <CameraCard key={stream.camera_id} stream={stream} data={telemetry[stream.camera_id]} stale={stale} onConfigure={configure} />)}</div> : <div className="empty-state"><Video size={24} /><span>No camera streams configured</span></div>}
  </section>
}
