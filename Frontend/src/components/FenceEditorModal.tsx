import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { EyeOff, Eraser, LoaderCircle, MousePointerClick, Plus, RotateCcw, Save, Spline, Square, Undo2, X } from 'lucide-react'
import { useFenceConfig, type FencePoint, type FenceType } from '../hooks/useFenceConfig'
import { useCameraStream } from '../hooks/useCameraStream'
import type { CameraStream } from '../types/telemetry'

interface Props { open: boolean; cameraId: string; streams: CameraStream[]; onClose: () => void }
type Mode = FenceType | 'ignore'

const MIN_POINTS: Record<Mode, number> = { line: 2, zone: 3, ignore: 3 }
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)
const centroid = (pts: FencePoint[]): FencePoint =>
  [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length]

export default function FenceEditorModal({ open, cameraId, streams, onClose }: Props) {
  const [selectedCamera, setSelectedCamera] = useState(cameraId)
  const [mode, setMode] = useState<Mode>('line')
  const [fenceType, setFenceType] = useState<FenceType>('line')
  const [fencePts, setFencePts] = useState<FencePoint[]>([])
  const [ignore, setIgnore] = useState<FencePoint[][]>([])        // finished ignore areas
  const [draft, setDraft] = useState<FencePoint[]>([])             // ignore area being drawn
  const [saved, setSaved] = useState({ type: 'line' as FenceType, points: [] as FencePoint[], ignore: [] as FencePoint[][] })
  const [history, setHistory] = useState<FencePoint[][]>([])
  const [frame, setFrame] = useState({ width: 1920, height: 1080 })
  const [loading, setLoading] = useState(false)
  const { saveFence, saveIgnore, loadFence, saving, error } = useFenceConfig()
  const svgRef = useRef<SVGSVGElement | null>(null)
  const dragIndex = useRef<number | null>(null)
  const { imageRef, hasFrame } = useCameraStream(selectedCamera)

  const editingIgnore = mode === 'ignore'
  const points = editingIgnore ? draft : fencePts
  const setPoints = editingIgnore ? setDraft : setFencePts

  useEffect(() => { if (open) setSelectedCamera(cameraId) }, [cameraId, open])

  // Load the camera's current fence + ignore areas (in the backend's frame coordinates)
  useEffect(() => {
    if (!open) return
    let cancelled = false
    const stream = streams.find(s => s.camera_id === selectedCamera)
    setFrame({ width: stream?.width ?? 1920, height: stream?.height ?? 1080 })
    setLoading(true)
    void loadFence(selectedCamera).then(fence => {
      if (cancelled) return
      setLoading(false)
      if (!fence) return
      if (fence.frame?.width && fence.frame?.height) setFrame(fence.frame)
      setFenceType(fence.type); setMode(fence.type); setFencePts(fence.points); setIgnore(fence.ignore); setDraft([])
      setSaved({ type: fence.type, points: fence.points, ignore: fence.ignore }); setHistory([])
    })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCamera, open])

  const remember = useCallback(() => setHistory(h => [...h.slice(-49), points]), [points])
  const undo = useCallback(() => {
    if (!history.length) return
    setPoints(history[history.length - 1])
    setHistory(history.slice(0, -1))
  }, [history, setPoints])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo() }
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, undo, onClose])

  if (!open) return null

  const switchMode = (m: Mode) => {
    setHistory([])
    if (m !== 'ignore') setFenceType(m)
    setMode(m)
  }

  const { width: W, height: H } = frame
  const handle = Math.max(W, H) / 95
  const toFrame = (e: { clientX: number; clientY: number }): FencePoint => {
    const rect = svgRef.current!.getBoundingClientRect()
    const x = Math.min(W, Math.max(0, ((e.clientX - rect.left) / rect.width) * W))
    const y = Math.min(H, Math.max(0, ((e.clientY - rect.top) / rect.height) * H))
    return [Math.round(x), Math.round(y)]
  }
  const startDrag = (index: number, e: ReactPointerEvent) => {
    dragIndex.current = index
    svgRef.current?.setPointerCapture(e.pointerId)
  }
  const onBackgroundDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return
    const t = e.target as Element
    if (t !== e.currentTarget && t.getAttribute('data-bg') !== '1') return
    remember()
    setPoints(p => [...p, toFrame(e)])
  }
  const onVertexDown = (i: number) => (e: ReactPointerEvent) => {
    e.stopPropagation()
    if (e.button !== 0) return
    remember(); startDrag(i, e)
  }
  const onMidpointDown = (i: number) => (e: ReactPointerEvent) => {
    e.stopPropagation()
    if (e.button !== 0) return
    remember()
    const a = points[i], b = points[(i + 1) % points.length]
    setPoints(p => [...p.slice(0, i + 1), [Math.round((a[0] + b[0]) / 2), Math.round((a[1] + b[1]) / 2)], ...p.slice(i + 1)])
    startDrag(i + 1, e)
  }
  const onMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const i = dragIndex.current
    if (i === null) return
    const pt = toFrame(e)
    setPoints(p => p.map((q, k) => (k === i ? pt : q)))
  }
  const endDrag = () => { dragIndex.current = null }
  const removeVertex = (i: number) => (e: React.MouseEvent) => {
    e.preventDefault(); e.stopPropagation()
    remember(); setPoints(p => p.filter((_, k) => k !== i))
  }
  const finishIgnoreArea = () => {
    if (draft.length < 3) return
    setIgnore(z => [...z, draft]); setDraft([]); setHistory([])
  }

  // Ignore areas to save = finished ones + the draft if it is already a valid shape
  const ignoreToSave = draft.length >= 3 ? [...ignore, draft] : ignore
  const closed = (mode !== 'line') && points.length >= 3
  const segments = points.length < 2 ? [] : [...points.keys()].slice(0, closed ? points.length : points.length - 1)
  const fenceValid = fencePts.length === 0 || fencePts.length >= MIN_POINTS[fenceType]
  const draftValid = draft.length === 0 || draft.length >= 3
  const fenceChanged = fenceType !== saved.type || !same(fencePts, saved.points)
  const ignoreChanged = !same(ignoreToSave, saved.ignore)
  const canSave = fenceValid && draftValid && (fenceChanged || ignoreChanged)

  const save = async () => {
    if (!canSave) return
    if (fenceChanged && !(await saveFence(selectedCamera, fencePts, fenceType))) return
    if (ignoreChanged && !(await saveIgnore(selectedCamera, ignoreToSave))) return
    setSaved({ type: fenceType, points: fencePts, ignore: ignoreToSave })
    onClose()
  }

  const pointsAttr = points.map(p => p.join(',')).join(' ')
  const stroke = editingIgnore ? '#ece6cc' : '#e879f9'
  const status = editingIgnore
    ? (draft.length === 0
        ? `${ignore.length} ignore area${ignore.length === 1 ? '' : 's'} · click to outline a poster, screen or mirror`
        : draft.length < 3 ? `Add ${3 - draft.length} more corner${draft.length === 2 ? '' : 's'}` : `Area with ${draft.length} corners · press "Add area" or Save`)
    : points.length === 0
      ? 'No fence: click on the video to start drawing'
      : points.length < MIN_POINTS[mode]
        ? `Add ${MIN_POINTS[mode] - points.length} more point${MIN_POINTS[mode] - points.length > 1 ? 's' : ''}`
        : mode === 'zone' ? `Restricted zone · ${points.length} corners` : `Tripwire · ${points.length - 1} segment${points.length > 2 ? 's' : ''}`
  const modeBtn = (m: Mode, Icon: typeof Spline, label: string, title: string) =>
    <button onClick={() => switchMode(m)} title={title}
            aria-pressed={mode === m} className={`inline-flex items-center gap-1.5 px-3 py-2 text-[12.5px] font-medium ${mode === m ? (m === 'ignore' ? 'bg-bone text-abyss' : 'bg-tide text-abyss') : 'text-mist hover:text-bone'}`}>
      <Icon size={13} />{label}</button>

  return <div className="modal-backdrop" data-lenis-prevent role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <section role="dialog" aria-modal="true" aria-labelledby="fence-title" className="modal flex max-h-[95vh] max-w-5xl flex-col">
      <div className="flex items-start justify-between gap-4 px-6 pb-4 pt-6">
        <div><h2 id="fence-title" className="modal-title">Virtual fence</h2><p className="mt-1.5 text-[13px] muted">Draw directly on the live camera view. Any shape, any number of points.</p></div>
        <button className="icon-button" onClick={onClose} aria-label="Close"><X size={17} /></button>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-y border-line px-6 py-3">
        <select aria-label="Camera" value={selectedCamera} onChange={e => setSelectedCamera(e.target.value)} className="filter-select">
          {streams.map(s => <option key={s.camera_id} value={s.camera_id}>{s.camera_name} ({s.camera_id})</option>)}
        </select>
        <div className="ml-1 inline-flex border border-line-strong">
          {modeBtn('line', Spline, 'Tripwire', 'Alert when someone crosses the line')}
          {modeBtn('zone', Square, 'Restricted zone', 'Alert while someone is inside the area')}
          {modeBtn('ignore', EyeOff, 'Ignore areas', 'Ignore people/vehicles inside: posters, TV screens, mirrors')}
        </div>
        {editingIgnore && <button onClick={finishIgnoreArea} disabled={draft.length < 3} className="inline-flex items-center gap-1 border border-line-strong px-3 py-2 text-[12.5px] text-bone hover:border-bone disabled:opacity-40"><Plus size={13} />Add area</button>}
        <div className="ml-auto flex items-center gap-1">
          <button onClick={undo} disabled={!history.length} className="icon-button disabled:opacity-40" title="Undo (Ctrl+Z)"><Undo2 size={15} /></button>
          <button onClick={() => { remember(); setPoints([]) }} disabled={!points.length} className="icon-button disabled:opacity-40" title={editingIgnore ? 'Clear the area being drawn' : 'Clear all points'}><Eraser size={15} /></button>
          <button onClick={() => { setHistory([]); setFenceType(saved.type); setMode(m => (m === 'ignore' ? m : saved.type)); setFencePts(saved.points); setIgnore(saved.ignore); setDraft([]) }}
                  disabled={!fenceChanged && !ignoreChanged} className="icon-button disabled:opacity-40" title="Revert to what is saved"><RotateCcw size={15} /></button>
        </div>
      </div>

      <div className="min-h-0 overflow-auto px-6 py-5">
        <div className="relative mx-auto select-none overflow-hidden border border-line bg-black"
             style={{ aspectRatio: `${W} / ${H}`, maxHeight: '62vh', maxWidth: '100%', width: H > W ? 'auto' : '100%', height: H > W ? '62vh' : 'auto' }}>
          <img ref={imageRef} alt="" className={`absolute inset-0 h-full w-full object-fill ${hasFrame ? '' : 'hidden'}`} draggable={false} />
          {!hasFrame && <div className="absolute inset-0 flex items-center justify-center text-[13px] faint">{loading ? 'Loading…' : 'Waiting for video…'}</div>}
          <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
               onPointerDown={onBackgroundDown} onPointerMove={onMove} onPointerUp={endDrag} onPointerCancel={endDrag}
               onContextMenu={e => e.preventDefault()}>
            <rect data-bg="1" x={0} y={0} width={W} height={H} fill="transparent" />

            {/* finished ignore areas (grey, hatched look), each with a delete button */}
            {ignore.map((zone, zi) => {
              const [cx, cy] = centroid(zone)
              return <g key={`ig${zi}`}>
                <polygon points={zone.map(p => p.join(',')).join(' ')} fill="rgba(100,116,139,0.45)" stroke="#94a3b8" strokeWidth={2}
                         strokeDasharray="8 6" vectorEffect="non-scaling-stroke" pointerEvents="none" />
                {editingIgnore && <g className="cursor-pointer" onPointerDown={e => { e.stopPropagation(); setIgnore(z => z.filter((_, k) => k !== zi)) }}>
                  <circle cx={cx} cy={cy} r={handle * 1.1} fill="#030506" stroke="#ff5b4f" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                  <path d={`M${cx - handle * 0.5},${cy - handle * 0.5} L${cx + handle * 0.5},${cy + handle * 0.5} M${cx + handle * 0.5},${cy - handle * 0.5} L${cx - handle * 0.5},${cy + handle * 0.5}`}
                        stroke="#ff5b4f" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                  <title>Delete this ignore area</title></g>}
              </g>
            })}

            {/* the fence, shown faintly while editing ignore areas */}
            {editingIgnore && fencePts.length >= 2 && (fenceType === 'zone' && fencePts.length >= 3
              ? <polygon points={fencePts.map(p => p.join(',')).join(' ')} fill="none" stroke="#e879f9" strokeOpacity={0.4} strokeWidth={2} vectorEffect="non-scaling-stroke" pointerEvents="none" />
              : <polyline points={fencePts.map(p => p.join(',')).join(' ')} fill="none" stroke="#e879f9" strokeOpacity={0.4} strokeWidth={2} vectorEffect="non-scaling-stroke" pointerEvents="none" />)}

            {/* the shape being edited */}
            {closed && <polygon points={pointsAttr} fill={editingIgnore ? 'rgba(100,116,139,0.35)' : 'rgba(217,70,239,0.18)'} stroke="none" pointerEvents="none" />}
            {points.length >= 2 && (closed
              ? <polygon points={pointsAttr} fill="none" stroke={stroke} strokeWidth={3} vectorEffect="non-scaling-stroke" pointerEvents="none" />
              : <polyline points={pointsAttr} fill="none" stroke={stroke} strokeWidth={3} vectorEffect="non-scaling-stroke" pointerEvents="none" />)}
            {segments.map(i => {
              const a = points[i], b = points[(i + 1) % points.length]
              return <circle key={`m${i}`} cx={(a[0] + b[0]) / 2} cy={(a[1] + b[1]) / 2} r={handle * 0.6}
                             fill="rgba(3,5,6,0.85)" stroke={stroke} strokeWidth={1.5} vectorEffect="non-scaling-stroke"
                             className="cursor-copy" onPointerDown={onMidpointDown(i)}><title>Drag to add a bend</title></circle>
            })}
            {points.map((p, i) => <circle key={`v${i}`} cx={p[0]} cy={p[1]} r={handle}
                                          fill={i === 0 ? '#22d3cb' : stroke} stroke="#030506" strokeWidth={2} vectorEffect="non-scaling-stroke"
                                          className="cursor-move" onPointerDown={onVertexDown(i)} onContextMenu={removeVertex(i)} onDoubleClick={removeVertex(i)}>
              <title>Drag to move · right-click or double-click to delete</title></circle>)}
          </svg>
        </div>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[12.5px]">
          <span className={(editingIgnore ? draftValid : fenceValid) ? 'text-bone' : 'text-caution'}>{status}</span>
          <span className="inline-flex items-center gap-1.5 faint"><MousePointerClick size={13} />Click: add point · Drag point: move · Drag small dot: add bend · Right-click point: delete</span>
        </div>
        {error && <p role="alert" className="mt-2 text-[13px] text-breach">{error}</p>}
      </div>

      <div className="flex items-center justify-between gap-4 border-t border-line px-6 py-4">
        <span className="max-w-[60ch] text-[12.5px] faint">{editingIgnore
          ? 'People or vehicles mostly inside a grey area are ignored (photos on posters, TV screens, mirrors). Someone standing in front is still detected.'
          : `${fenceType === 'zone' ? 'Alert while a person’s feet are inside the shaded area.' : 'Alert when a person’s feet cross the line.'} The magenta outline in the video is the fence currently active.`}</span>
        <div className="flex shrink-0 gap-2">
          <button onClick={onClose} className="btn-quiet">Cancel</button>
          <button disabled={saving || !canSave} onClick={() => void save()} className="btn-primary min-w-36">{saving ? <LoaderCircle className="animate-spin" size={16} /> : <Save size={16} />}{saving ? 'Saving fence' : 'Save fence'}</button>
        </div>
      </div>
    </section>
  </div>
}
