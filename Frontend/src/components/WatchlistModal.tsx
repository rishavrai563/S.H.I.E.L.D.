import { useEffect, useState, type FormEvent } from 'react'
import { Car, LoaderCircle, Plus, Trash2, Upload, UserRound, X } from 'lucide-react'
import { useWatchlist } from '../hooks/useWatchlist'

interface Props { open: boolean; onClose: () => void }

const FACE_CATEGORIES = ['SUSPECT', 'WANTED', 'BLACKLISTED', 'MISSING', 'VIP', 'AUTHORIZED']
const PLATE_TAGS = ['BLACKLISTED', 'STOLEN', 'WANTED', 'RESTRICTED', 'VIP']
const THREATS = ['SUSPECT', 'WANTED', 'BLACKLISTED', 'STOLEN', 'RESTRICTED', 'SMUGGLER', 'INTRUDER']
const tagClass = (value: string) => THREATS.includes(value) ? 'tag-critical' : value === 'AUTHORIZED' ? 'tag-normal' : 'tag-warning'
const label = (value: string) => value.charAt(0) + value.slice(1).toLowerCase()

export default function WatchlistModal({ open, onClose }: Props) {
  const [tab, setTab] = useState<'faces' | 'plates'>('faces')
  const wl = useWatchlist()
  const [name, setName] = useState(''), [category, setCategory] = useState('SUSPECT'), [notes, setNotes] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [plate, setPlate] = useState(''), [tag, setTag] = useState('BLACKLISTED'), [plateNotes, setPlateNotes] = useState('')
  const { refresh } = wl

  useEffect(() => { if (open) void refresh() }, [open, refresh])
  if (!open) return null

  const submitFace = async (event: FormEvent) => {
    event.preventDefault()
    if (!name.trim() || !files.length) return
    if (await wl.enrollFace(name.trim(), category, notes, files)) { setName(''); setNotes(''); setFiles([]) }
  }
  const submitPlate = async (event: FormEvent) => {
    event.preventDefault()
    if (!plate.trim()) return
    if (await wl.addPlate(plate.trim(), tag, plateNotes)) { setPlate(''); setPlateNotes('') }
  }

  return <div className="modal-backdrop" data-lenis-prevent role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <section role="dialog" aria-modal="true" aria-labelledby="watchlist-title" className="modal flex max-h-[92vh] max-w-3xl flex-col">
      <div className="flex items-start justify-between gap-4 px-6 pt-6">
        <div><h2 id="watchlist-title" className="modal-title">Watchlists</h2><p className="mt-1.5 text-[13px] muted">Every camera checks faces and number plates against these lists.</p></div>
        <button className="icon-button" onClick={onClose} aria-label="Close"><X size={17} /></button>
      </div>

      <div role="tablist" className="mt-4 flex border-b border-line px-6">
        {([['faces', 'Faces', UserRound, wl.faces.length], ['plates', 'Number plates', Car, wl.plates.length]] as const).map(([id, text, Icon, count]) =>
          <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)} className="tab inline-flex items-center gap-2"><Icon size={15} />{text}<span className="mono text-[11px] faint">{count}</span></button>)}
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-6">
        {wl.error && <p role="alert" className="mb-4 border border-breach/40 bg-breach/10 p-3 text-[13px] text-breach">{wl.error}</p>}

        {tab === 'faces' ? <>
          <form onSubmit={e => void submitFace(e)}>
            <div className="slab-dark grid gap-3 sm:grid-cols-2">
              <label className="field-label">Name<input value={name} onChange={e => setName(e.target.value)} placeholder="Full name or alias" className="field" /></label>
              <label className="field-label">Category<select value={category} onChange={e => setCategory(e.target.value)} className="field">{FACE_CATEGORIES.map(c => <option key={c} value={c}>{label(c)}</option>)}</select></label>
              <label className="field-label sm:col-span-2">Notes<input value={notes} onChange={e => setNotes(e.target.value)} placeholder="Case reference, last known location" className="field" /></label>
            </div>
            <label className="slab-light flex cursor-pointer items-center gap-3 text-[13px]">
              <Upload size={16} className="shrink-0" />
              <span className="flex-1 truncate">{files.length ? files.map(f => f.name).join(', ') : 'Choose reference photos. Clear, front-facing photos from a few angles match best.'}</span>
              <input type="file" accept="image/jpeg,image/png" multiple className="sr-only" onChange={e => setFiles(Array.from(e.target.files ?? []))} />
            </label>
            <button disabled={wl.busy || !name.trim() || !files.length} className="btn-primary w-full">{wl.busy ? <LoaderCircle className="animate-spin" size={16} /> : <Plus size={16} />}Enrol face</button>
          </form>
          <ul className="mt-6 grid gap-px bg-line sm:grid-cols-2">
            {wl.faces.map(f => <li key={f.identity_id} className="flex items-center gap-3 bg-panel p-3">
              {f.thumbnail ? <img src={`data:image/jpeg;base64,${f.thumbnail}`} alt="" className="h-14 w-14 object-cover grayscale-[35%]" /> : <div className="grid h-14 w-14 place-items-center bg-raised faint"><UserRound size={18} /></div>}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2"><span className="truncate text-[14px] font-medium">{f.name}</span><span className={`tag ${tagClass(f.category)}`}>{label(f.category)}</span></div>
                <p className="mt-1 truncate text-[12px] faint"><span className="mono">{f.identity_id}</span>, {f.num_references} photo{f.num_references === 1 ? '' : 's'}{f.notes ? `. ${f.notes}` : ''}</p>
              </div>
              <button title="Remove from watchlist" aria-label={`Remove ${f.name}`} onClick={() => void wl.removeFace(f.identity_id)} className="icon-button hover:text-breach"><Trash2 size={15} /></button>
            </li>)}
          </ul>
          {!wl.faces.length && <p className="mt-6 text-center text-[13px] faint">No faces enrolled. Add a photo above to start matching.</p>}
        </> : <>
          <form onSubmit={e => void submitPlate(e)}>
            <div className="slab-dark grid gap-3 sm:grid-cols-[1fr_170px]">
              <label className="field-label">Plate number<input value={plate} onChange={e => setPlate(e.target.value.toUpperCase())} placeholder="MH01AB1234" className="field mono" /></label>
              <label className="field-label">Reason<select value={tag} onChange={e => setTag(e.target.value)} className="field">{PLATE_TAGS.map(t => <option key={t} value={t}>{label(t)}</option>)}</select></label>
              <label className="field-label sm:col-span-2">Notes<input value={plateNotes} onChange={e => setPlateNotes(e.target.value)} placeholder="Why this vehicle is listed" className="field" /></label>
              <p className="text-[12px] faint sm:col-span-2">Matches still fire when the camera misreads look-alike characters (0 and O, 1 and I, 8 and B) or one other character.</p>
            </div>
            <button disabled={wl.busy || !plate.trim()} className="btn-primary w-full"><Plus size={16} />Add plate</button>
          </form>
          <ul className="mt-6 border-t border-line">
            {wl.plates.map(p => <li key={p.plate} className="flex items-center gap-3 border-b border-line py-2.5">
              <span className="mono w-32 text-[15px]">{p.plate}</span>
              <span className={`tag ${tagClass(p.tag)}`}>{label(p.tag)}</span>
              <span className="flex-1 truncate text-[12.5px] faint">{p.notes}</span>
              <button title="Remove from watchlist" aria-label={`Remove ${p.plate}`} onClick={() => void wl.removePlate(p.plate)} className="icon-button hover:text-breach"><Trash2 size={15} /></button>
            </li>)}
            {!wl.plates.length && <li className="py-6 text-center text-[13px] faint">No plates listed.</li>}
          </ul>
          <h3 className="mb-2 mt-8 text-[15px] font-semibold">Recently read plates</h3>
          <ul className="border-t border-line">
            {wl.reads.map((r, i) => <li key={`${r.plate}-${r.timestamp}-${i}`} className="flex items-center gap-3 border-b border-line py-2 text-[13px]">
              <span className="mono w-32">{r.plate}</span>
              <span className="faint">{r.vehicle_class || 'Vehicle'} on <span className="mono">{r.camera_id}</span></span>
              {r.watchlist && <span className={`tag ${tagClass(r.watchlist.tag)}`}>{label(r.watchlist.tag)}</span>}
              <time className="mono ml-auto text-[12px] faint">{new Date(r.timestamp * 1000).toLocaleTimeString()}</time>
            </li>)}
            {!wl.reads.length && <li className="py-5 text-center text-[13px] faint">No plates confirmed yet. A plate is confirmed after two matching reads.</li>}
          </ul>
        </>}
      </div>
    </section>
  </div>
}
