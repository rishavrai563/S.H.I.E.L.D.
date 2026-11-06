import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import {
  ArrowRight, BellRing, Car, Cctv, Check, ChevronDown, ChevronUp, Cpu, Database, EyeOff, Fence,
  Footprints, Gauge, Moon, MonitorSmartphone, Radar, ScanFace, ScanLine, ShieldCheck, Siren, Users, Webhook, WifiOff,
} from 'lucide-react'
import PixelArrow from './PixelArrow'
import { scrollToId } from '../lib/smoothScroll'
import { Backdrop, Hud, type BackdropKind } from './DeckBackdrops'

/*
 * About: the project told as a scroll-driven deck. Each slide is one screen; it "plays"
 * (staggered reveals, drawings, counters, demos) when it enters the view and resets when it
 * leaves, so scrolling back replays it like a presentation. One rAF scroll loop drives
 * everything: active slide, slide visibility, the pinned architecture drawing and the
 * progress bar. All figures are ones measured on this project.
 */

const SLIDES = ['Title', 'Problem', 'Solution', 'Capabilities', 'Architecture', 'Cross-camera', 'Fences',
  'ANPR', 'Face recognition', 'Performance', 'Integration', 'Outcomes', 'Next'] as const

const d = (n: number) => ({ '--d': n } as CSSProperties)

/** Letters rise one by one out of a masked line. */
function SplitText({ text, className = '', delay = 0 }: { text: string; className?: string; delay?: number }) {
  return <span className={`split ${className}`} aria-label={text}>
    {[...text].map((ch, i) => <span key={i} aria-hidden className="ch" style={d(delay + i * 0.45)}>{ch === ' ' ? ' ' : ch}</span>)}
  </span>
}

/** Counts up from 0 when its slide becomes active. */
function CountUp({ to, active, decimals = 0, suffix = '' }: { to: number; active: boolean; decimals?: number; suffix?: string }) {
  const [v, setV] = useState(0)
  useEffect(() => {
    if (!active) { setV(0); return }
    let raf = 0; const start = performance.now(), dur = 1400
    const step = (t: number) => {
      const k = Math.min(1, (t - start) / dur), e = 1 - Math.pow(1 - k, 3)
      setV(to * e); if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [active, to])
  return <>{v.toFixed(decimals)}{suffix}</>
}

/** Characters decode from noise into the final text (OCR feel). */
function Scramble({ text, active }: { text: string; active: boolean }) {
  const [out, setOut] = useState(text.replace(/[A-Z0-9]/g, '·'))
  useEffect(() => {
    if (!active) { setOut(text.replace(/[A-Z0-9]/g, '·')); return }
    const pool = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'
    let frame = 0
    const id = window.setInterval(() => {
      frame++
      setOut([...text].map((c, i) => (c === ' ' ? ' ' : frame > 6 + i * 2.2 ? c : pool[(Math.random() * pool.length) | 0])).join(''))
      if (frame > 8 + text.length * 2.2) window.clearInterval(id)
    }, 55)
    return () => window.clearInterval(id)
  }, [active, text])
  return <>{out}</>
}

const BACKDROPS: BackdropKind[] = ['radar', 'monitors', 'beams', 'constellation', 'circuit', 'map', 'wire',
  'glyphs', 'mesh', 'wave', 'orbits', 'ticks', 'smoke']

function Slide({ index, children, className = '', label }: { index: number; children: ReactNode; className?: string; label: string }) {
  const coord = `${(28.6139 + index * 0.0137).toFixed(4)}N ${(77.209 + index * 0.0211).toFixed(4)}E`
  const layers = <><Backdrop kind={BACKDROPS[index]} /><Hud index={index} total={SLIDES.length} coord={coord} /></>
  return <section id={`slide-${index}`} data-slide={index} aria-label={label} className={`slide slide-bg-${BACKDROPS[index]} ${className}`}>
    {className.includes('pinned') ? <div className="pin-bg">{layers}</div> : layers}
    <span className="slide-num mono" aria-hidden>{String(index + 1).padStart(2, '0')} / {String(SLIDES.length).padStart(2, '0')} · {label}</span>
    {children}
  </section>
}

export default function About({ onOpenConsole }: { onOpenConsole: () => void }) {
  const rootRef = useRef<HTMLDivElement | null>(null)
  const barRef = useRef<HTMLDivElement | null>(null)
  const [active, setActive] = useState(0)
  const activeRef = useRef(0)

  // One scroll loop: which slide is active / in view, pinned-slide progress, page progress
  useEffect(() => {
    let raf = 0
    const update = () => {
      raf = 0
      const root = rootRef.current
      if (!root) return
      const vh = window.innerHeight
      root.querySelectorAll<HTMLElement>('.slide').forEach(el => {
        const r = el.getBoundingClientRect()
        const i = Number(el.dataset.slide)
        el.classList.toggle('is-in', r.top < vh * 0.72 && r.bottom > vh * 0.28)
        if (r.top <= vh * 0.5 && r.bottom >= vh * 0.5 && activeRef.current !== i) { activeRef.current = i; setActive(i) }
        if (el.classList.contains('pinned')) {
          const span = r.height - vh
          el.style.setProperty('--p', String(Math.min(1, Math.max(0, span > 0 ? -r.top / span : 1))))
        }
      })
      const total = root.scrollHeight - vh
      const rt = root.getBoundingClientRect()
      if (barRef.current) barRef.current.style.transform = `scaleX(${Math.min(1, Math.max(0, -rt.top / Math.max(1, total)))})`
    }
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(update) }
    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('scroll', onScroll); window.removeEventListener('resize', onScroll) }
  }, [])

  // Presenter keys: next / previous slide
  useEffect(() => {
    const go = (i: number) => scrollToId(`slide-${Math.max(0, Math.min(SLIDES.length - 1, i))}`, 0, 1.1)
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return
      if (document.querySelector('.modal-backdrop')) return
      if (['ArrowDown', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); go(activeRef.current + 1) }
      if (['ArrowUp', 'PageUp'].includes(e.key)) { e.preventDefault(); go(activeRef.current - 1) }
      if (e.key === 'Home') { e.preventDefault(); go(0) }
      if (e.key === 'End') { e.preventDefault(); go(SLIDES.length - 1) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const is = (i: number) => active === i

  return <div ref={rootRef} className="deck">
    <div className="deck-progress" aria-hidden><div ref={barRef} /></div>
    <nav className="deck-dots" aria-label="Slides">
      {SLIDES.map((name, i) => <button key={name} onClick={() => scrollToId(`slide-${i}`, 0, 1.1)} aria-label={`Slide ${i + 1}: ${name}`}
        aria-current={active === i} className={active === i ? 'on' : ''}><span>{name}</span></button>)}
    </nav>
    <div className="deck-keys mono" aria-hidden>
      <button onClick={() => scrollToId(`slide-${Math.max(0, active - 1)}`, 0, 1.1)}><ChevronUp size={14} /></button>
      <span>{String(active + 1).padStart(2, '0')}</span>
      <button onClick={() => scrollToId(`slide-${Math.min(SLIDES.length - 1, active + 1)}`, 0, 1.1)}><ChevronDown size={14} /></button>
    </div>

    {/* 1 — Title: letters rise, radar rings pulse */}
    <Slide index={0} label="Title" className="slide-title">
      <div className="radar" aria-hidden><i /><i /><i /><i /></div>
      <p className="kicker rv rv-up" style={d(0)}>AI-based video analytics · Existing CCTV infrastructure</p>
      <h1 className="deck-hero"><SplitText text="SHIELD" /></h1>
      <p className="deck-sub rv rv-blur" style={d(6)}>IBVAP — Intelligent Border Video Analytics Platform</p>
      <p className="deck-lede rv rv-up" style={d(9)}>AI-based video analytics that turns the CCTV already installed at border outposts, check posts and roads into an intelligent surveillance network, in software.</p>
      <p className="scroll-hint rv rv-up" style={d(12)}><ChevronDown size={16} />Scroll, or use the arrow keys</p>
    </Slide>

    {/* 2 — Problem: statements struck through one after another */}
    <Slide index={1} label="Problem">
      <p className="kicker rv rv-up">The problem</p>
      <h2 className="deck-h2 rv rv-up" style={d(1)}>Cameras record.<br />People watch.</h2>
      <ul className="problems">
        <li className="rv rv-left" style={d(3)}><Cctv size={22} /><div><b className="strike" style={d(6)}>Continuous human observation</b><span>Conventional CCTV only records and shows live video. Detecting anything depends on someone watching every screen.</span></div></li>
        <li className="rv rv-left" style={d(4)}><ScanFace size={22} /><div><b className="strike" style={d(9)}>Specialised hardware</b><span>Face recognition, number-plate reading and smart tracking usually need dedicated cameras and proprietary systems.</span></div></li>
        <li className="rv rv-left" style={d(5)}><WifiOff size={22} /><div><b className="strike" style={d(12)}>Costly at remote posts</b><span>Buying, installing and maintaining that kit across remote border locations is expensive and slow.</span></div></li>
      </ul>
    </Slide>

    {/* 3 — Solution: before/after wipe */}
    <Slide index={2} label="Solution">
      <p className="kicker rv rv-up">Our answer</p>
      <h2 className="deck-h2"><span className="rv rv-up" style={d(1)}>Same cameras.</span><br /><span className="rv rv-wipe tide" style={d(4)}><PixelArrow className="arrow" />New intelligence.</span></h2>
      <div className="before-after">
        <div className="ba-card rv rv-left" style={d(6)}><span className="ba-tag">Before</span><Cctv size={34} /><p>IP / RTSP cameras that record and stream</p></div>
        <ArrowRight className="ba-arrow rv rv-scale" style={d(8)} size={34} />
        <div className="ba-card ba-after rv rv-right" style={d(9)}><span className="ba-tag">With SHIELD</span><ShieldCheck size={34} /><p>Detect, recognise, track and alert on the same video, using software and one GPU</p></div>
      </div>
      <p className="deck-lede rv rv-up" style={d(11)}>No dedicated FRS, ANPR or smart-camera hardware. Works with CCTV, IP cameras, NVR streams, and even phones and laptop webcams.</p>
    </Slide>

    {/* 4 — Capabilities: cards flip in, staggered */}
    <Slide index={3} label="Capabilities">
      <p className="kicker rv rv-up">What it does</p>
      <h2 className="deck-h2 rv rv-up" style={d(1)}>Eight capabilities, one platform</h2>
      <div className="caps">
        {[
          [Users, 'Human detection & tracking', 'YOLO11 + multi-object tracker; one ID per person'],
          [Car, 'Vehicle detection & classification', 'Car, motorcycle, bus, truck with stable vehicle IDs'],
          [ScanFace, 'Face recognition', 'Matches faces against a watchlist gallery'],
          [ScanLine, 'Number plate recognition', 'Reads Indian plates; flags watchlisted vehicles'],
          [Fence, 'Virtual fences', 'Tripwires and restricted zones of any shape'],
          [Footprints, 'Suspicious activity', 'Loitering, sudden running, crowding'],
          [Moon, 'Night-time movement', 'Low-light enhancement; night alerts'],
          [BellRing, 'Real-time alerts & logs', 'Live alert feed, event log, command-centre webhook'],
        ].map(([Icon, title, text], i) => {
          const I = Icon as typeof Users
          return <div key={title as string} className="cap rv rv-flip" style={d(2 + i)}><I size={22} /><b>{title as string}</b><span>{text as string}</span></div>
        })}
      </div>
    </Slide>

    {/* 5 — Architecture: pinned, the pipeline draws itself as you scroll */}
    <Slide index={4} label="Architecture" className="pinned">
      <div className="pin-stage">
        <p className="kicker">How it works</p>
        <h2 className="deck-h2 small">From camera to command centre</h2>
        <svg className="pipeline" viewBox="0 0 1210 470" role="img" aria-label="Pipeline: cameras, stream manager, YOLO11 detection, then vehicles go to tracking and ANPR while people go to re-identification, tracking, cross-camera matching, face recognition and behaviour analysis; all feed the alert manager, the API and the dashboard.">
          {[
            ['M175,235 L235,235', 0.02], ['M390,235 L460,235', 0.10],
            ['M615,235 C650,235 645,110 675,110', 0.18], ['M615,235 C650,235 645,350 675,350', 0.18],
            ['M830,110 L855,110', 0.32], ['M830,350 L855,350', 0.32],
            ['M1010,110 C1035,110 1030,235 1045,235', 0.55], ['M1010,350 C1035,350 1030,235 1045,235', 0.55],
          ].map(([dd, s], i) => <path key={i} d={dd as string} pathLength={1} className="pipe" style={{ '--s': s } as CSSProperties} />)}
          {[
            [20, 205, 'Cameras', 'CCTV · RTSP · phone', 0],
            [235, 205, 'Stream manager', 'capture · night', 0.06],
            [460, 205, 'YOLO11', 'people + vehicles', 0.14],
            [680, 70, 'Vehicles', 'tracker · ANPR', 0.26],
            [855, 70, 'Plate watchlist', 'voting · fuzzy', 0.40],
            [680, 310, 'People', 'Re-ID · tracker', 0.26],
            [855, 310, 'Global IDs · FRS', 'fences · risk', 0.40],
            [1045, 205, 'Alerts', 'API · UI · C&C', 0.66],
          ].map(([x, y, t, sub, s]) => <g key={t as string} className="node" style={{ '--s': s } as CSSProperties} transform={`translate(${x},${y})`}>
            <rect width={155} height="60" rx="0" /><text x="12" y="26" className="nt">{t}</text><text x="12" y="46" className="ns">{sub}</text>
          </g>)}
        </svg>
        <p className="pin-caption">Detection runs on its own GPU thread while tracking, re-identification and risk analysis process the previous frame. Every alert goes to the dashboard, a permanent event log and, optionally, a command-centre webhook.</p>
      </div>
    </Slide>

    {/* 6 — Cross-camera: a person walks from one camera into another and keeps the ID */}
    <Slide index={5} label="Cross-camera tracking">
      <p className="kicker rv rv-up">Multi-camera tracking</p>
      <h2 className="deck-h2 rv rv-up" style={d(1)}>One person, one ID, every camera</h2>
      <div className={`handoff rv rv-scale ${is(5) ? 'play' : ''}`} style={d(3)}>
        <div className="cam"><span className="cam-name mono">CAM 1 · Laptop</span><div className="walker wa"><em>G5</em></div></div>
        <div className="cam"><span className="cam-name mono">CAM 2 · Phone</span><div className="walker wb"><em className="tmp">G9</em><em className="real">G5</em></div></div>
        <div className="toast mono"><BellRing size={13} />G5 moved Laptop → Phone</div>
      </div>
      <div className="stat-row">
        <div className="stat rv rv-up" style={d(5)}><span className="stat-v mono"><CountUp to={97.9} decimals={1} suffix="%" active={is(5)} /></span><span>re-identification accuracy (rank-1) on our crowd footage, up from 87.6%</span></div>
        <div className="stat rv rv-up" style={d(6)}><span className="stat-v mono">0.78</span><span>stricter bar for cross-camera matches: a new ID is safer than merging two people</span></div>
      </div>
    </Slide>

    {/* 7 — Fences: zone draws itself, intruder turns it red; car crosses a tripwire */}
    <Slide index={6} label="Virtual fences">
      <p className="kicker rv rv-up">Intrusion detection</p>
      <h2 className="deck-h2 rv rv-up" style={d(1)}>Draw a fence. Any shape.</h2>
      <div className={`fence-demo rv rv-blur ${is(6) ? 'play' : ''}`} style={d(3)}>
        <svg viewBox="0 0 640 300" aria-hidden>
          <path className="trip" d="M30,250 L250,215 L330,250" pathLength={1} />
          <polygon className="zone" points="380,60 600,80 580,250 400,230 360,150" pathLength={1} />
          <g className="intruder"><circle r="9" /><text y="-16" textAnchor="middle">G7</text></g>
          <g className="truck"><rect x="-22" y="-12" width="44" height="24" /><text y="-18" textAnchor="middle">V3</text></g>
        </svg>
        <div className="fence-labels mono"><span className="l-trip"><Siren size={12} />Tripwire crossed by truck V3 · CRITICAL</span><span className="l-zone"><Radar size={12} />Suspicious: G7 in restricted zone · until they leave</span></div>
      </div>
      <div className="three-col">
        <p className="rv rv-up" style={d(5)}><b>Tripwire</b> — any number of segments. Crossing it flags the person or vehicle at once.</p>
        <p className="rv rv-up" style={d(6)}><b>Restricted zone</b> — any polygon. Anyone inside is flagged suspicious until they step out; entry and exit are logged.</p>
        <p className="rv rv-up" style={d(7)}><b>Ignore area</b> — mask posters, TV screens and mirrors so pictures of people never raise alerts.</p>
      </div>
    </Slide>

    {/* 8 — ANPR: plate decodes, steps tick in */}
    <Slide index={7} label="ANPR">
      <p className="kicker rv rv-up">Number plate recognition</p>
      <h2 className="deck-h2 rv rv-up" style={d(1)}>Reading plates without ANPR cameras</h2>
      <div className="anpr">
        <div className="plate rv rv-scale mono" style={d(3)}><span className="ind">IND</span><Scramble text="MH 12 AB 1234" active={is(7)} /></div>
        <ol className="steps">
          {['Find the plate in the vehicle (edges, contrast, shape)', 'Clean it up: denoise, contrast, deskew',
            'Read it with several OCR passes', 'Fix the Indian format (0↔O, 1↔I, 8↔B)',
            'Confirm with two matching reads', 'Check the watchlist (tolerates OCR slips)'].map((s, i) =>
            <li key={s} className="rv rv-left" style={d(4 + i)}><Check size={14} />{s}</li>)}
        </ol>
      </div>
    </Slide>

    {/* 9 — FRS: similarity bars grow */}
    <Slide index={8} label="Face recognition">
      <p className="kicker rv rv-up">Facial recognition</p>
      <h2 className="deck-h2 rv rv-up" style={d(1)}>Faces matched in software</h2>
      <div className={`bars ${is(8) ? 'play' : ''}`}>
        <div className="threshold"><span className="mono">match threshold</span></div>
        {[['Same person, flipped', 0.91], ['Same person, darker', 0.90], ['Same person, smaller', 0.89], ['Different person', 0.13], ['Different person', -0.10]].map(([l, v], i) =>
          <div key={i} className="bar-row"><span className="bar-label">{l}</span><div className="bar"><i style={{ '--w': `${Math.max(0.02, v as number) * 100}%`, ...d(i) } as CSSProperties} className={(v as number) > 0.4 ? 'hit' : ''} /></div><span className="mono bar-v">{(v as number).toFixed(2)}</span></div>)}
      </div>
      <p className="deck-lede rv rv-up" style={d(8)}>YuNet finds faces, SFace turns each into a signature. A match needs to clear the threshold, beat the next candidate clearly, and repeat twice before a suspect alert is raised.</p>
    </Slide>

    {/* 10 — Performance: counters and before/after bars */}
    <Slide index={9} label="Performance">
      <p className="kicker rv rv-up">Speed</p>
      <h2 className="deck-h2 rv rv-up" style={d(1)}>Real time on a laptop GPU</h2>
      <div className="perf">
        {[[78, 'FPS', 'detection alone, one camera'], [41, 'FPS', 'full pipeline, one camera'], [11, 'FPS', 'each, with four cameras at once'], [30, 'FPS', 'smooth dashboard video']].map(([v, u, l], i) =>
          <div key={l as string} className="perf-cell rv rv-up" style={d(2 + i)}><span className="perf-v mono"><CountUp to={v as number} active={is(9)} /></span><span className="perf-u">{u as string}</span><span className="perf-l">{l as string}</span></div>)}
      </div>
      <div className={`compare ${is(9) ? 'play' : ''}`}>
        <div className="cmp-row"><span>Before optimisation</span><div className="cmp"><i style={{ '--w': '12%' } as CSSProperties} /></div><span className="mono">~5 FPS</span></div>
        <div className="cmp-row"><span>After</span><div className="cmp"><i className="hit" style={{ '--w': '100%' } as CSSProperties} /></div><span className="mono">41 FPS</span></div>
      </div>
      <p className="foot-note rv rv-up" style={d(8)}>Measured on an NVIDIA RTX 3050 6 GB laptop GPU: FP16 detection, detection and analysis in parallel, smart re-identification, a plate-reading budget.</p>
    </Slide>

    {/* 11 — Integration: items orbit in */}
    <Slide index={10} label="Integration">
      <p className="kicker rv rv-up">Deployment</p>
      <h2 className="deck-h2 rv rv-up" style={d(1)}>Fits the existing command chain</h2>
      <div className="orbit">
        <div className="core rv rv-scale" style={d(2)}><ShieldCheck size={30} /><b>SHIELD</b></div>
        {[[Webhook, 'Webhook to C&C', 'JSON per alert, site ID stamped'], [Database, 'Event log', 'permanent, survives restarts'],
          [MonitorSmartphone, 'Any camera', 'CCTV, NVR, IP, phone, webcam'], [WifiOff, 'Works offline', 'local models, no cloud'],
          [Cpu, 'One GPU', 'a single laptop runs a post'], [Gauge, 'REST + WebSocket', 'live video, telemetry, control']].map(([Icon, t, s], i) => {
          const I = Icon as typeof Users
          return <div key={t as string} className={`sat sat-${i} rv rv-orbit`} style={d(4 + i)}><I size={18} /><b>{t as string}</b><span>{s as string}</span></div>
        })}
      </div>
    </Slide>

    {/* 12 — Outcomes: requirement marquee */}
    <Slide index={11} label="Outcomes">
      <p className="kicker rv rv-up">Against the problem statement</p>
      <h2 className="deck-h2 rv rv-up" style={d(1)}>What it delivers</h2>
      <div className="marquee" aria-hidden><div className="marquee-track">
        {[0, 1].map(k => <span key={k}>No dedicated surveillance hardware ✦ Real-time alerts ✦ Face, vehicle and behaviour analytics in software ✦ Faster response ✦ Command-centre integration ✦ Cost-effective ✦ Scalable to remote posts ✦ </span>)}
      </div></div>
      <ul className="outcomes">
        {['Eliminates dependence on expensive dedicated hardware', 'Intelligent monitoring through AI-powered video analytics',
          'Real-time alerts for incidents and border intrusions', 'Facial recognition, vehicle identification and behavioural analytics through software',
          'Better situational awareness and response time', 'Integration with existing command and control systems'].map((t, i) =>
          <li key={t} className="rv rv-left" style={d(2 + i)}><Check size={16} />{t}</li>)}
      </ul>
    </Slide>

    {/* 13 — Status + close */}
    <Slide index={12} label="Status and next steps" className="slide-close">
      <p className="kicker rv rv-up">Where it stands</p>
      <div className="close-grid">
        <div className="rv rv-left" style={d(1)}><span className="stat-v mono"><CountUp to={314} active={is(12)} /></span><span>automated tests passing</span></div>
        <ul className="next rv rv-right" style={d(2)}>
          <li><b>Next</b></li><li>TensorRT acceleration for one-camera 60 FPS</li><li>Tune plate reading on Indian footage</li><li>Tune face thresholds on live footage</li>
        </ul>
      </div>
      <h2 className="deck-hero small"><SplitText text="EVERY CAMERA" delay={4} /><br /><span className="offset"><PixelArrow className="arrow" /><SplitText text="ON WATCH" delay={10} /></span></h2>
      <button className="btn-primary rv rv-up" style={d(16)} onClick={onOpenConsole}>Open the console <ArrowRight size={16} /></button>
      <p className="faint rv rv-up" style={d(18)}><EyeOff size={12} className="inline" /> Built for air-gapped border networks: fonts and models are bundled, nothing calls the cloud.</p>
    </Slide>
  </div>
}
