import { useRef, type CSSProperties } from 'react'
import SmokeCanvas from './SmokeCanvas'

/*
 * Background layers for the About deck. Pure SVG/CSS (plus the existing smoke shader on the
 * closing slide), absolutely positioned behind slide content, animated only while their slide
 * has the `is-in` class. Positions come from a seeded generator so they never jump on re-render.
 */

export type BackdropKind = 'radar' | 'monitors' | 'beams' | 'constellation' | 'circuit' | 'map' | 'wire'
  | 'glyphs' | 'mesh' | 'wave' | 'orbits' | 'ticks' | 'smoke'

function rng(seed: number) {
  let s = seed >>> 0
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 }
}
const v = (o: Record<string, string | number>) => o as CSSProperties

/* ── shared HUD: viewfinder corners, REC marker, watermark number, scan line ── */
export function Hud({ index, total, coord }: { index: number; total: number; coord: string }) {
  return <div className="bg hud" aria-hidden>
    <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
    <span className="rec mono"><b />REC · CAM {String(index + 1).padStart(2, '0')} · {coord}</span>
    <span className="watermark mono">{String(index + 1).padStart(2, '0')}</span>
    <span className="hud-total mono">{String(index + 1).padStart(2, '0')}/{String(total).padStart(2, '0')}</span>
    <i className="scanline" />
  </div>
}

const R = rng(7)
const MONITORS = Array.from({ length: 24 }, () => ({ d: R() * 6, dur: 2 + R() * 4 }))
const STARS = Array.from({ length: 34 }, () => ({ x: R() * 100, y: R() * 100 }))
const LINKS = STARS.flatMap((a, i) => STARS.slice(i + 1).map((b, j) => ({ a, b, k: i * 100 + j })))
  .filter(({ a, b }) => Math.hypot(a.x - b.x, (a.y - b.y) * 0.6) < 15).slice(0, 60)
const GLYPH_COLS = Array.from({ length: 14 }, (_, i) => ({
  x: i * 7.4 + R() * 2, dur: 14 + R() * 14, d: -R() * 20,
  text: Array.from({ length: 30 }, () => 'ABCDEFGHJKLMNPRSTUVWXYZ0123456789'[(R() * 33) | 0]).join('\n'),
}))
const TICKS = Array.from({ length: 22 }, () => ({ x: R() * 100, dur: 7 + R() * 8, d: -R() * 14, s: 0.6 + R() * 1.2 }))
// face oval + eyes + nose + mouth landmark-ish points
const FACE: [number, number][] = [
  ...Array.from({ length: 22 }, (_, i) => { const t = (i / 22) * Math.PI * 2; return [50 + Math.cos(t) * 30, 52 + Math.sin(t) * 40] as [number, number] }),
  ...[[38, 42], [34, 43], [42, 43], [38, 45], [62, 42], [58, 43], [66, 43], [62, 45]] as [number, number][],
  [50, 48], [50, 56], [46, 60], [54, 60], [42, 70], [46, 72], [50, 73], [54, 72], [58, 70],
]
const FACE_LINKS = FACE.flatMap((a, i) => FACE.slice(i + 1).map((b, j) => ({ a, b, k: `${i}-${j}` })))
  .filter(({ a, b }) => Math.hypot(a[0] - b[0], a[1] - b[1]) < 12)

export function Backdrop({ kind }: { kind: BackdropKind }) {
  const smokeProgress = useRef(0.08)
  switch (kind) {
    case 'radar':
      return <div className="bg bg-radar" aria-hidden><i className="sweep" /><i className="blip b1" /><i className="blip b2" /><i className="blip b3" /></div>

    case 'monitors':
      return <div className="bg bg-monitors" aria-hidden>
        {MONITORS.map((m, i) => <span key={i} style={v({ '--d': `${m.d}s`, '--dur': `${m.dur}s` })}><em className="mono">CAM {String(i + 1).padStart(2, '0')}</em></span>)}
      </div>

    case 'beams':
      return <div className="bg bg-beams" aria-hidden><i /><i /><i /></div>

    case 'constellation':
      return <svg className="bg bg-constellation" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
        <g className="drift">
          {LINKS.map(({ a, b, k }) => <line key={k} x1={a.x} y1={a.y} x2={b.x} y2={b.y} />)}
          {STARS.map((s, i) => <circle key={i} cx={s.x} cy={s.y} r={0.35} style={v({ '--d': `${(i % 7) * 0.4}s` })} />)}
        </g>
      </svg>

    case 'circuit':
      return <svg className="bg bg-circuit" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice" aria-hidden>
        {['M0,120 H260 L320,180 H620 L680,120 H1200', 'M0,690 H180 L240,630 H520 L580,690 H900 L960,630 H1200',
          'M140,0 V220 L200,280 V520', 'M1060,800 V560 L1000,500 V300 L1060,240 V0', 'M420,800 V700 L480,640 H760',
        ].map((d, i) => <g key={i}><path d={d} className="trace" /><path d={d} className="packet" style={v({ '--d': `${-i * 1.3}s` })} /></g>)}
        {[[320, 180], [620, 180], [240, 630], [580, 690], [200, 280], [1000, 500], [480, 640]].map(([x, y], i) => <rect key={i} x={x - 4} y={y - 4} width={8} height={8} className="pad" />)}
      </svg>

    case 'map':
      return <svg className="bg bg-map" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice" aria-hidden>
        {Array.from({ length: 9 }, (_, i) => <path key={i} className="contour" d={`M-50,${120 + i * 70} C250,${60 + i * 75} 450,${200 + i * 60} 700,${110 + i * 72} S1050,${40 + i * 80} 1260,${150 + i * 66}`} />)}
        <path className="route" d="M220,560 C420,420 640,640 980,300" />
        <circle className="pin" cx="220" cy="560" r="7" /><circle className="pin" cx="980" cy="300" r="7" />
        <circle className="traveller" r="5"><animateMotion dur="7s" repeatCount="indefinite" path="M220,560 C420,420 640,640 980,300" /></circle>
        <text x="236" y="590" className="maplbl">CAM 1</text><text x="996" y="290" className="maplbl">CAM 2</text>
      </svg>

    case 'wire':
      return <div className="bg bg-wire" aria-hidden><i className="mesh" /><svg viewBox="0 0 100 100" preserveAspectRatio="none"><rect x="6" y="10" width="88" height="80" className="ants" /></svg></div>

    case 'glyphs':
      return <div className="bg bg-glyphs mono" aria-hidden>
        {GLYPH_COLS.map((c, i) => <pre key={i} style={v({ left: `${c.x}%`, '--dur': `${c.dur}s`, '--d': `${c.d}s` })}>{c.text}{'\n'}{c.text}</pre>)}
        <i className="ocr-bar" />
      </div>

    case 'mesh':
      return <svg className="bg bg-mesh" viewBox="0 0 100 100" preserveAspectRatio="xMaxYMid meet" aria-hidden>
        {FACE_LINKS.map(({ a, b, k }) => <line key={k} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />)}
        {FACE.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={0.7} style={v({ '--d': `${(i % 9) * 0.25}s` })} />)}
        <rect x="16" y="8" width="68" height="88" className="facebox" />
      </svg>

    case 'wave':
      return <svg className="bg bg-wave" viewBox="0 0 1200 400" preserveAspectRatio="none" aria-hidden>
        {Array.from({ length: 9 }, (_, i) => <line key={i} x1="0" x2="1200" y1={i * 50} y2={i * 50} className="gl" />)}
        <g className="scroll">
          {[0, 1].map(k => <path key={k} transform={`translate(${k * 1200},0)`}
            d={'M0,220 ' + Array.from({ length: 48 }, (_, i) => `L${(i + 1) * 25},${200 + Math.sin(i * 0.9) * 60 * Math.sin(i * 0.23) + (i % 7 === 0 ? -70 : 0)}`).join(' ')} />)}
        </g>
      </svg>

    case 'orbits':
      return <div className="bg bg-orbits" aria-hidden>
        {[1, 2, 3, 4].map(i => <i key={i} className={`orbit-ring o${i}`}><b /></i>)}
      </div>

    case 'ticks':
      return <div className="bg bg-ticks" aria-hidden>
        {TICKS.map((t, i) => <span key={i} style={v({ left: `${t.x}%`, '--dur': `${t.dur}s`, '--d': `${t.d}s`, '--s': t.s })}>✓</span>)}
      </div>

    case 'smoke':
      return <div className="bg bg-smoke" aria-hidden><SmokeCanvas progress={smokeProgress} className="smoke-canvas" /></div>
  }
}
