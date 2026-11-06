import { useEffect, useRef } from 'react'
import SmokeCanvas from './SmokeCanvas'
import PixelArrow from './PixelArrow'
import { scrollToId } from '../lib/smoothScroll'

interface Props { cameraCount: number; liveCount: number }

const clamp01 = (v: number) => Math.min(1, Math.max(0, v))
const ease = (a: number, b: number, v: number) => { const t = clamp01((v - a) / (b - a)); return t * t * (3 - 2 * t) }

/**
 * Scroll-driven opening: the headline sits on a low fog bank; scrolling raises the fog,
 * swallows the headline, surfaces a one-line status, then clears onto the console.
 */
export default function Landing({ cameraCount, liveCount }: Props) {
  const sectionRef = useRef<HTMLElement | null>(null)
  const headRef = useRef<HTMLDivElement | null>(null)
  const midRef = useRef<HTMLParagraphElement | null>(null)
  const cueRef = useRef<HTMLDivElement | null>(null)
  const progress = useRef(0)

  useEffect(() => {
    let raf = 0
    const update = () => {
      raf = 0
      const el = sectionRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const span = rect.height - window.innerHeight
      // No scroll runway (reduced motion makes the landing one screen tall): hold the opening frame.
      const p = span < 50 ? 0 : clamp01(-rect.top / span)
      progress.current = p
      const head = headRef.current, mid = midRef.current, cue = cueRef.current
      if (head) {
        const out = ease(0.08, 0.42, p)
        head.style.opacity = String(1 - out)
        head.style.transform = `translate3d(0, ${-out * 14}vh, 0) scale(${1 + out * 0.06})`
        head.style.filter = `blur(${out * 10}px)`
      }
      if (mid) {
        const inn = ease(0.42, 0.56, p), gone = ease(0.72, 0.86, p)
        mid.style.opacity = String(inn * (1 - gone))
        mid.style.transform = `translate3d(0, ${(1 - inn) * 24 - gone * 24}px, 0)`
        mid.style.letterSpacing = `${0.02 + (1 - inn) * 0.18}em`
      }
      if (cue) cue.style.opacity = String(1 - ease(0.02, 0.12, p))
    }
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(update) }
    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('scroll', onScroll); window.removeEventListener('resize', onScroll) }
  }, [])

  return <section ref={sectionRef} className="landing" aria-label="Introduction">
    <div className="landing-stage">
      <div className="landing-fallback" aria-hidden />
      <SmokeCanvas progress={progress} className="landing-smoke" />
      <div className="landing-grid" aria-hidden />

      <div ref={headRef} className="landing-head">
        <h2 className="landing-line">Every camera</h2>
        <p className="landing-line offset"><PixelArrow className="arrow" />on watch</p>
        <p className="landing-note">SHIELD turns the CCTV already standing at your border posts into a surveillance network. It recognises faces and number plates, guards virtual fences and raises alerts, in software.</p>
      </div>

      <p ref={midRef} className="landing-mid" aria-hidden>{liveCount ? `${liveCount} of ${cameraCount} cameras reporting` : `${cameraCount} cameras on the line`}</p>

      <div ref={cueRef} className="landing-cue">
        <button onClick={() => scrollToId('console')} className="pill pill-ghost">Go to the console</button>
        <span className="faint">or scroll</span>
      </div>
    </div>
  </section>
}
