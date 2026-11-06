import Lenis from 'lenis'

/**
 * Inertial page scrolling (Lenis). Skipped entirely when the operator prefers reduced motion.
 * Nested scroll areas (event list, dialogs) opt out with the `data-lenis-prevent` attribute.
 */
let lenis: Lenis | null = null

export function startSmoothScroll(): () => void {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return () => {}
  lenis = new Lenis({ lerp: 0.085, wheelMultiplier: 0.9, smoothWheel: true })
  let raf = 0
  const loop = (time: number) => { lenis?.raf(time); raf = requestAnimationFrame(loop) }
  raf = requestAnimationFrame(loop)
  return () => { cancelAnimationFrame(raf); lenis?.destroy(); lenis = null }
}

export function scrollToId(id: string, offset = -90, duration = 1.6) {
  const target = document.getElementById(id)
  if (!target) return
  if (lenis) lenis.scrollTo(target, { offset, duration })
  else window.scrollTo({ top: target.getBoundingClientRect().top + window.scrollY + offset, behavior: 'auto' })
}

/** Jump to the top instantly (used when switching between Console and About). */
export function scrollToTop() {
  if (lenis) lenis.scrollTo(0, { immediate: true })
  else window.scrollTo(0, 0)
}
