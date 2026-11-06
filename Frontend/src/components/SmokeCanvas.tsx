import { useEffect, useRef, type MutableRefObject } from 'react'

/**
 * Volumetric-looking smoke drawn by a single fragment shader (domain-warped fBm noise).
 * No textures or video, so nothing extra is downloaded on a low-bandwidth border network.
 *
 * `progress` (0 → 1, scroll-driven) moves the fog: a low bank at rest, rising to fill the
 * view mid-scroll, then thinning out so the console underneath shows through.
 */

const VERT = `attribute vec2 a; void main(){ gl_Position = vec4(a, 0.0, 1.0); }`

const FRAG = `
precision highp float;
uniform vec2 uRes;
uniform float uTime;
uniform float uProgress;
uniform vec2 uMouse;

float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
  for (int i = 0; i < 6; i++) { v += a * noise(p); p = m * p; a *= 0.5; }
  return v;
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 p = uv * vec2(uRes.x / uRes.y, 1.0) * 2.0;
  float t = uTime * 0.045;
  float pr = uProgress;

  // Two levels of domain warping give the slow, curling body of smoke.
  vec2 q = vec2(fbm(p + vec2(0.0, -t)), fbm(p + vec2(5.2, 1.3) + t * 0.6));
  vec2 r = vec2(fbm(p + 3.2 * q + vec2(1.7, 9.2) + t * 1.2 + uMouse * 0.35),
                fbm(p + 3.2 * q + vec2(8.3, 2.8) - t * 0.9));
  float f = fbm(p + 2.8 * r + vec2(0.0, -pr * 2.2 - t * 1.5));

  // Fog bank: its top edge climbs with scroll; then everything thins away.
  float top = mix(0.42, 1.35, smoothstep(0.0, 0.55, pr));
  float bank = smoothstep(top, top - 0.85, uv.y);
  float fade = 1.0 - smoothstep(0.62, 0.98, pr);
  float haze = smoothstep(0.5, 1.0, f) * 0.22;
  float d = clamp((smoothstep(0.32, 0.92, f) * bank + haze) * fade, 0.0, 1.0);

  vec3 abyss = vec3(0.012, 0.03, 0.035);
  vec3 tide  = vec3(0.10, 0.27, 0.28);
  vec3 bone  = vec3(0.86, 0.83, 0.74);
  vec3 col = mix(abyss, tide, smoothstep(0.25, 0.7, f));
  col = mix(col, bone, smoothstep(0.6, 0.98, f + r.x * 0.18) * 0.62);
  // Light catching the upper edge of the bank, like sun on cloud tops.
  col += bone * 0.08 * smoothstep(0.0, 0.25, bank) * (1.0 - smoothstep(0.25, 0.7, bank));

  gl_FragColor = vec4(col * d, d); // premultiplied alpha
}`

interface Props {
  progress: MutableRefObject<number>
  className?: string
}

export default function SmokeCanvas({ progress, className = '' }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const gl = canvas.getContext('webgl', { premultipliedAlpha: true, antialias: false, alpha: true, powerPreference: 'low-power' })
    if (!gl) return // No WebGL: the CSS fallback gradient on the parent stays visible.

    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!
      gl.shaderSource(s, src); gl.compileShader(s)
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.warn('Smoke shader:', gl.getShaderInfoLog(s))
      return s
    }
    const prog = gl.createProgram()!
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT))
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG))
    gl.linkProgram(prog)
    gl.useProgram(prog)
    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW)
    const loc = gl.getAttribLocation(prog, 'a')
    gl.enableVertexAttribArray(loc)
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)
    const uRes = gl.getUniformLocation(prog, 'uRes')
    const uTime = gl.getUniformLocation(prog, 'uTime')
    const uProgress = gl.getUniformLocation(prog, 'uProgress')
    const uMouse = gl.getUniformLocation(prog, 'uMouse')

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    // Render at reduced resolution: smoke is soft, and this keeps the GPU free for the dashboard.
    const scale = Math.min(window.devicePixelRatio || 1, 1.5) * 0.5
    const resize = () => {
      const w = Math.max(1, Math.round(canvas.clientWidth * scale))
      const h = Math.max(1, Math.round(canvas.clientHeight * scale))
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h) }
    }
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)
    resize()

    const mouse = { x: 0, y: 0, tx: 0, ty: 0 }
    const onMove = (e: PointerEvent) => { mouse.tx = e.clientX / window.innerWidth - 0.5; mouse.ty = 0.5 - e.clientY / window.innerHeight }
    window.addEventListener('pointermove', onMove, { passive: true })

    let visible = true
    const io = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; if (visible) schedule() })
    io.observe(canvas)

    let raf = 0
    let smoothed = progress.current
    const start = performance.now()
    const draw = (now: number) => {
      raf = 0
      smoothed += (progress.current - smoothed) * 0.12
      mouse.x += (mouse.tx - mouse.x) * 0.04
      mouse.y += (mouse.ty - mouse.y) * 0.04
      gl.uniform2f(uRes, canvas.width, canvas.height)
      gl.uniform1f(uTime, reduced ? 12 : (now - start) / 1000)
      gl.uniform1f(uProgress, smoothed)
      gl.uniform2f(uMouse, mouse.x, mouse.y)
      gl.clearColor(0, 0, 0, 0)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.drawArrays(gl.TRIANGLES, 0, 3)
      // Reduced motion: redraw only while scroll position is still settling.
      if (!reduced || Math.abs(progress.current - smoothed) > 0.001) schedule()
    }
    function schedule() { if (!raf && visible && !document.hidden) raf = requestAnimationFrame(draw) }
    const onVisibility = () => schedule()
    const onScroll = () => schedule()
    document.addEventListener('visibilitychange', onVisibility)
    window.addEventListener('scroll', onScroll, { passive: true })
    schedule()

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect(); io.disconnect()
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('scroll', onScroll)
      document.removeEventListener('visibilitychange', onVisibility)
      gl.getExtension('WEBGL_lose_context')?.loseContext()
    }
  }, [progress])

  return <canvas ref={canvasRef} className={className} aria-hidden />
}
