import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { RiskLevel } from '../types/telemetry'

export interface SectorCamera {
  id: string
  name: string
  online: boolean
  risk: RiskLevel
  people: number
  vehicles: number
  night: boolean
  lng?: number
}

const COLORS = {
  bone: new THREE.Color('#ece6cc'),
  tide: new THREE.Color('#22d3cb'),
  caution: new THREE.Color('#f2b544'),
  breach: new THREE.Color('#ff5b4f'),
  fog: new THREE.Color('#5f6a69'),
  night: new THREE.Color('#a59bff'),
  fence: new THREE.Color('#e879f9'),
  abyss: new THREE.Color('#030506'),
}
const stateColor = (c: SectorCamera) => !c.online ? COLORS.fog : c.risk === 'CRITICAL' ? COLORS.breach : c.risk === 'WARNING' ? COLORS.caution : COLORS.tide
const stateText = (c: SectorCamera) => !c.online ? 'No video' : c.risk === 'CRITICAL' ? 'Respond' : c.risk === 'WARNING' ? 'Watch' : 'Clear'

// ── Terrain height field (deterministic value noise, so the sector looks the same every load) ──
function hash(x: number, y: number) { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s) }
function vnoise(x: number, y: number) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf)
  const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1)
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}
function heightAt(x: number, z: number) {
  let h = 0, amp = 1.6, f = 0.09
  for (let i = 0; i < 5; i++) { h += vnoise(x * f + 11, z * f + 7) * amp; amp *= 0.5; f *= 2.05 }
  // A ridge line behind the border, a valley along it.
  h += Math.max(0, -z - 6) * 0.22 + Math.exp(-((z - borderZ(x)) ** 2) / 10) * -0.6
  return h - 1.4
}
const borderZ = (x: number) => Math.sin(x * 0.17) * 2.4 + Math.sin(x * 0.41 + 1.3) * 0.8 - 1.5

const TERRAIN_VERT = `
varying float vH;
varying vec3 vPos;
void main() {
  vH = position.y;
  vPos = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`
const TERRAIN_FRAG = `
uniform vec3 uColor;
uniform float uTime;
varying float vH;
varying vec3 vPos;
float contour(float v) { float g = abs(fract(v - 0.5) - 0.5) / fwidth(v); return 1.0 - min(g, 1.0); }
void main() {
  float minor = contour(vH * 3.0);
  float major = contour(vH * 0.6);
  float grid = max(contour(vPos.x * 0.5), contour(vPos.z * 0.5)) * 0.35;
  float r = length(vPos.xz / vec2(26.0, 17.0));
  float falloff = 1.0 - smoothstep(0.55, 1.0, r);
  float a = (0.05 + minor * 0.28 + major * 0.55 + grid * 0.25) * falloff;
  gl_FragColor = vec4(uColor * (0.55 + 0.45 * major), a);
}`

interface Props { cameras: SectorCamera[] }

export default function SectorMap({ cameras }: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const labelsRef = useRef<HTMLDivElement | null>(null)
  const dataRef = useRef(cameras)
  const [failed, setFailed] = useState(false)
  dataRef.current = cameras

  useEffect(() => {
    const host = hostRef.current, labelHost = labelsRef.current
    if (!host || !labelHost) return
    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' })
    } catch { setFailed(true); return }
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75))
    renderer.setClearColor(0x000000, 0)
    host.appendChild(renderer.domElement)
    renderer.domElement.style.display = 'block'

    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(COLORS.abyss.getHex(), 0.03)
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 200)
    camera.position.set(3, 17, 19)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.enablePan = false
    controls.enableZoom = false        // the wheel keeps scrolling the page
    controls.minPolarAngle = 0.45
    controls.maxPolarAngle = 1.2
    controls.autoRotate = !reduced
    controls.autoRotateSpeed = 0.35
    controls.target.set(0, -1.2, 0.5)
    let resumeTimer = 0
    controls.addEventListener('start', () => { controls.autoRotate = false; window.clearTimeout(resumeTimer) })
    controls.addEventListener('end', () => { if (!reduced) resumeTimer = window.setTimeout(() => { controls.autoRotate = true }, 6000) })

    // ── Terrain ──
    const terrainGeo = new THREE.PlaneGeometry(52, 34, 208, 136)
    terrainGeo.rotateX(-Math.PI / 2)
    const pos = terrainGeo.attributes.position as THREE.BufferAttribute
    for (let i = 0; i < pos.count; i++) pos.setY(i, heightAt(pos.getX(i), pos.getZ(i)))
    terrainGeo.computeVertexNormals()
    const terrainMat = new THREE.ShaderMaterial({
      vertexShader: TERRAIN_VERT, fragmentShader: TERRAIN_FRAG, transparent: true, depthWrite: true,
      uniforms: { uColor: { value: COLORS.bone }, uTime: { value: 0 } },
    })
    scene.add(new THREE.Mesh(terrainGeo, terrainMat))

    // ── Border line and the virtual-fence curtain above it ──
    const borderPts: THREE.Vector3[] = []
    for (let x = -24; x <= 24; x += 0.4) borderPts.push(new THREE.Vector3(x, heightAt(x, borderZ(x)) + 0.04, borderZ(x)))
    const borderLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(borderPts),
      new THREE.LineDashedMaterial({ color: COLORS.bone, dashSize: 0.5, gapSize: 0.3, transparent: true, opacity: 0.85 }))
    borderLine.computeLineDistances()
    scene.add(borderLine)

    const curtainPos: number[] = [], curtainAlpha: number[] = []
    borderPts.forEach(p => { curtainPos.push(p.x, p.y, p.z, p.x, p.y + 1.6, p.z); curtainAlpha.push(1, 0) })
    const curtainIdx: number[] = []
    for (let i = 0; i < borderPts.length - 1; i++) { const a = i * 2; curtainIdx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2) }
    const curtainGeo = new THREE.BufferGeometry()
    curtainGeo.setAttribute('position', new THREE.Float32BufferAttribute(curtainPos, 3))
    curtainGeo.setAttribute('alpha', new THREE.Float32BufferAttribute(curtainAlpha, 1))
    curtainGeo.setIndex(curtainIdx)
    const curtainMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
      uniforms: { uColor: { value: COLORS.fence }, uTime: { value: 0 } },
      vertexShader: `attribute float alpha; varying float vA; varying float vX; void main(){ vA = alpha; vX = position.x; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
      fragmentShader: `uniform vec3 uColor; uniform float uTime; varying float vA; varying float vX;
        void main(){ float scan = 0.55 + 0.45 * sin(vX * 0.9 - uTime * 1.6); gl_FragColor = vec4(uColor, vA * vA * 0.38 * scan); }`,
    })
    scene.add(new THREE.Mesh(curtainGeo, curtainMat))

    // ── Camera posts ──
    interface Post {
      id: string; group: THREE.Group; head: THREE.Mesh; cone: THREE.Mesh; rim: THREE.LineLoop
      rings: THREE.Mesh[]; dots: THREE.Points; dotSeed: Float32Array; dotKey: string; anchor: THREE.Vector3
      label: HTMLDivElement
    }
    const posts = new Map<string, Post>()
    const ringGeo = new THREE.RingGeometry(0.92, 1, 64)
    ringGeo.rotateX(-Math.PI / 2)
    const coneGeo = new THREE.ConeGeometry(1.7, 4.6, 40, 1, true)
    coneGeo.translate(0, -2.3, 0)          // apex at origin, opening along -y
    coneGeo.rotateX(Math.PI / 2 - 0.36)    // swing -y to face the border (-z), tilted down
    const rimGeo = new THREE.BufferGeometry().setFromPoints(
      Array.from({ length: 48 }, (_, i) => { const a = i / 48 * Math.PI * 2; return new THREE.Vector3(Math.cos(a) * 1.7, -4.6, Math.sin(a) * 1.7) }))
    rimGeo.rotateX(Math.PI / 2 - 0.36)

    const layout = (list: SectorCamera[]) => {
      const ordered = [...list].sort((a, b) => (a.lng ?? 0) - (b.lng ?? 0) || a.id.localeCompare(b.id))
      const span = Math.min(30, 9 * Math.max(1, ordered.length - 1))
      return new Map(ordered.map((c, i) => {
        const x = ordered.length === 1 ? 0 : -span / 2 + (span * i) / (ordered.length - 1)
        const z = borderZ(x) + 4.2
        return [c.id, new THREE.Vector3(x, heightAt(x, z), z)]
      }))
    }

    const makePost = (c: SectorCamera, at: THREE.Vector3): Post => {
      const group = new THREE.Group()
      group.position.copy(at)
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.05, 1.8, 6), new THREE.MeshBasicMaterial({ color: COLORS.bone, transparent: true, opacity: 0.7 }))
      mast.position.y = 0.9
      group.add(mast)
      const head = new THREE.Mesh(new THREE.OctahedronGeometry(0.24), new THREE.MeshBasicMaterial({ color: stateColor(c) }))
      head.position.y = 1.9
      group.add(head)
      const cone = new THREE.Mesh(coneGeo, new THREE.MeshBasicMaterial({ color: stateColor(c), transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false }))
      cone.position.y = 1.9
      group.add(cone)
      const rim = new THREE.LineLoop(rimGeo, new THREE.LineBasicMaterial({ color: stateColor(c), transparent: true, opacity: 0.55 }))
      rim.position.y = 1.9
      group.add(rim)
      const rings = [0, 1, 2].map(() => {
        const r = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: stateColor(c), transparent: true, opacity: 0, depthWrite: false, depthTest: false, side: THREE.DoubleSide }))
        r.position.y = 0.06
        r.renderOrder = 2  // pulse stays readable where terrain rises through it
        group.add(r)
        return r
      })
      const dots = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 0.26, vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false, sizeAttenuation: true }))
      group.add(dots)
      scene.add(group)
      const label = document.createElement('div')
      label.className = 'sector-label'
      labelHost.appendChild(label)
      return { id: c.id, group, head, cone, rim, rings, dots, dotSeed: new Float32Array(0), dotKey: '', anchor: at.clone(), label }
    }

    const syncPosts = () => {
      const list = dataRef.current
      const places = layout(list)
      for (const [id, post] of posts) if (!places.has(id)) {
        scene.remove(post.group); post.label.remove(); posts.delete(id)
      }
      for (const c of list) {
        const at = places.get(c.id)!
        let post = posts.get(c.id)
        if (!post) { post = makePost(c, at); posts.set(c.id, post) }
        post.group.position.copy(at); post.anchor.copy(at)
        const col = stateColor(c)
        ;(post.head.material as THREE.MeshBasicMaterial).color.copy(col)
        ;(post.cone.material as THREE.MeshBasicMaterial).color.copy(c.night && c.online ? COLORS.night : col)
        ;(post.rim.material as THREE.LineBasicMaterial).color.copy(c.night && c.online ? COLORS.night : col)
        post.rings.forEach(r => (r.material as THREE.MeshBasicMaterial).color.copy(col))
        // Activity dots: one per tracked person (bone) or vehicle (tide), scattered in the field of view.
        const people = Math.min(c.people, 40), vehicles = Math.min(c.vehicles, 40)
        const key = `${people}:${vehicles}`
        if (key !== post.dotKey) {
          const n = people + vehicles
          const seed = new Float32Array(n * 3)
          const colors = new Float32Array(n * 3)
          for (let i = 0; i < n; i++) {
            seed[i * 3] = Math.random() * Math.PI * 2
            seed[i * 3 + 1] = 1.2 + Math.random() * 3.2
            seed[i * 3 + 2] = 0.2 + Math.random() * 0.6
            const cc = i < people ? COLORS.bone : COLORS.tide
            colors.set([cc.r, cc.g, cc.b], i * 3)
          }
          post.dots.geometry.dispose()
          post.dots.geometry = new THREE.BufferGeometry()
          post.dots.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3))
          post.dots.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
          post.dotSeed = seed
          post.dotKey = key
        }
        post.label.innerHTML = `<b>${c.name}</b><span class="sector-state" style="color:#${col.getHexString()}">${stateText(c)}${c.night && c.online ? ', night' : ''}</span>`
      }
    }
    syncPosts()

    // ── Sizing, visibility, loop ──
    const resize = () => {
      const w = host.clientWidth, h = host.clientHeight
      renderer.setSize(w, h, false)
      renderer.domElement.style.width = `${w}px`; renderer.domElement.style.height = `${h}px`
      camera.aspect = w / Math.max(1, h)
      camera.position.sub(controls.target).setLength(w < 640 ? 33 : 25).add(controls.target)
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(host)
    resize()

    let visible = true, raf = 0, lastSync = 0
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) schedule() }, { rootMargin: '100px' })
    io.observe(host)
    const t0 = performance.now()
    const tmp = new THREE.Vector3()

    const frame = () => {
      raf = 0
      const t = (performance.now() - t0) / 1000
      if (t - lastSync > 0.5) { syncPosts(); lastSync = t }
      controls.update()
      terrainMat.uniforms.uTime.value = t
      curtainMat.uniforms.uTime.value = reduced ? 0 : t
      const w = host.clientWidth, h = host.clientHeight
      const list = dataRef.current
      for (const c of list) {
        const post = posts.get(c.id)
        if (!post) continue
        post.head.rotation.y = reduced ? 0.6 : t * 0.8
        const speed = !c.online ? 0 : c.risk === 'CRITICAL' ? 1.4 : c.risk === 'WARNING' ? 0.75 : 0.35
        post.rings.forEach((r, k) => {
          const phase = reduced || !speed ? (k + 1) / 4 : (t * speed + k / 3) % 1
          r.scale.setScalar(0.6 + phase * (c.risk === 'CRITICAL' ? 4.2 : 3))
          ;(r.material as THREE.MeshBasicMaterial).opacity = speed ? (1 - phase) * (c.risk === 'CRITICAL' ? 0.75 : 0.4) : 0
        })
        const arr = (post.dots.geometry.getAttribute('position') as THREE.BufferAttribute | undefined)
        if (arr) {
          const n = arr.count
          for (let i = 0; i < n; i++) {
            const a0 = post.dotSeed[i * 3], rad = post.dotSeed[i * 3 + 1], spd = post.dotSeed[i * 3 + 2]
            const a = a0 + (reduced ? 0 : t * spd * 0.25)
            // Wander inside the camera's field of view (toward the border, i.e. -z).
            const lx = Math.cos(a) * rad * 0.6, lz = -2.2 - Math.abs(Math.sin(a * 1.3)) * rad * 0.7
            const wx = post.anchor.x + lx, wz = post.anchor.z + lz
            arr.setXYZ(i, lx, heightAt(wx, wz) - post.anchor.y + 0.14, lz)
          }
          arr.needsUpdate = true
        }
        tmp.copy(post.anchor); tmp.y += 2.5
        tmp.project(camera)
        const sx = (tmp.x * 0.5 + 0.5) * w, sy = (-tmp.y * 0.5 + 0.5) * h
        post.label.style.transform = `translate(-50%, -100%) translate(${sx}px, ${sy}px)`
        post.label.style.opacity = tmp.z < 1 ? '1' : '0'
      }
      renderer.render(scene, camera)
      if (!reduced || controls.autoRotate) schedule()
    }
    function schedule() { if (!raf && visible && !document.hidden) raf = requestAnimationFrame(frame) }
    const onVis = () => schedule()
    document.addEventListener('visibilitychange', onVis)
    controls.addEventListener('change', schedule)
    schedule()
    // Reduced motion still redraws when telemetry changes.
    const dataTimer = window.setInterval(schedule, 1000)

    return () => {
      cancelAnimationFrame(raf); window.clearInterval(dataTimer); window.clearTimeout(resumeTimer)
      ro.disconnect(); io.disconnect()
      document.removeEventListener('visibilitychange', onVis)
      controls.dispose()
      scene.traverse(o => {
        const m = o as THREE.Mesh
        m.geometry?.dispose?.()
        const mat = m.material as THREE.Material | THREE.Material[] | undefined
        if (Array.isArray(mat)) mat.forEach(x => x.dispose()); else mat?.dispose?.()
      })
      renderer.dispose()
      renderer.domElement.remove()
      labelHost.innerHTML = ''
    }
  }, [])

  if (failed) return <div className="empty-state min-h-[360px]">3D view needs WebGL, which this browser has turned off.</div>
  return <div className="relative h-full w-full">
    <div ref={hostRef} className="absolute inset-0 cursor-grab active:cursor-grabbing" />
    <div ref={labelsRef} className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden />
  </div>
}
