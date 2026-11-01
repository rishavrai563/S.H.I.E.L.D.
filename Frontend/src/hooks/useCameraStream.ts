import { useEffect, useRef, useState } from 'react'

/** Receives binary JPEG frames from the backend's per-camera WebSocket. */
export function useCameraStream(cameraId: string) {
  const [hasFrame, setHasFrame] = useState(false)
  const hasFrameRef = useRef(false)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const frameUrlRef = useRef<string | null>(null)

  useEffect(() => {
    let disposed = false
    let retry = 0
    let reconnectTimer: number | undefined
    let socket: WebSocket | null = null

    const connect = () => {
      if (disposed) return
      const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const currentSocket = new WebSocket(`${scheme}//${window.location.host}/ws/stream/${encodeURIComponent(cameraId)}`)
      socket = currentSocket
      currentSocket.binaryType = 'blob'
      currentSocket.onopen = () => { retry = 0 }
      currentSocket.onmessage = event => {
        if (disposed || !(event.data instanceof Blob)) return
        const nextUrl = URL.createObjectURL(event.data)
        const image = imageRef.current
        if (!image) {
          URL.revokeObjectURL(nextUrl)
          return
        }
        image.src = nextUrl
        if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current)
        frameUrlRef.current = nextUrl
        if (!hasFrameRef.current) {
          hasFrameRef.current = true
          setHasFrame(true)
        }
      }
      currentSocket.onerror = () => currentSocket.close()
      currentSocket.onclose = () => {
        if (disposed) return
        const delay = Math.min(1000 * (2 ** retry), 10_000)
        retry += 1
        reconnectTimer = window.setTimeout(connect, delay)
      }
    }

    setHasFrame(false)
    hasFrameRef.current = false
    frameUrlRef.current = null
    connect()

    return () => {
      disposed = true
      window.clearTimeout(reconnectTimer)
      socket?.close()
      if (imageRef.current) imageRef.current.removeAttribute('src')
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current)
      frameUrlRef.current = null
    }
  }, [cameraId])

  return { imageRef, hasFrame }
}
