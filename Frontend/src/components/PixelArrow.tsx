/** Stepped, pixel-built arrow drawn on a 9×7 grid, so it reads as part of the survey canvas. */
const CELLS: [number, number][] = [
  ...Array.from({ length: 8 }, (_, x) => [x, 3] as [number, number]),   // shaft
  [5, 0], [6, 1], [7, 2], [8, 3], [7, 4], [6, 5], [5, 6],                // head
  [4, 0], [5, 1], [6, 2], [6, 4], [5, 5], [4, 6],                        // head, second pixel of weight
]

export default function PixelArrow({ className = '' }: { className?: string }) {
  return <svg className={className} viewBox="0 0 9 7" aria-hidden shapeRendering="crispEdges" fill="currentColor">
    {CELLS.map(([x, y]) => <rect key={`${x}-${y}`} x={x} y={y} width="1" height="1" />)}
  </svg>
}
