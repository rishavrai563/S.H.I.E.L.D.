import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertOctagon, RotateCcw } from 'lucide-react'

interface Props { children: ReactNode }
interface State { hasError: boolean }
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }
  static getDerivedStateFromError(): State { return { hasError: true } }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('Dashboard section failed to render', error, info.componentStack) }
  render() {
    if (this.state.hasError) return <div className="empty-state min-h-40"><AlertOctagon className="text-rose-300" /><span>Section unavailable due to a telemetry rendering error.</span><button onClick={() => this.setState({ hasError: false })} className="mt-2 inline-flex items-center gap-2 rounded bg-slate-800 px-3 py-2 text-xs text-slate-300"><RotateCcw size={13} /> Retry section</button></div>
    return this.props.children
  }
}
