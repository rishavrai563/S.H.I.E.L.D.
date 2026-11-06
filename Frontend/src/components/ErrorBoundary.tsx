import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertOctagon, RotateCcw } from 'lucide-react'

interface Props { children: ReactNode }
interface State { hasError: boolean }
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }
  static getDerivedStateFromError(): State { return { hasError: true } }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('Dashboard section failed to render', error, info.componentStack) }
  render() {
    if (this.state.hasError) return <div className="empty-state min-h-40"><AlertOctagon className="text-breach" /><span>This section couldn't display the latest data.</span><button onClick={() => this.setState({ hasError: false })} className="btn-quiet mt-2 inline-flex items-center gap-2"><RotateCcw size={14} /> Reload section</button></div>
    return this.props.children
  }
}
