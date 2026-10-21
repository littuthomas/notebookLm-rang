import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { AlertTriangle, RotateCcw, Home } from 'lucide-react'

interface ErrorBoundaryProps {
  children: ReactNode
  /** Optional custom title, e.g. "This notebook could not be displayed" */
  title?: string
  /** Show a "Go to dashboard" link in addition to Try again */
  showHomeLink?: boolean
}

interface ErrorBoundaryState {
  error: Error | null
}

/**
 * Catches render/runtime errors anywhere below it and shows a friendly,
 * on-brand page instead of a blank screen. Users can retry (remounts the
 * subtree) or head back to the dashboard.
 *
 * Usage: wrap route elements or any subtree that may throw, e.g.
 *   <ErrorBoundary><NotebookDetail /></ErrorBoundary>
 *
 * Note: it does NOT catch errors inside event handlers or async callbacks —
 * those should be handled with try/catch (the pages already do).
 */
export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Log for debugging; swap for a monitoring service call later if needed
    console.error('[ErrorBoundary] Uncaught render error:', error, info.componentStack)
  }

  private handleRetry = () => {
    this.setState({ error: null })
  }

  private handleGoHome = () => {
    window.location.assign('/dashboard')
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    return (
      <div className="min-h-full flex items-center justify-center p-6 bg-cream">
        <div
          className="bg-card rounded-2xl border border-line shadow-sm max-w-md w-full p-8 text-center anim-fade-up"
          role="alert"
        >
          <div className="w-14 h-14 mx-auto rounded-2xl bg-coral-soft flex items-center justify-center text-coral mb-4">
            <AlertTriangle size={26} aria-hidden="true" />
          </div>
          <h1 className="text-lg font-semibold text-ink mb-1.5">
            {this.props.title ?? 'Something went wrong'}
          </h1>
          <p className="text-sm text-muted mb-2">
            An unexpected error occurred while displaying this page. Your data is safe — nothing
            was lost.
          </p>
          <p className="text-xs text-muted/80 bg-cream rounded-lg px-3 py-2 mb-5 break-words line-clamp-3 font-mono">
            {error.message || 'Unknown error'}
          </p>
          <div className="flex items-center justify-center gap-2.5">
            <button
              type="button"
              onClick={this.handleRetry}
              className="inline-flex items-center gap-2 h-10 px-4 rounded-xl bg-lavender text-white text-sm font-medium hover:bg-lavender-deep transition-colors active:scale-[0.98]"
            >
              <RotateCcw size={15} aria-hidden="true" /> Try again
            </button>
            {this.props.showHomeLink !== false && (
              <button
                type="button"
                onClick={this.handleGoHome}
                className="inline-flex items-center gap-2 h-10 px-4 rounded-xl border border-line text-sm font-medium text-ink hover:bg-cream transition-colors active:scale-[0.98]"
              >
                <Home size={15} aria-hidden="true" /> Go to Dashboard
              </button>
            )}
          </div>
        </div>
      </div>
    )
  }
}
