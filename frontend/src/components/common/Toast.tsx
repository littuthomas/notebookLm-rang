/**
 * Toast UI host. The imperative `toast()` helper lives in utils/toast.ts.
 */
import { useEffect, useState } from 'react'
import { AlertTriangle, CheckCircle2, Info, X, type LucideIcon } from 'lucide-react'
import { onToast } from '../../utils/toast'
import type { ToastKind } from '../../utils/toast'

interface ToastEntry {
  id: number
  message: string
  kind: ToastKind
}

let _nextId = 0

const styles: Record<ToastKind, string> = {
  success: 'bg-card border-mint/30 text-ink',
  error: 'bg-card border-coral/40 text-ink',
  info: 'bg-card border-line text-ink',
}

const iconMap: Record<ToastKind, LucideIcon> = {
  success: CheckCircle2,
  error: AlertTriangle,
  info: Info,
}

const iconColor: Record<ToastKind, string> = {
  success: 'text-mint',
  error: 'text-coral-deep',
  info: 'text-lavender',
}

export function ToastHost() {
  const [toasts, setToasts] = useState<ToastEntry[]>([])

  useEffect(() => {
    const off = onToast(({ message, kind }) => {
      const entry: ToastEntry = { id: ++_nextId, message, kind }
      setToasts((prev) => [...prev.slice(-3), entry]) // keep at most 4 visible
      window.setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== entry.id))
      }, kind === 'error' ? 7000 : 4200)
    })
    return off
  }, [])

  if (toasts.length === 0) return null

  return (
    <div
      className="fixed bottom-5 right-5 z-[60] space-y-2 w-80 max-w-[calc(100vw-2.5rem)]"
      role="status"
      aria-live="polite"
    >
      {toasts.map((t) => {
        const Icon = iconMap[t.kind]
        return (
          <div
            key={t.id}
            className={`flex items-start gap-2.5 rounded-xl border px-3.5 py-3 shadow-lg anim-fade-up ${styles[t.kind]}`}
          >
            <Icon size={16} className={`shrink-0 mt-0.5 ${iconColor[t.kind]}`} aria-hidden="true" />
            <p className="text-sm flex-1 min-w-0 break-words">{t.message}</p>
            <button
              type="button"
              onClick={() => setToasts((prev) => prev.filter((x) => x.id !== t.id))}
              className="p-0.5 rounded text-muted hover:text-ink shrink-0"
              aria-label="Dismiss notification"
            >
              <X size={13} aria-hidden="true" />
            </button>
          </div>
        )
      })}
    </div>
  )
}
