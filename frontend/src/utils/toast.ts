/**
 * Toast event plumbing (no UI).
 *
 * `toast(message, kind)` can be called from anywhere; the single <ToastHost />
 * in common/Toast.tsx listens and renders them bottom-right with auto-dismiss.
 */
export type ToastKind = 'success' | 'error' | 'info'

interface ToastDetail {
  message: string
  kind: ToastKind
}

const EVENT = 'app:toast'

export function toast(message: string, kind: ToastKind = 'info'): void {
  window.dispatchEvent(new CustomEvent<ToastDetail>(EVENT, { detail: { message, kind } }))
}

export function onToast(handler: (detail: ToastDetail) => void): () => void {
  const listener = (e: Event) => {
    const detail = (e as CustomEvent<ToastDetail>).detail
    if (detail?.message) handler(detail)
  }
  window.addEventListener(EVENT, listener)
  return () => window.removeEventListener(EVENT, listener)
}
