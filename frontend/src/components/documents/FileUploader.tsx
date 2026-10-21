/**
 * Reusable document upload surface:
 * drag-and-drop zone, browse-files picker, staged file list with removal,
 * per-file upload progress (real browser progress), status and retry.
 *
 * The component only stages files; the caller decides when to push them via
 * `handle.uploadAll(notebookId)` (imperative ref). This lets the Create
 * Notebook dialog create the notebook FIRST and then upload the files into it,
 * while the notebook detail page uploads straight into an existing notebook.
 */
import { forwardRef, useCallback, useImperativeHandle, useRef, useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  CloudUpload,
  FileText,
  Loader2,
  RotateCcw,
  X,
} from 'lucide-react'
import { uploadDocumentWithProgress } from '../../services/api'
import type { UploadResult } from '../../services/api'
import {
  FILE_ACCEPT_ATTR,
  SUPPORTED_EXTENSIONS_LABEL,
  formatBytes,
  typeLabel,
  validateFile,
} from '../../utils/files'

export type FileStatus = 'staged' | 'uploading' | 'ready' | 'failed'

export interface UploadItem {
  id: string
  file: File
  status: FileStatus
  /** 0-100 real upload transfer progress. */
  progress: number
  error?: string
  result?: UploadResult
}

export interface UploadSummary {
  succeeded: number
  failed: { filename: string; error: string }[]
}

export interface FileUploaderHandle {
  /** Upload every staged/failed file into the given notebook. */
  uploadAll: (notebookId: number | string) => Promise<UploadSummary>
  /** Number of files that would be uploaded (staged + failed retryables). */
  pendingCount: () => number
}

interface FileUploaderProps {
  /** Short helper text shown under the dropzone. */
  hint?: string
  compact?: boolean
  /** Fired whenever the staged set changes (no uploads implied). */
  onItemsChange?: (items: UploadItem[]) => void
}

let _idCounter = 0
/** Notebook id of the last uploadAll() call — used by inline retry buttons. */
let _lastNotebookId: number | string | null = null

export const FileUploader = forwardRef<FileUploaderHandle, FileUploaderProps>(
  function FileUploader({ hint, compact = false, onItemsChange }, ref) {
    const [items, setItems] = useState<UploadItem[]>([])
    const [dragging, setDragging] = useState(false)
    const inputRef = useRef<HTMLInputElement>(null)
    const itemsRef = useRef<UploadItem[]>([])
    const uploadingRef = useRef(false)

    const commit = useCallback(
      (next: UploadItem[]) => {
        itemsRef.current = next
        setItems(next)
        onItemsChange?.(next)
      },
      [onItemsChange],
    )

    const patch = useCallback(
      (id: string, changes: Partial<UploadItem>) => {
        commit(itemsRef.current.map((it) => (it.id === id ? { ...it, ...changes } : it)))
      },
      [commit],
    )

    /** Validate + stage a batch of files from the picker or a drop event. */
    const addFiles = useCallback(
      (fileList: FileList | File[] | null) => {
        if (!fileList) return
        const files = Array.from(fileList)
        if (files.length === 0) return

        const existing = new Set(itemsRef.current.map((it) => `${it.file.name}:${it.file.size}`))
        const additions: UploadItem[] = []
        for (const file of files) {
          const key = `${file.name}:${file.size}`
          if (existing.has(key)) continue // ignore duplicates
          existing.add(key)
          const error = validateFile(file)
          additions.push({
            id: `f${++_idCounter}`,
            file,
            status: error ? 'failed' : 'staged',
            progress: 0,
            error: error ?? undefined,
          })
        }
        if (additions.length > 0) commit([...itemsRef.current, ...additions])
        // Reset the input so selecting the same file again re-fires onChange.
        if (inputRef.current) inputRef.current.value = ''
      },
      [commit],
    )

    const removeItem = useCallback(
      (id: string) => {
        if (uploadingRef.current) return // don't yank files out mid-upload
        commit(itemsRef.current.filter((it) => it.id !== id))
      },
      [commit],
    )

    const uploadOne = useCallback(
      async (item: UploadItem, notebookId: number | string): Promise<boolean> => {
        patch(item.id, { status: 'uploading', progress: 0, error: undefined })
        try {
          const result = await uploadDocumentWithProgress(item.file, notebookId, (p) =>
            patch(item.id, { progress: p }),
          )
          // A 2xx response from the backend is the ONLY success signal — files
          // are never marked ready before the server confirms processing.
          patch(item.id, { status: 'ready', progress: 100, result, error: undefined })
          return true
        } catch (e) {
          patch(item.id, {
            status: 'failed',
            error: e instanceof Error ? e.message : 'Upload failed.',
          })
          return false
        }
      },
      [patch],
    )

    const uploadAll = useCallback(
      async (notebookId: number | string): Promise<UploadSummary> => {
        _lastNotebookId = notebookId
        // Staged + failed = everything the user still wants uploaded.
        const queue = itemsRef.current.filter(
          (it) => it.status === 'staged' || it.status === 'failed',
        )
        if (queue.length === 0) return { succeeded: 0, failed: [] }
        uploadingRef.current = true
        let succeeded = 0
        const failed: { filename: string; error: string }[] = []
        for (const item of queue) {
          const ok = await uploadOne(item, notebookId)
          if (ok) succeeded += 1
          else
            failed.push({
              filename: item.file.name,
              error: item.error ?? 'Upload failed.',
            })
        }
        uploadingRef.current = false
        return { succeeded, failed }
      },
      [uploadOne],
    )

    useImperativeHandle(
      ref,
      () => ({
        uploadAll,
        pendingCount: () =>
          itemsRef.current.filter((it) => it.status === 'staged' || it.status === 'failed').length,
      }),
      [uploadAll],
    )

    const busy = items.some((it) => it.status === 'uploading')
    const stagedCount = items.filter((it) => it.status === 'staged').length
    const failedCount = items.filter((it) => it.status === 'failed').length
    const doneCount = items.filter((it) => it.status === 'ready').length

    return (
      <div className="space-y-3">
        {/* Dropzone */}
        <div
          className={`relative rounded-2xl border-2 border-dashed text-center transition-all duration-200 ${
            compact ? 'p-5' : 'p-7'
          } ${dragging ? 'border-lavender bg-lavender-soft/40 scale-[1.01]' : 'border-line hover:border-lavender/50'}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            addFiles(e.dataTransfer.files)
          }}
          aria-label="Upload area"
        >
          <span className="inline-flex h-12 w-12 rounded-2xl bg-lavender-soft items-center justify-center text-lavender mb-2.5">
            <CloudUpload size={22} aria-hidden="true" />
          </span>
          <p className="font-medium text-ink text-sm">
            Drag &amp; drop files here, or{' '}
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="text-lavender-deep font-semibold hover:underline"
            >
              Browse Files
            </button>
          </p>
          <p className="text-xs text-muted mt-1.5">
            {hint ?? `Supports ${SUPPORTED_EXTENSIONS_LABEL} — max 20 MB per file`}
          </p>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={FILE_ACCEPT_ATTR}
            className="hidden"
            onChange={(e) => addFiles(e.target.files)}
            aria-label="Choose files to upload"
          />
        </div>

        {/* Staged file list */}
        {items.length > 0 && (
          <ul className="space-y-2" aria-label="Selected files">
            {items.map((it) => {
              // Files rejected client-side (unsupported/empty/oversized) can't
              // be retried — only selectable-reason failures make sense here.
              const retryable =
                it.status === 'failed' &&
                it.error !== undefined &&
                !it.error.includes('not a supported type') &&
                !it.error.includes('is empty') &&
                !it.error.includes('larger than') &&
                _lastNotebookId !== null
              return (
                <li
                  key={it.id}
                  className="bg-card rounded-xl border border-line px-3 py-2.5"
                  data-status={it.status}
                >
                  <div className="flex items-center gap-2.5">
                    <FileText size={15} className="text-lavender shrink-0" aria-hidden="true" />
                    <span className="text-sm text-ink truncate flex-1" title={it.file.name}>
                      {it.file.name}
                    </span>
                    <span className="text-[11px] text-muted shrink-0">
                      {formatBytes(it.file.size)}
                    </span>

                    {/* Status */}
                    {it.status === 'ready' && (
                      <span className="inline-flex items-center gap-1 text-[11px] font-medium text-mint shrink-0">
                        <CheckCircle2 size={13} aria-hidden="true" /> Ready
                      </span>
                    )}
                    {it.status === 'uploading' && (
                      <span
                        className="inline-flex items-center gap-1 text-[11px] font-medium text-lavender-deep shrink-0"
                        role="status"
                      >
                        <Loader2 size={13} className="animate-spin" aria-hidden="true" />
                        {it.progress < 100 ? `${it.progress}%` : 'Processing…'}
                      </span>
                    )}
                    {it.status === 'failed' && (
                      <span className="inline-flex items-center gap-1 text-[11px] font-medium text-coral-deep shrink-0">
                        <AlertCircle size={13} aria-hidden="true" /> Failed
                      </span>
                    )}

                    {/* Remove */}
                    <button
                      type="button"
                      onClick={() => removeItem(it.id)}
                      disabled={busy || it.status === 'ready'}
                      className="p-1 rounded-lg text-muted hover:text-coral-deep hover:bg-coral-soft disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
                      aria-label={`Remove ${it.file.name}`}
                    >
                      <X size={14} aria-hidden="true" />
                    </button>
                  </div>

                  {/* Transfer progress */}
                  {it.status === 'uploading' && (
                    <div
                      className="mt-2 h-1.5 rounded-full bg-cream overflow-hidden"
                      role="progressbar"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={it.progress}
                      aria-label={`Uploading ${it.file.name}`}
                    >
                      <div
                        className="h-full bg-lavender rounded-full transition-all duration-200"
                        style={{ width: `${Math.max(4, it.progress)}%` }}
                      />
                    </div>
                  )}

                  {/* Error message + inline retry */}
                  {it.status === 'failed' && it.error && (
                    <div className="mt-1.5 flex items-start gap-2 text-xs text-coral-deep">
                      <span className="min-w-0 break-words">{it.error}</span>
                      {retryable && (
                        <button
                          type="button"
                          onClick={() => void uploadOne(it, _lastNotebookId ?? '')}
                          className="ml-auto inline-flex items-center gap-1 font-medium hover:underline shrink-0"
                        >
                          <RotateCcw size={11} aria-hidden="true" /> Retry
                        </button>
                      )}
                    </div>
                  )}

                  {/* Success detail from the backend confirmation */}
                  {it.status === 'ready' && it.result && (
                    <p className="mt-1.5 text-[11px] text-muted">
                      {typeLabel(it.result.file_type)} · {it.result.page_count} page
                      {it.result.page_count === 1 ? '' : 's'} · {it.result.chunks_stored} chunk
                      {it.result.chunks_stored === 1 ? '' : 's'} indexed
                    </p>
                  )}
                </li>
              )
            })}
          </ul>
        )}

        {/* Batch summary line */}
        {items.length > 0 && (stagedCount > 0 || failedCount > 0 || doneCount > 0) && (
          <p className="text-xs text-muted">
            {[
              stagedCount > 0 &&
                `${stagedCount} file${stagedCount === 1 ? '' : 's'} ready to upload`,
              failedCount > 0 && `${failedCount} failed`,
              doneCount > 0 && `${doneCount} uploaded`,
            ]
              .filter(Boolean)
              .join(' · ')}
          </p>
        )}
      </div>
    )
  },
)
