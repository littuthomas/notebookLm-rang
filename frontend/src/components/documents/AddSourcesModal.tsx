/**
 * "Add Sources" dialog for an EXISTING notebook (opened from the notebook
 * detail page). Uploads straight into the given notebook — the user never
 * needs to create another notebook to add sources.
 */
import { useRef, useState } from 'react'
import { CloudUpload, RotateCcw } from 'lucide-react'
import Modal from '../common/Modal'
import Button from '../common/Button'
import { FileUploader } from './FileUploader'
import type { FileUploaderHandle, UploadItem, UploadSummary } from './FileUploader'
import { toast } from '../../utils/toast'

interface AddSourcesModalProps {
  notebookId: number
  open: boolean
  onClose: () => void
  /** Fired after at least one document finished uploading successfully. */
  onUploaded: () => void
}

export default function AddSourcesModal({ notebookId, open, onClose, onUploaded }: AddSourcesModalProps) {
  const uploaderRef = useRef<FileUploaderHandle | null>(null)
  const [items, setItems] = useState<UploadItem[]>([])
  const [running, setRunning] = useState(false)
  const [failed, setFailed] = useState<UploadSummary['failed']>([])

  const busy = running || items.some((it) => it.status === 'uploading')
  const pending = items.filter((it) => it.status === 'staged' || it.status === 'failed').length
  const ready = items.filter((it) => it.status === 'ready').length

  const reset = () => {
    setItems([])
    setFailed([])
    setRunning(false)
  }

  const handleClose = () => {
    if (busy) return
    onClose()
    reset()
  }

  async function startUpload() {
    if (pending === 0 || running) return
    setRunning(true)
    setFailed([])
    try {
      const summary = (await uploaderRef.current?.uploadAll(notebookId)) ?? {
        succeeded: 0,
        failed: [],
      }
      if (summary.succeeded > 0) {
        toast(
          `${summary.succeeded} document${summary.succeeded === 1 ? '' : 's'} added to the notebook.`,
          'success',
        )
        onUploaded() // refresh the parent's source list — no full app reload
      }
      if (summary.failed.length > 0) {
        setFailed(summary.failed)
        // Stay open: failed items remain in the uploader with retry buttons.
      } else {
        onClose()
        reset()
      }
    } finally {
      setRunning(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title={busy ? 'Uploading…' : 'Add Sources'}
      width="max-w-lg"
    >
      <div className="space-y-4">
        <FileUploader ref={uploaderRef} compact onItemsChange={setItems} />

        {failed.length > 0 && (
          <div className="bg-coral-soft/40 border border-coral/25 rounded-xl px-3.5 py-3 text-sm" role="alert">
            <p className="font-medium text-coral-deep mb-1.5">
              {failed.length} upload{failed.length === 1 ? '' : 's'} failed:
            </p>
            <ul className="text-xs text-coral-deep/90 space-y-1 max-h-28 overflow-y-auto">
              {failed.map((f) => (
                <li key={f.filename} className="break-words">
                  <strong>{f.filename}</strong> — {f.error}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="flex items-center justify-end gap-2">
          <Button type="button" variant="ghost" onClick={handleClose} disabled={busy}>
            {ready > 0 ? 'Done' : 'Cancel'}
          </Button>
          <Button
            type="button"
            onClick={() => void startUpload()}
            loading={running}
            disabled={pending === 0 || busy}
          >
            {failed.length > 0 ? (
              <>
                <RotateCcw size={14} aria-hidden="true" /> Retry failed
              </>
            ) : (
              <>
                <CloudUpload size={14} aria-hidden="true" /> Upload {pending > 0 ? `${pending} file${pending === 1 ? '' : 's'}` : ''}
              </>
            )}
          </Button>
        </div>
      </div>
    </Modal>
  )
}
