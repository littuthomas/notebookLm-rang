/**
 * Create Notebook dialog with the complete document-upload experience:
 *
 *   1. Title / description / colour
 *   2. Drag-and-drop + browse file staging (removable before submission)
 *   3. Create Notebook -> POST /api/notebooks (exactly once, no duplicates)
 *   4. Sequential upload of staged files into the new notebook with
 *      per-file progress, statuses and clear errors (retryable)
 *
 * Notebook creation and document upload failures are handled separately:
 * - If creation fails, the dialog stays open with all staged files intact.
 * - If creation succeeds but some uploads fail, the dialog shows a summary
 *   with a Retry button and an "Open notebook anyway" escape hatch — the
 *   user never loses their selected files.
 */
import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import Modal from '../common/Modal'
import Button from '../common/Button'
import { FileUploader } from './FileUploader'
import type { FileUploaderHandle, UploadSummary } from './FileUploader'
import { createNotebook } from '../../services/api'
import type { Notebook } from '../../services/api'
import { toast } from '../../utils/toast'

const colors = ['lavender', 'coral', 'sun', 'mint'] as const
const colorBar: Record<string, string> = {
  lavender: 'bg-lavender',
  coral: 'bg-coral',
  sun: 'bg-sun',
  mint: 'bg-mint',
}

type Phase = 'editing' | 'creating' | 'uploading' | 'partial'

interface CreateNotebookModalProps {
  open: boolean
  onClose: () => void
  /** Called with the fresh notebook once creation, and all uploads, succeeded. */
  onCreated?: (nb: Notebook) => void
}

export default function CreateNotebookModal({ open, onClose, onCreated }: CreateNotebookModalProps) {
  const navigate = useNavigate()

  const [name, setName] = useState('')
  const [desc, setDesc] = useState('')
  const [color, setColor] = useState<string>('lavender')
  const [titleError, setTitleError] = useState<string | null>(null)
  const [generalError, setGeneralError] = useState<string | null>(null)
  const [phase, setPhase] = useState<Phase>('editing')
  const [stagedCount, setStagedCount] = useState(0)
  const [failedSummary, setFailedSummary] = useState<UploadSummary['failed']>([])
  const [createdNb, setCreatedNb] = useState<Notebook | null>(null)

  const uploaderRef = useRef<FileUploaderHandle | null>(null)
  const busy = phase === 'creating' || phase === 'uploading'
  const inputDisabled = busy || phase === 'partial'

  const reset = () => {
    setName('')
    setDesc('')
    setColor('lavender')
    setTitleError(null)
    setGeneralError(null)
    setStagedCount(0)
    setFailedSummary([])
    setCreatedNb(null)
    setPhase('editing')
  }

  const handleClose = () => {
    if (busy) return // never abandon a running upload batch
    onClose()
    // Parent unmounts via `open`, but clear local state for next open too.
    reset()
  }

  const openCreated = () => {
    if (createdNb === null) return
    onClose()
    reset()
    navigate(`/notebooks/${createdNb.id}`)
  }

  async function handleSubmit() {
    const trimmed = name.trim()
    if (!trimmed) {
      setTitleError('Please give your notebook a title.')
      return
    }
    setTitleError(null)
    setGeneralError(null)
    setPhase('creating')

    // 1) Create the notebook — exactly one POST per submission attempt.
    let nb: Notebook
    try {
      nb = await createNotebook(trimmed, desc.trim(), color)
    } catch (e) {
      // Nothing was created — keep the staged files, let the user retry.
      setPhase('editing')
      setGeneralError(e instanceof Error ? e.message : 'Failed to create the notebook.')
      return
    }
    setCreatedNb(nb)

    // 2) Upload staged files into the new notebook (if any).
    const pending = uploaderRef.current?.pendingCount() ?? 0
    if (pending === 0) {
      finished(nb, { succeeded: 0, failed: [] })
      return
    }
    setPhase('uploading')
    try {
      const summary = await uploaderRef.current?.uploadAll(nb.id)
      finished(nb, summary ?? { succeeded: 0, failed: [] })
    } catch {
      setPhase('partial')
      setFailedSummary([{ filename: '', error: 'Upload interrupted. You can retry below.' }])
    }
  }

  function finished(nb: Notebook, summary: UploadSummary) {
    if (summary.failed.length === 0) {
      toast(
        summary.succeeded > 0
          ? `Notebook created — ${summary.succeeded} document${summary.succeeded === 1 ? '' : 's'} uploaded.`
          : 'Notebook created.',
        'success',
      )
      onCreated?.(nb)
      onClose()
      reset()
      navigate(`/notebooks/${nb.id}`)
      return
    }
    // Partial success: stay open so the user can retry failed files without
    // losing them (the notebook already exists — no duplicate will be created).
    setFailedSummary(summary.failed)
    setPhase('partial')
  }

  async function retryFailed() {
    if (createdNb === null) return
    setGeneralError(null)
    setPhase('uploading')
    try {
      const summary = await uploaderRef.current?.uploadAll(createdNb.id)
      if ((summary?.failed.length ?? 0) === 0) {
        toast(`All documents uploaded to the notebook.`, 'success')
        onCreated?.(createdNb) // parent refreshes its list + sidebar too
        onClose()
        reset()
        navigate(`/notebooks/${createdNb.id}`)
      } else {
        setFailedSummary(summary?.failed ?? [])
        setPhase('partial')
      }
    } catch {
      setPhase('partial')
    }
  }

  return (
    <Modal
      open={open}
      onClose={handleClose} // guarded: no-ops while uploads are running
      title={phase === 'uploading' ? 'Uploading documents…' : 'Create Notebook'}
      width="max-w-lg"
    >
      <form
        onSubmit={(e) => {
          e.preventDefault()
          void handleSubmit()
        }}
        className="space-y-4"
      >
        <fieldset disabled={inputDisabled} className="space-y-4 disabled:opacity-90">
          <div>
            <label htmlFor="cn-title" className="block text-sm font-medium text-ink mb-1.5">
              Title
            </label>
            <input
              id="cn-title"
              type="text"
              value={name}
              onChange={(e) => {
                setName(e.target.value)
                if (titleError) setTitleError(null)
              }}
              placeholder="e.g. Biology Research"
              maxLength={120}
              aria-invalid={titleError !== null}
              aria-describedby={titleError ? 'cn-title-error' : undefined}
              className="w-full h-11 px-3.5 rounded-xl border border-line bg-cream/50 text-sm text-ink outline-none focus:border-lavender focus:bg-card transition-colors"
            />
            {titleError && (
              <p id="cn-title-error" className="text-xs text-coral-deep mt-1" role="alert">
                {titleError}
              </p>
            )}
          </div>

          <div>
            <label htmlFor="cn-desc" className="block text-sm font-medium text-ink mb-1.5">
              Description <span className="text-muted font-normal">(optional)</span>
            </label>
            <textarea
              id="cn-desc"
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="What is this notebook about?"
              rows={2}
              maxLength={2000}
              className="w-full px-3.5 py-2.5 rounded-xl border border-line bg-cream/50 text-sm text-ink outline-none focus:border-lavender focus:bg-card transition-colors resize-none"
            />
          </div>

          <div>
            <span className="block text-sm font-medium text-ink mb-1.5">Colour</span>
            <div className="flex gap-2" role="radiogroup" aria-label="Notebook colour">
              {colors.map((c) => (
                <button
                  key={c}
                  type="button"
                  role="radio"
                  aria-checked={color === c}
                  onClick={() => setColor(c)}
                  className={`h-8 w-8 rounded-full ${colorBar[c]} ${
                    color === c ? 'ring-2 ring-offset-2 ring-ink/30 scale-110' : 'opacity-70 hover:opacity-100'
                  } transition-all`}
                  aria-label={`Colour ${c}`}
                />
              ))}
            </div>
          </div>
        </fieldset>

        {/* Documents (optional but fully supported at creation time) */}
        <div>
          <p className="block text-sm font-medium text-ink mb-1.5">
            Documents <span className="text-muted font-normal">(optional — you can add sources later)</span>
          </p>
          <FileUploader ref={uploaderRef} compact onItemsChange={(items) => setStagedCount(items.length)} />
        </div>

        {generalError && (
          <div className="flex items-start gap-2 text-sm text-coral-deep bg-coral-soft/50 rounded-xl px-3.5 py-2.5" role="alert">
            <span className="min-w-0 break-words">{generalError}</span>
          </div>
        )}

        {phase === 'partial' && (
          <div className="bg-coral-soft/40 border border-coral/25 rounded-xl px-3.5 py-3 text-sm" role="alert">
            <p className="font-medium text-coral-deep mb-1">
              Notebook created, but {failedSummary.length} file{failedSummary.length === 1 ? '' : 's'} failed to upload.
            </p>
            <ul className="text-xs text-coral-deep/90 space-y-1 mb-2.5 max-h-24 overflow-y-auto">
              {failedSummary.map((f) => (
                <li key={f.filename} className="break-words">
                  <strong>{f.filename}</strong> — {f.error}
                </li>
              ))}
            </ul>
            <div className="flex gap-2">
              <Button type="button" size="sm" variant="danger" onClick={() => void retryFailed()}>
                Retry failed uploads
              </Button>
              <Button type="button" size="sm" variant="outline" onClick={openCreated}>
                <ArrowRight size={13} aria-hidden="true" /> Open notebook anyway
              </Button>
            </div>
          </div>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <Button type="button" variant="ghost" onClick={handleClose} disabled={busy}>
            Cancel
          </Button>
          <Button type="submit" loading={busy} disabled={!name.trim() && phase === 'editing'}>
            {busy ? 'Creating…' : stagedCount > 0 ? `Create Notebook (${stagedCount})` : 'Create Notebook'}
          </Button>
        </div>
      </form>
    </Modal>
  )
}
