import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Grid2X2,
  LayoutList,
  MoreVertical,
  Notebook as NotebookIcon,
  Pencil,
  Plus,
  Search,
  Trash2,
} from 'lucide-react'
import Button from '../components/common/Button'
import Modal from '../components/common/Modal'
import EmptyState from '../components/common/EmptyState'
import ErrorState from '../components/common/ErrorState'
import { SkeletonCard } from '../components/common/Loading'
import CreateNotebookModal from '../components/documents/CreateNotebookModal'
import { deleteNotebook, listNotebooks, updateNotebook } from '../services/api'
import type { Notebook } from '../services/api'

const colorClasses: Record<string, { chip: string; bar: string }> = {
  lavender: { chip: 'bg-lavender-soft text-lavender-deep', bar: 'bg-lavender' },
  coral: { chip: 'bg-coral-soft text-coral-deep', bar: 'bg-coral' },
  sun: { chip: 'bg-sun-soft text-ink', bar: 'bg-sun' },
  mint: { chip: 'bg-mint-soft text-mint', bar: 'bg-mint' },
}

export default function Notebooks() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [notebooks, setNotebooks] = useState<Notebook[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [view, setView] = useState<'grid' | 'list'>('grid')
  const [sort, setSort] = useState<'recent' | 'name' | 'sources'>('recent')

  // create modal (auto-opened via ?new=1); the upload-ready dialog lives in
  // components/documents/CreateNotebookModal
  const [createOpen, setCreateOpen] = useState(searchParams.get('new') === '1')

  // rename / delete
  const [menuFor, setMenuFor] = useState<number | null>(null)
  const [renameTarget, setRenameTarget] = useState<Notebook | null>(null)
  const [renameName, setRenameName] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<Notebook | null>(null)
  const [deleting, setDeleting] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setNotebooks(await listNotebooks())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load notebooks')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  // Clear ?new=1 once the modal opens
  useEffect(() => {
    if (createOpen && searchParams.get('new')) {
      setSearchParams({}, { replace: true })
    }
  }, [createOpen, searchParams, setSearchParams])

  const sorted = useMemo(() => {
    const q = query.trim().toLowerCase()
    let list = notebooks.filter(
      (nb) => nb.name.toLowerCase().includes(q) || nb.description.toLowerCase().includes(q),
    )
    if (sort === 'name') list = [...list].sort((a, b) => a.name.localeCompare(b.name))
    else if (sort === 'sources') list = [...list].sort((a, b) => b.source_count - a.source_count)
    // 'recent' order comes from the API (pinned + updated desc)
    return list
  }, [notebooks, query, sort])

  function handleCreated(nb: Notebook) {
    // Fresh notebook from the create-with-upload dialog: prepend and let the
    // sidebar/topbar refresh too.
    setNotebooks((prev) => [nb, ...prev])
    window.dispatchEvent(new Event('notebooks:changed'))
  }

  async function handleRename() {
    if (!renameTarget || !renameName.trim()) return
    try {
      const updated = await updateNotebook(renameTarget.id, { name: renameName.trim() })
      setNotebooks((prev) => prev.map((nb) => (nb.id === updated.id ? updated : nb)))
      setRenameTarget(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to rename notebook')
    }
  }

  async function handleDelete() {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await deleteNotebook(deleteTarget.id)
      setNotebooks((prev) => prev.filter((nb) => nb.id !== deleteTarget.id))
      setDeleteTarget(null)
      window.dispatchEvent(new Event('notebooks:changed'))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete notebook')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="p-5 lg:p-8 max-w-6xl mx-auto space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 anim-fade-up">
        <div>
          <h1 className="text-2xl font-semibold text-ink tracking-tight">My Notebooks</h1>
          <p className="text-sm text-muted mt-0.5">{notebooks.length} notebook{notebooks.length === 1 ? '' : 's'}</p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus size={16} aria-hidden="true" /> Create Notebook
        </Button>
      </div>

      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3 anim-fade-up">
        <div className="relative flex-1 min-w-[200px] max-w-md">
          <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-muted" aria-hidden="true" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search notebooks…"
            aria-label="Search notebooks"
            className="w-full h-10 pl-10 pr-4 rounded-xl border border-line bg-card text-sm text-ink outline-none focus:border-lavender transition-colors"
          />
        </div>
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as typeof sort)}
          aria-label="Sort notebooks"
          className="h-10 px-3 rounded-xl border border-line bg-card text-sm text-ink outline-none focus:border-lavender"
        >
          <option value="recent">Most recent</option>
          <option value="name">Name A–Z</option>
          <option value="sources">Most sources</option>
        </select>
        <div className="flex rounded-xl border border-line bg-card overflow-hidden" role="group" aria-label="View mode">
          <button
            type="button"
            onClick={() => setView('grid')}
            className={`p-2.5 ${view === 'grid' ? 'bg-lavender-soft text-lavender-deep' : 'text-muted hover:text-ink'}`}
            aria-label="Grid view"
            aria-pressed={view === 'grid'}
          >
            <Grid2X2 size={16} />
          </button>
          <button
            type="button"
            onClick={() => setView('list')}
            className={`p-2.5 ${view === 'list' ? 'bg-lavender-soft text-lavender-deep' : 'text-muted hover:text-ink'}`}
            aria-label="List view"
            aria-pressed={view === 'list'}
          >
            <LayoutList size={16} />
          </button>
        </div>
      </div>

      {error && <ErrorState message={error} onRetry={load} />}

      {loading ? (
        <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <SkeletonCard key={i} className="h-44" />
          ))}
        </div>
      ) : sorted.length === 0 ? (
        <EmptyState
          icon={<NotebookIcon size={30} aria-hidden="true" />}
          title={query ? 'No matching notebooks' : 'No notebooks yet'}
          description={
            query
              ? 'Try a different search term.'
              : 'Create your first notebook and start turning documents into knowledge.'
          }
          action={
            !query && (
              <Button onClick={() => setCreateOpen(true)}>
                <Plus size={16} aria-hidden="true" /> Create Notebook
              </Button>
            )
          }
        />
      ) : (
        <div className={view === 'grid' ? 'grid sm:grid-cols-2 xl:grid-cols-3 gap-4' : 'space-y-3'}>
          {sorted.map((nb, i) => {
            const c = colorClasses[nb.color] ?? colorClasses.lavender
            return (
              <div
                key={nb.id}
                className={`group relative bg-card rounded-2xl border border-line shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 anim-fade-up ${
                  view === 'list' ? 'flex items-center gap-4 p-4' : 'p-5 overflow-hidden'
                }`}
                style={{ animationDelay: `${i * 40}ms` }}
              >
                {view === 'grid' && <span className={`absolute top-0 left-0 h-full w-1 ${c.bar}`} aria-hidden="true" />}
                <div className={view === 'list' ? 'contents' : ''}>
                  <div className="flex items-start gap-3">
                    <div className={`h-10 w-10 rounded-xl flex items-center justify-center shrink-0 ${c.chip}`}>
                      <NotebookIcon size={18} aria-hidden="true" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <Link to={`/notebooks/${nb.id}`} className="font-semibold text-ink hover:text-lavender-deep transition-colors block truncate">
                        {nb.name}
                      </Link>
                      <p className={`text-sm text-muted ${view === 'grid' ? 'line-clamp-1' : 'line-clamp-1'}`}>
                        {nb.description || 'No description'}
                      </p>
                    </div>
                    {/* Menu */}
                    <div className="relative shrink-0">
                      <button
                        type="button"
                        onClick={() => setMenuFor(menuFor === nb.id ? null : nb.id)}
                        className="p-1.5 rounded-lg text-muted hover:text-ink hover:bg-cream transition-colors"
                        aria-label={`Options for ${nb.name}`}
                        aria-expanded={menuFor === nb.id}
                      >
                        <MoreVertical size={16} />
                      </button>
                      {menuFor === nb.id && (
                        <div
                          className="absolute right-0 mt-1 w-40 bg-card rounded-xl border border-line shadow-lg py-1 z-10 anim-pop-in"
                          onMouseLeave={() => setMenuFor(null)}
                        >
                          <button
                            type="button"
                            onClick={() => {
                              setRenameTarget(nb)
                              setRenameName(nb.name)
                              setMenuFor(null)
                            }}
                            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-ink hover:bg-cream"
                          >
                            <Pencil size={14} /> Rename
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setDeleteTarget(nb)
                              setMenuFor(null)
                            }}
                            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-coral-deep hover:bg-coral-soft"
                          >
                            <Trash2 size={14} /> Delete
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className={`flex items-center gap-4 text-xs text-muted ${view === 'grid' ? 'mt-4' : 'ml-auto'}`}>
                    <span>{nb.source_count} sources</span>
                    <span>{nb.question_count} questions</span>
                    <span>{new Date(nb.updated_at).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Create modal (title/description/colour + drag-and-drop upload) */}
      <CreateNotebookModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={handleCreated}
      />

      {/* Rename modal */}
      <Modal open={renameTarget !== null} onClose={() => setRenameTarget(null)} title="Rename Notebook">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            void handleRename()
          }}
          className="space-y-4"
        >
          <input
            type="text"
            value={renameName}
            onChange={(e) => setRenameName(e.target.value)}
            autoFocus
            aria-label="Notebook name"
            className="w-full h-11 px-3.5 rounded-xl border border-line bg-cream/50 text-sm text-ink outline-none focus:border-lavender focus:bg-card transition-colors"
          />
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" onClick={() => setRenameTarget(null)}>Cancel</Button>
            <Button type="submit" disabled={!renameName.trim()}>Save</Button>
          </div>
        </form>
      </Modal>

      {/* Delete confirm modal */}
      <Modal open={deleteTarget !== null} onClose={() => setDeleteTarget(null)} title="Delete Notebook">
        <p className="text-sm text-muted mb-5">
          Delete <strong className="text-ink">{deleteTarget?.name}</strong>? Its notes and saved answers will also be
          removed. Uploaded documents stay in your Documents page.
        </p>
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setDeleteTarget(null)}>Cancel</Button>
          <Button variant="danger" loading={deleting} onClick={() => void handleDelete()}>
            <Trash2 size={15} aria-hidden="true" /> Delete
          </Button>
        </div>
      </Modal>
    </div>
  )
}
