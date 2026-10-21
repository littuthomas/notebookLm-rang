import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  ClipboardCopy,
  FileText,
  Layers,
  Loader2,
  MessageSquareText,
  Plus,
  RotateCcw,
  Send,
  Sparkles,
  StickyNote,
  Trash2,
  X,
} from 'lucide-react'
import Button from '../components/common/Button'
import ErrorState from '../components/common/ErrorState'
import Loading from '../components/common/Loading'
import Modal from '../components/common/Modal'
import AddSourcesModal from '../components/documents/AddSourcesModal'
import { askQuestion, deleteDocument, getNotebook, saveAnswer } from '../services/api'
import type { ChatResponse, DocumentMeta, NotebookDetail as NotebookDetailType } from '../services/api'
import { formatBytes, iconForFileType, typeLabel } from '../utils/files'
import { toast } from '../utils/toast'

interface ChatMsg {
  role: 'user' | 'assistant'
  text: string
  sources?: ChatResponse['sources']
  foundContext?: boolean
}

const studioTools = [
  { id: 'summary', label: 'Document Summary', icon: Sparkles, prompt: 'Provide a concise summary of the key points across my sources.' },
  { id: 'guide', label: 'Study Guide', icon: Layers, prompt: 'Create a study guide from my sources, organised by topic with review questions.' },
  { id: 'faq', label: 'FAQ', icon: MessageSquareText, prompt: 'List frequently asked questions about the content of my sources, with brief answers.' },
  { id: 'concepts', label: 'Key Takeaways', icon: StickyNote, prompt: 'Extract the most important takeaways and key terms defined in my sources, with short definitions.' },
]

/** Status chip for an indexed document (backend statuses: ready / failed / processing). */
function StatusChip({ status }: { status: DocumentMeta['status'] }) {
  if (status === 'ready')
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-mint-soft text-mint shrink-0">
        <CheckCircle2 size={10} aria-hidden="true" /> Ready
      </span>
    )
  if (status === 'failed')
    return (
      <span
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-coral-soft text-coral-deep shrink-0"
        title="Processing failed. Remove this document and upload it again."
      >
        <AlertCircle size={10} aria-hidden="true" /> Failed
      </span>
    )
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-lavender-soft text-lavender-deep shrink-0">
      <Loader2 size={10} className="animate-spin" aria-hidden="true" /> Processing
    </span>
  )
}

export default function NotebookDetail() {
  const { id } = useParams<{ id: string }>()
  const notebookId = Number(id)
  const [nb, setNb] = useState<NotebookDetailType | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Chat state
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const chatEndRef = useRef<HTMLDivElement>(null)

  // Add-sources modal + upload-note banner
  const [addSourcesOpen, setAddSourcesOpen] = useState(false)
  const [uploadNote, setUploadNote] = useState<string | null>(null)

  // Delete-document confirmation
  const [deleteTarget, setDeleteTarget] = useState<DocumentMeta | null>(null)
  const [deleting, setDeleting] = useState(false)

  // Expanded citation (per-message source inspection)
  const [openSource, setOpenSource] = useState<{ msgIdx: number; chunkId: number } | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setNb(await getNotebook(notebookId))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load notebook')
    } finally {
      setLoading(false)
    }
  }, [notebookId])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking])

  async function ask(question: string) {
    if (!question.trim() || thinking) return
    setMessages((prev) => [...prev, { role: 'user', text: question }])
    setInput('')
    setThinking(true)
    try {
      // Scope the answer to THIS notebook's documents only — chunks from
      // other notebooks are never retrieved into the context/sources.
      const res = await askQuestion(question, notebookId)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: res.answer, sources: res.sources, foundContext: res.found_context },
      ])
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: e instanceof Error ? e.message : 'The AI request failed. Please try again.', foundContext: false },
      ])
    } finally {
      setThinking(false)
    }
  }

  async function handleRemoveDocument() {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await deleteDocument(deleteTarget.id)
      setNb(
        (prev) =>
          prev && {
            ...prev,
            documents: prev.documents.filter((d) => d.id !== deleteTarget.id),
            source_count: Math.max(0, prev.source_count - 1),
          },
      )
      toast(`"${deleteTarget.filename}" removed from this notebook.`, 'info')
      setDeleteTarget(null)
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Failed to remove the document.', 'error')
    } finally {
      setDeleting(false)
    }
  }

  /** document_uuid -> display filename for citation chips. */
  const filenameByUuid = new Map((nb?.documents ?? []).map((d) => [d.document_uuid, d.filename]))

  if (loading) return <Loading label="Loading notebook…" />
  if (error || !nb)
    return (
      <div className="p-8 max-w-lg mx-auto">
        <ErrorState message={error ?? 'Notebook not found.'} onRetry={load} />
      </div>
    )

  const hasSources = nb.documents.length > 0

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-5 lg:px-8 h-16 border-b border-line bg-card/60 backdrop-blur shrink-0">
        <Link to="/notebooks" className="p-2 -ml-2 rounded-xl text-muted hover:text-ink hover:bg-cream transition-colors" aria-label="Back to notebooks">
          <ArrowLeft size={18} />
        </Link>
        <div className="min-w-0">
          <h1 className="font-semibold text-ink truncate">{nb.name}</h1>
          <p className="text-xs text-muted">{nb.source_count} sources · {nb.question_count} questions</p>
        </div>
        <Button size="sm" variant="outline" className="ml-auto" onClick={() => setAddSourcesOpen(true)}>
          <Plus size={14} aria-hidden="true" /> Add Sources
        </Button>
      </div>
      {uploadNote && (
        <div className="px-5 lg:px-8 py-2.5 bg-mint-soft text-mint text-sm flex items-center gap-2 anim-fade-in">
          <CheckCircle2 size={15} aria-hidden="true" /> {uploadNote}
          <button type="button" onClick={() => setUploadNote(null)} className="ml-auto p-0.5 hover:opacity-70" aria-label="Dismiss">
            <X size={14} />
          </button>
        </div>
      )}

      <div className="flex flex-1 min-h-0 flex-col xl:flex-row">
        {/* LEFT: sources */}
        <aside className="xl:w-64 shrink-0 border-b xl:border-b-0 xl:border-r border-line bg-sidebar/50 p-4 overflow-y-auto max-h-52 xl:max-h-none">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted mb-3 flex items-center gap-1.5">
            <Layers size={13} aria-hidden="true" /> Sources
          </h2>
          {!hasSources ? (
            <div className="text-center py-6">
              <p className="text-sm text-muted mb-3">No sources yet.</p>
              <Button size="sm" variant="outline" onClick={() => setAddSourcesOpen(true)}>
                <Plus size={13} aria-hidden="true" /> Upload files
              </Button>
              <p className="text-[11px] text-muted mt-3">PDF, DOCX, XLSX, TXT, MD or CSV — up to 20 MB.</p>
            </div>
          ) : (
            <ul className="space-y-1.5">
              {nb.documents.map((d) => {
                const Icon = iconForFileType(d.file_type)
                return (
                  <li key={d.id} className="group flex items-start gap-2.5 px-3 py-2.5 bg-card rounded-xl border border-line">
                    <Icon size={15} className="text-lavender shrink-0 mt-0.5" aria-hidden="true" />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm text-ink truncate" title={d.filename}>{d.filename}</p>
                      <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                        <span className="text-[10px] text-muted">{typeLabel(d.file_type)} · {formatBytes(d.size_bytes)}</span>
                        <StatusChip status={d.status} />
                      </div>
                      {d.error_message && d.error_message.length > 0 && (
                        <p className="text-[10px] text-coral-deep mt-0.5 line-clamp-2" title={d.error_message}>{d.error_message}</p>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={() => setDeleteTarget(d)}
                      className="p-1 rounded-lg text-muted opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-coral-deep hover:bg-coral-soft transition-all shrink-0"
                      aria-label={`Remove ${d.filename}`}
                    >
                      <Trash2 size={13} aria-hidden="true" />
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
          {hasSources && (
            <button
              type="button"
              onClick={() => setAddSourcesOpen(true)}
              className="mt-3 w-full flex items-center justify-center gap-2 py-2.5 rounded-xl border border-dashed border-line text-sm text-muted hover:text-lavender hover:border-lavender/50 transition-colors"
            >
              <Plus size={14} aria-hidden="true" /> Add source
            </button>
          )}
        </aside>

        {/* CENTER: chat */}
        <section className="flex-1 flex flex-col min-w-0 min-h-[420px]" aria-label="AI chat">
          <div className="flex-1 overflow-y-auto p-5 lg:p-6 space-y-5">
            {messages.length === 0 && (
              <div className="text-center py-14 anim-fade-up">
                <span className="inline-flex h-14 w-14 rounded-2xl bg-lavender-soft items-center justify-center text-lavender mb-4">
                  <MessageSquareText size={24} aria-hidden="true" />
                </span>
                {hasSources ? (
                  <>
                    <h3 className="font-semibold text-ink mb-1">Ask anything about your sources</h3>
                    <p className="text-sm text-muted max-w-sm mx-auto mb-6">
                      Answers are grounded in this notebook&apos;s documents with citations.
                    </p>
                    <div className="flex flex-wrap justify-center gap-2">
                      {studioTools.slice(0, 3).map((t) => (
                        <button
                          key={t.id}
                          type="button"
                          onClick={() => void ask(t.prompt)}
                          className="px-3.5 py-2 rounded-xl bg-card border border-line text-sm text-ink hover:border-lavender/50 hover:bg-lavender-soft/40 transition-all"
                        >
                          <t.icon size={14} className="inline mr-1.5 -mt-0.5 text-lavender" aria-hidden="true" />
                          {t.label}
                        </button>
                      ))}
                    </div>
                  </>
                ) : (
                  <>
                    <h3 className="font-semibold text-ink mb-1">Add sources to get started</h3>
                    <p className="text-sm text-muted max-w-sm mx-auto mb-6">
                      Upload documents to this notebook, then ask questions — every answer is
                      grounded in your sources with citations.
                    </p>
                    <Button onClick={() => setAddSourcesOpen(true)}>
                      <Plus size={15} aria-hidden="true" /> Add Sources
                    </Button>
                  </>
                )}
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={`flex gap-3 anim-fade-up ${m.role === 'user' ? 'justify-end' : ''}`}>
                {m.role === 'assistant' && (
                  <span className="h-8 w-8 rounded-xl bg-lavender text-white flex items-center justify-center text-xs shrink-0" aria-hidden="true">✦</span>
                )}
                <div className={`max-w-[85%] md:max-w-[75%] ${m.role === 'user' ? 'order-first' : ''}`}>
                  <div
                    className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                      m.role === 'user'
                        ? 'bg-lavender text-white rounded-br-md'
                        : 'bg-card border border-line text-ink rounded-tl-md'
                    }`}
                  >
                    {m.text}
                  </div>

                  {/* Citations */}
                  {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                    <div className="mt-2.5">
                      <p className="text-[11px] font-semibold uppercase tracking-wide text-muted mb-1.5">Sources</p>
                      <div className="flex flex-wrap gap-1.5">
                        {m.sources.map((s) => {
                          const open = openSource?.msgIdx === i && openSource.chunkId === s.chunk_id
                          return (
                            <button
                              key={s.chunk_id}
                              type="button"
                              onClick={() => setOpenSource(open ? null : { msgIdx: i, chunkId: s.chunk_id })}
                              aria-expanded={open}
                              className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-medium transition-colors ${
                                open
                                  ? 'bg-lavender text-white'
                                  : 'bg-lavender-soft text-lavender-deep hover:bg-lavender hover:text-white'
                              }`}
                            >
                              <FileText size={11} aria-hidden="true" />
                              {filenameByUuid.get(s.document_id) ?? s.document_id.slice(0, 8)} · chunk {s.chunk_index + 1}
                            </button>
                          )
                        })}
                      </div>

                      {/* Source detail panel (clickable citation) */}
                      {openSource?.msgIdx === i &&
                        m.sources
                          .filter((s) => s.chunk_id === openSource.chunkId)
                          .map((s) => (
                            <div key={s.chunk_id} className="mt-2 bg-cream/70 border border-line rounded-xl p-3 anim-fade-in">
                              <div className="flex items-center justify-between gap-2 mb-1.5">
                                <p className="text-[11px] font-semibold text-ink truncate">
                                  {filenameByUuid.get(s.document_id) ?? s.document_id.slice(0, 8)} · chunk {s.chunk_index + 1}
                                  {s.similarity_score !== null && ` · ${Math.round(s.similarity_score * 100)}% match`}
                                </p>
                                <button
                                  type="button"
                                  onClick={() => setOpenSource(null)}
                                  className="p-0.5 rounded text-muted hover:text-ink"
                                  aria-label="Close source preview"
                                >
                                  <X size={12} aria-hidden="true" />
                                </button>
                              </div>
                              <p className="text-xs text-ink/85 leading-relaxed line-clamp-6 whitespace-pre-wrap">{s.snippet}</p>
                            </div>
                          ))}

                      <div className="flex gap-2 mt-2.5">
                        <button
                          type="button"
                          onClick={() => void navigator.clipboard.writeText(m.text)}
                          className="text-xs text-muted hover:text-ink inline-flex items-center gap-1 transition-colors"
                        >
                          <ClipboardCopy size={12} aria-hidden="true" /> Copy
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            void saveAnswer(messages[i - 1]?.text ?? '', m.text, m.sources ?? [], notebookId)
                            toast('Answer saved. Find it under Saved Answers.', 'success')
                          }}
                          className="text-xs text-muted hover:text-lavender-deep inline-flex items-center gap-1 transition-colors"
                        >
                          <StickyNote size={12} aria-hidden="true" /> Save answer
                        </button>
                        <button
                          type="button"
                          onClick={() => void ask(messages[i - 1]?.text ?? '')}
                          className="text-xs text-muted hover:text-ink inline-flex items-center gap-1 transition-colors"
                        >
                          <RotateCcw size={12} aria-hidden="true" /> Regenerate
                        </button>
                      </div>
                    </div>
                  )}
                  {m.role === 'assistant' && m.foundContext === false && (
                    <p className="text-xs text-sun mt-1.5">No relevant context was found in your sources.</p>
                  )}
                </div>
              </div>
            ))}

            {thinking && (
              <div className="flex gap-3 anim-fade-in">
                <span className="h-8 w-8 rounded-xl bg-lavender text-white flex items-center justify-center text-xs shrink-0" aria-hidden="true">✦</span>
                <div className="bg-card border border-line rounded-2xl rounded-tl-md px-4 py-3.5 flex items-center gap-1.5" role="status" aria-label="AI is thinking">
                  <span className="thinking-dot" style={{ animationDelay: '0ms' }} />
                  <span className="thinking-dot" style={{ animationDelay: '150ms' }} />
                  <span className="thinking-dot" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-line bg-card/60 backdrop-blur">
            <form
              onSubmit={(e) => {
                e.preventDefault()
                void ask(input)
              }}
              className="flex items-end gap-2 max-w-3xl mx-auto"
            >
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    void ask(input)
                  }
                }}
                placeholder={hasSources ? 'Ask anything about your sources…' : 'Add sources first, then ask anything…'}
                rows={1}
                aria-label="Chat message"
                className="flex-1 resize-none max-h-32 px-4 py-3 rounded-xl border border-line bg-card text-sm text-ink outline-none focus:border-lavender transition-colors"
              />
              <Button type="submit" size="lg" disabled={!input.trim() || thinking} aria-label="Send message">
                <Send size={16} aria-hidden="true" />
              </Button>
            </form>
            <p className="text-center text-[11px] text-muted mt-2">
              Enter to send · Shift + Enter for a new line
            </p>
          </div>
        </section>

        {/* RIGHT: studio */}
        <aside className="xl:w-60 shrink-0 border-t xl:border-t-0 xl:border-l border-line bg-sidebar/50 p-4 overflow-y-auto max-h-40 xl:max-h-none">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted mb-3 flex items-center gap-1.5">
            <Sparkles size={13} aria-hidden="true" /> Studio
          </h2>
          {hasSources ? (
            <div className="space-y-1.5">
              {studioTools.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => void ask(t.prompt)}
                  disabled={thinking}
                  className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl bg-card border border-line text-sm text-ink hover:border-lavender/50 hover:bg-lavender-soft/40 hover:shadow-sm transition-all disabled:opacity-50 text-left"
                >
                  <t.icon size={15} className="text-lavender shrink-0" aria-hidden="true" />
                  {t.label}
                </button>
              ))}
              <p className="text-[10px] text-muted px-1 pt-1.5 leading-relaxed">
                Each tool asks the AI scoped to this notebook&apos;s sources and opens the result in chat.
              </p>
            </div>
          ) : (
            <p className="text-xs text-muted">Studio tools appear once at least one source is uploaded.</p>
          )}
        </aside>
      </div>

      {/* Add Sources (uploads into THIS notebook) */}
      <AddSourcesModal
        notebookId={notebookId}
        open={addSourcesOpen}
        onClose={() => setAddSourcesOpen(false)}
        onUploaded={async () => {
          await load()
          if (nb && nb.documents.length === 0) {
            setUploadNote('Documents uploaded. You can now ask questions about them.')
          }
        }}
      />

      {/* Remove document confirmation */}
      <Modal open={deleteTarget !== null} onClose={() => setDeleteTarget(null)} title="Remove Source">
        <p className="text-sm text-muted mb-5">
          Remove <strong className="text-ink">{deleteTarget?.filename}</strong> from this notebook? Its indexed
          chunks will be deleted and it can no longer be used to answer questions.
        </p>
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setDeleteTarget(null)}>Cancel</Button>
          <Button variant="danger" loading={deleting} onClick={() => void handleRemoveDocument()}>
            <Trash2 size={15} aria-hidden="true" /> Remove
          </Button>
        </div>
      </Modal>
    </div>
  )
}
