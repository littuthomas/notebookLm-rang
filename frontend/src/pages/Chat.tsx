import { useEffect, useRef, useState } from 'react'
import {
  BookmarkPlus,
  ClipboardCopy,
  FileText,
  MessageSquareText,
  RotateCcw,
  Send,
  Sparkles,
} from 'lucide-react'
import Button from '../components/common/Button'
import EmptyState from '../components/common/EmptyState'
import { askQuestion, listNotebooks, saveAnswer } from '../services/api'
import type { ChatResponse, Notebook } from '../services/api'

interface ChatMsg {
  role: 'user' | 'assistant'
  text: string
  sources?: ChatResponse['sources']
  foundContext?: boolean
  model?: string
}

export default function Chat() {
  const [notebooks, setNotebooks] = useState<Notebook[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const [savedIdx, setSavedIdx] = useState<Set<number>>(new Set())
  const endRef = useRef<HTMLDivElement>(null)

  // Notebooks are independent knowledge spaces, so every question must pick
  // which one to answer from.
  useEffect(() => {
    void listNotebooks()
      .then((nbs) => {
        setNotebooks(nbs)
        setSelectedId(nbs.length > 0 ? nbs[0].id : null)
      })
      .catch(() => setNotebooks([]))
  }, [])

  const selected = notebooks.find((n) => n.id === selectedId) ?? null

  function selectNotebook(id: number | null) {
    setSelectedId(id)
    // Changing notebooks must clear the old notebook's chat + sources so no
    // retrieval context or conversation leaks across notebooks.
    setMessages([])
    setSavedIdx(new Set())
    setInput('')
  }

  const scrollDown = () => requestAnimationFrame(() => endRef.current?.scrollIntoView({ behavior: 'smooth' }))

  async function ask(question: string) {
    if (!question.trim() || thinking || selectedId === null) return
    setMessages((prev) => [...prev, { role: 'user', text: question }])
    setInput('')
    setThinking(true)
    scrollDown()
    try {
      const res = await askQuestion(question, selectedId)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: res.answer, sources: res.sources, foundContext: res.found_context, model: res.model },
      ])
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: e instanceof Error ? e.message : 'The AI request failed. Please try again.', foundContext: false },
      ])
    } finally {
      setThinking(false)
      scrollDown()
    }
  }

  function handleSave(idx: number) {
    const msg = messages[idx]
    const question = messages[idx - 1]?.text ?? ''
    void saveAnswer(question, msg.text, msg.sources ?? [], selectedId)
    setSavedIdx((prev) => new Set(prev).add(idx))
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2.5 px-5 lg:px-8 h-14 border-b border-line bg-card/50 backdrop-blur shrink-0">
        <span className="h-8 w-8 rounded-xl bg-lavender text-white flex items-center justify-center text-sm" aria-hidden="true">✦</span>
        <div className="min-w-0">
          <h1 className="font-semibold text-ink text-sm">AI Chat</h1>
          <p className="text-[11px] text-muted truncate max-w-[220px] md:max-w-none">
            {selected
              ? `Answering only from: ${selected.name}`
              : 'Grounded in a single notebook — pick one below'}
          </p>
        </div>

        {/* Notebook selector: the selected notebook scopes every answer */}
        <label className="ml-auto flex items-center gap-2 shrink-0">
          <span className="text-[11px] font-medium text-muted hidden sm:inline">Notebook</span>
          <select
            value={selectedId ?? ''}
            onChange={(e) => selectNotebook(e.target.value === '' ? null : Number(e.target.value))}
            aria-label="Select notebook for this chat"
            className="h-9 max-w-[160px] md:max-w-none px-2.5 rounded-xl border border-line bg-card text-sm text-ink outline-none focus:border-lavender transition-colors"
          >
            {notebooks.length === 0 && <option value="">No notebooks yet</option>}
            {notebooks.map((n) => (
              <option key={n.id} value={n.id}>
                {n.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-5 lg:p-8">
        <div className="max-w-3xl mx-auto space-y-5">
          {messages.length === 0 && (
            <EmptyState
              icon={<MessageSquareText size={30} aria-hidden="true" />}
              title={selected ? `Ask anything about ${selected.name}` : 'Pick a notebook to start asking'}
              description="Every answer comes only from the selected notebook's documents and cites the chunks it was grounded in."
            />
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex gap-3 anim-fade-up ${m.role === 'user' ? 'justify-end' : ''}`}>
              {m.role === 'assistant' && (
                <span className="h-8 w-8 rounded-xl bg-lavender text-white flex items-center justify-center text-xs shrink-0" aria-hidden="true">✦</span>
              )}
              <div className="max-w-[85%] md:max-w-[75%]">
                <div
                  className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                    m.role === 'user'
                      ? 'bg-lavender text-white rounded-br-md'
                      : 'bg-card border border-line text-ink rounded-tl-md'
                  }`}
                >
                  {m.text}
                </div>

                {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                  <div className="mt-2.5">
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-muted mb-1.5">Sources</p>
                    <div className="flex flex-wrap gap-1.5">
                      {m.sources.map((s) => (
                        <span
                          key={s.chunk_id}
                          title={s.snippet}
                          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-lavender-soft text-lavender-deep text-[11px] font-medium"
                        >
                          <FileText size={11} aria-hidden="true" />
                          Doc {s.document_id.slice(0, 6)} · chunk {s.chunk_index + 1}
                          {s.similarity_score !== null && ` · ${Math.round(s.similarity_score * 100)}%`}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {m.role === 'assistant' && m.foundContext === false && (
                  <p className="text-xs text-sun mt-1.5">No relevant context was found in your indexed documents.</p>
                )}

                {m.role === 'assistant' && i >= 1 && (
                  <div className="flex gap-3 mt-2.5">
                    <button
                      type="button"
                      onClick={() => void navigator.clipboard.writeText(m.text)}
                      className="text-xs text-muted hover:text-ink inline-flex items-center gap-1 transition-colors"
                    >
                      <ClipboardCopy size={12} aria-hidden="true" /> Copy
                    </button>
                    <button
                      type="button"
                      onClick={() => handleSave(i)}
                      disabled={savedIdx.has(i)}
                      className="text-xs text-muted hover:text-lavender-deep inline-flex items-center gap-1 transition-colors disabled:text-mint disabled:hover:text-mint"
                    >
                      <BookmarkPlus size={12} aria-hidden="true" /> {savedIdx.has(i) ? 'Saved' : 'Save answer'}
                    </button>
                    <button
                      type="button"
                      onClick={() => void ask(messages[i - 1]?.text ?? '')}
                      className="text-xs text-muted hover:text-ink inline-flex items-center gap-1 transition-colors"
                    >
                      <RotateCcw size={12} aria-hidden="true" /> Regenerate
                    </button>
                  </div>
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
          <div ref={endRef} />
        </div>
      </div>

      {/* Input */}
      <div className="p-4 border-t border-line bg-card/60 backdrop-blur shrink-0">
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
            placeholder={selectedId === null ? 'Select a notebook first…' : `Ask anything about ${selected?.name ?? 'this notebook'}…`}
            rows={1}
            aria-label="Chat message"
            className="flex-1 resize-none max-h-32 px-4 py-3 rounded-xl border border-line bg-card text-sm text-ink outline-none focus:border-lavender transition-colors"
          />
          <Button type="submit" size="lg" disabled={!input.trim() || thinking || selectedId === null} aria-label="Send message">
            <Send size={16} aria-hidden="true" />
          </Button>
        </form>
        <p className="text-center text-[11px] text-muted mt-2 flex items-center justify-center gap-1">
          <Sparkles size={11} aria-hidden="true" /> Answers are generated only from the selected notebook's documents
        </p>
      </div>
    </div>
  )
}
