import { useEffect, useState } from 'react'
import {
  BarChart3,
  Brain,
  ClipboardCopy,
  FileSearch,
  HelpCircle,
  ListChecks,
  Loader2,
  Sparkles,
  Target,
} from 'lucide-react'
import ErrorState from '../components/common/ErrorState'
import { askQuestion, listNotebooks } from '../services/api'
import type { Notebook } from '../services/api'

interface InsightTool {
  id: string
  title: string
  description: string
  icon: typeof Sparkles
  chip: string
  prompt: string
}

const tools: InsightTool[] = [
  { id: 'summary', title: 'Summary', description: 'A concise overview of the selected notebook.', icon: Sparkles, chip: 'bg-lavender-soft text-lavender-deep', prompt: 'Provide a concise summary of the key points in this notebook.' },
  { id: 'concepts', title: 'Key Concepts', description: 'The most important ideas and definitions found in the selected notebook.', icon: Brain, chip: 'bg-coral-soft text-coral-deep', prompt: 'Explain the key concepts found in this notebook.' },
  { id: 'faq', title: 'Frequently Asked Questions', description: 'Common questions the selected notebook answers, with brief answers.', icon: HelpCircle, chip: 'bg-sun-soft text-ink', prompt: 'List frequently asked questions about this notebook, with brief answers.' },
  { id: 'guide', title: 'Study Guide', description: 'A structured guide to help you learn the selected notebook.', icon: ListChecks, chip: 'bg-mint-soft text-mint', prompt: 'Create a study guide from this notebook.' },
  { id: 'quiz', title: 'Quiz', description: 'Test yourself with questions generated from the selected notebook.', icon: Target, chip: 'bg-lavender-soft text-lavender-deep', prompt: 'Create a short quiz based on this notebook.' },
  { id: 'topics', title: 'Important Topics', description: 'The topics that appear most prominently in the selected notebook.', icon: FileSearch, chip: 'bg-sun-soft text-ink', prompt: 'List the most important topics in this notebook.' },
  { id: 'analysis', title: 'Document Analysis', description: 'A high-level analysis of themes, tone and structure in the selected notebook.', icon: BarChart3, chip: 'bg-mint-soft text-mint', prompt: 'Provide a high-level analysis of this notebook: themes, tone and structure.' },
]

type Results = Record<string, { text: string; loading: boolean; error: string | null; empty: boolean }>

export default function Insights() {
  const [results, setResults] = useState<Results>({})
  const [docError, setDocError] = useState<string | null>(null)
  const [notebooks, setNotebooks] = useState<Notebook[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)

  // Insights are generated from one notebook at a time, so a target
  // notebook must be selected before any tool can run.
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
    // Switching notebooks clears the previous notebook's results so no
    // output generated from another notebook lingers here.
    setResults({})
    setDocError(null)
  }

  async function generate(tool: InsightTool) {
    if (selectedId === null) return
    setDocError(null)
    setResults((prev) => ({ ...prev, [tool.id]: { text: '', loading: true, error: null, empty: false } }))
    try {
      const res = await askQuestion(tool.prompt, selectedId)
      if (!res.found_context) {
        setResults((prev) => ({ ...prev, [tool.id]: { text: '', loading: false, error: null, empty: true } }))
      } else {
        setResults((prev) => ({ ...prev, [tool.id]: { text: res.answer, loading: false, error: null, empty: false } }))
      }
    } catch (e) {
      setResults((prev) => ({
        ...prev,
        [tool.id]: { text: '', loading: false, error: e instanceof Error ? e.message : 'Generation failed', empty: false },
      }))
    }
  }

  return (
    <div className="p-5 lg:p-8 max-w-6xl mx-auto space-y-6">
      <div className="anim-fade-up flex flex-wrap items-center gap-3">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold text-ink tracking-tight">AI Insights ✨</h1>
          <p className="text-sm text-muted mt-0.5">
            {selected
              ? `Generated only from: ${selected.name}`
              : 'Pick a notebook below to generate insights.'}
          </p>
        </div>

        {/* Notebook selector: every insight is scoped to this notebook */}
        <label className="ml-auto flex items-center gap-2 shrink-0">
          <span className="text-[11px] font-medium text-muted hidden sm:inline">Notebook</span>
          <select
            value={selectedId ?? ''}
            onChange={(e) => selectNotebook(e.target.value === '' ? null : Number(e.target.value))}
            aria-label="Select notebook for insights"
            className="h-9 max-w-[180px] md:max-w-none px-2.5 rounded-xl border border-line bg-card text-sm text-ink outline-none focus:border-lavender transition-colors"
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

      {docError && <ErrorState message={docError} />}

      <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
        {tools.map((tool, i) => {
          const r = results[tool.id]
          return (
            <div
              key={tool.id}
              className="bg-card rounded-2xl border border-line p-5 shadow-sm hover:shadow-md transition-all duration-200 flex flex-col anim-fade-up"
              style={{ animationDelay: `${i * 40}ms` }}
            >
              <div className="flex items-center gap-3 mb-3">
                <span className={`h-10 w-10 rounded-xl flex items-center justify-center shrink-0 ${tool.chip}`}>
                  <tool.icon size={18} aria-hidden="true" />
                </span>
                <h2 className="font-semibold text-ink">{tool.title}</h2>
              </div>
              <p className="text-sm text-muted mb-4">{tool.description}</p>

              {/* Result area */}
              <div className="flex-1 min-h-[80px]">
                {r?.loading ? (
                  <div className="flex items-center gap-2 text-sm text-lavender-deep py-3" role="status">
                    <Loader2 size={15} className="animate-spin" aria-hidden="true" /> Generating…
                  </div>
                ) : r?.error ? (
                  <ErrorState message={r.error} onRetry={() => void generate(tool)} />
                ) : r?.empty ? (
                  <p className="text-xs text-sun bg-sun-soft rounded-xl px-3 py-2.5">
                    No indexed documents yet — upload documents first.
                  </p>
                ) : r?.text ? (
                  <div className="text-sm text-ink/90 whitespace-pre-wrap leading-relaxed bg-cream/60 rounded-xl p-3.5 max-h-56 overflow-y-auto">
                    {r.text}
                  </div>
                ) : null}
              </div>

              <div className="flex items-center gap-2 mt-4">
                <button
                  type="button"
                  onClick={() => void generate(tool)}
                  disabled={r?.loading || selectedId === null}
                  className="flex-1 h-9 rounded-xl bg-lavender text-white text-sm font-medium hover:bg-lavender-deep transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {r?.text ? 'Regenerate' : 'Generate'}
                </button>
                {r?.text && (
                  <button
                    type="button"
                    onClick={() => void navigator.clipboard.writeText(r.text)}
                    className="h-9 px-3 rounded-xl border border-line text-muted hover:text-ink hover:bg-cream transition-colors"
                    aria-label={`Copy ${tool.title}`}
                  >
                    <ClipboardCopy size={14} className="inline" aria-hidden="true" />
                  </button>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {Object.keys(results).length === 0 && (
        <div className="bg-card rounded-2xl border border-dashed border-line py-10 px-6 text-center anim-fade-up">
          <Sparkles size={26} className="mx-auto text-sun mb-3" aria-hidden="true" />
          <p className="text-sm text-muted max-w-md mx-auto">
            Pick a notebook above, then any card — the insight is generated live from that notebook's documents with the
            same RAG pipeline used for chat.
          </p>
        </div>
      )}
    </div>
  )
}
