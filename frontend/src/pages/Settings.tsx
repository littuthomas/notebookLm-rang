import { useEffect, useState } from 'react'
import { Bot, Database, HardDrive, Palette, Shield, User } from 'lucide-react'
import { API_BASE_URL, getStats } from '../services/api'
import { useAuth } from '../context/AuthContext'

interface Section {
  id: string
  title: string
  icon: typeof User
  rows: { label: string; value: string }[]
}

export default function Settings() {
  const { user } = useAuth()
  const [stats, setStats] = useState<{ total_chunks: number; documents: number } | null>(null)

  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(() => setStats(null))
  }, [])

  const sections: Section[] = [
    {
      id: 'account',
      title: 'Account',
      icon: User,
      rows: [
        { label: 'Username', value: user?.username ?? '—' },
        { label: 'Email', value: user?.email ?? '—' },
        { label: 'Member since', value: user ? new Date(user.created_at).toLocaleDateString() : '—' },
      ],
    },
    {
      id: 'appearance',
      title: 'Appearance',
      icon: Palette,
      rows: [
        { label: 'Theme', value: 'Light (warm cream)' },
        { label: 'Accent colour', value: 'Lavender' },
      ],
    },
    {
      id: 'ai',
      title: 'AI Model',
      icon: Bot,
      rows: [
        { label: 'Answer generation', value: 'Google Gemini' },
        { label: 'Embeddings', value: 'gemini-embedding-2 · 768 dims' },
        { label: 'Fallback LLM', value: 'Nemotron Ultra via OpenRouter (if configured)' },
      ],
    },
    {
      id: 'database',
      title: 'Database',
      icon: Database,
      rows: [
        { label: 'Engine', value: 'PostgreSQL (Neon) with pgvector' },
        { label: 'Indexed chunks', value: stats ? String(stats.total_chunks) : '—' },
        { label: 'Documents', value: stats ? String(stats.documents) : '—' },
      ],
    },
    {
      id: 'storage',
      title: 'Storage & API',
      icon: HardDrive,
      rows: [
        { label: 'API base URL', value: API_BASE_URL },
        { label: 'Max upload size', value: '20 MB' },
        { label: 'Supported files', value: 'PDF, DOCX, XLSX, TXT, MD, CSV' },
      ],
    },
    {
      id: 'privacy',
      title: 'Privacy & Security',
      icon: Shield,
      rows: [
        { label: 'Authentication', value: 'JWT (Bearer tokens)' },
        { label: 'API keys', value: 'Stored server-side only — never exposed to this app' },
        { label: 'Answers', value: 'Generated only from your own indexed documents' },
      ],
    },
  ]

  return (
    <div className="p-5 lg:p-8 max-w-3xl mx-auto space-y-6">
      <div className="anim-fade-up">
        <h1 className="text-2xl font-semibold text-ink tracking-tight">Settings</h1>
        <p className="text-sm text-muted mt-0.5">Current application configuration. Sensitive values stay on the server.</p>
      </div>

      <div className="space-y-4">
        {sections.map((section, i) => (
          <section
            key={section.id}
            className="bg-card rounded-2xl border border-line shadow-sm overflow-hidden anim-fade-up"
            style={{ animationDelay: `${i * 40}ms` }}
          >
            <div className="flex items-center gap-3 px-5 py-4 border-b border-line bg-cream/40">
              <span className="h-8 w-8 rounded-lg bg-lavender-soft text-lavender-deep flex items-center justify-center">
                <section.icon size={16} aria-hidden="true" />
              </span>
              <h2 className="font-semibold text-ink text-sm">{section.title}</h2>
            </div>
            <dl className="divide-y divide-line">
              {section.rows.map((row) => (
                <div key={row.label} className="flex items-center justify-between gap-4 px-5 py-3">
                  <dt className="text-sm text-muted shrink-0">{row.label}</dt>
                  <dd className="text-sm text-ink font-medium text-right truncate">{row.value}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </div>
  )
}
