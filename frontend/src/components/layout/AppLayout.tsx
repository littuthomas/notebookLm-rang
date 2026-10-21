import { useCallback, useEffect, useMemo, useState } from 'react'
import { Outlet, useNavigate } from 'react-router-dom'
import { FileText, Library, MessageSquareText, BookmarkCheck, Search } from 'lucide-react'
import Sidebar from './Sidebar'
import Topbar from './Topbar'
import SearchModal from '../common/SearchModal'
import { ToastHost } from '../common/Toast'
import { listNotebooks } from '../../services/api'
import type { Notebook } from '../../services/api'

/**
 * Persistent application shell:
 * - Sidebar (collapsible on desktop, drawer on mobile)
 * - Topbar with global search + notifications + profile
 * - Global search across notebooks/documents/notes/answers (Ctrl+K)
 */
export default function AppLayout() {
  const navigate = useNavigate()
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)
  const [notebooks, setNotebooks] = useState<Notebook[]>([])

  // Load notebooks once for the sidebar + search
  const loadNotebooks = useCallback(async () => {
    try {
      setNotebooks(await listNotebooks())
    } catch {
      // Sidebar shows "No notebooks yet" on failure; pages show their own errors
    }
  }, [])

  useEffect(() => {
    void loadNotebooks()
  }, [loadNotebooks])

  // Ctrl/Cmd+K opens global search
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setSearchOpen(true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Expose a refresh hook so pages can update the sidebar after CRUD
  useEffect(() => {
    const handler = () => void loadNotebooks()
    window.addEventListener('notebooks:changed', handler)
    return () => window.removeEventListener('notebooks:changed', handler)
  }, [loadNotebooks])

  const searchItems = useMemo(() => {
    const items: { id: string; label: string; sub: string; icon: 'notebook' | 'chat' | 'doc' | 'answer'; to: string }[] = []
    for (const nb of notebooks) {
      items.push({ id: `nb-${nb.id}`, label: nb.name, sub: `Notebook · ${nb.source_count} sources`, icon: 'notebook', to: `/notebooks/${nb.id}` })
    }
    return items
  }, [notebooks])

  const iconFor = { notebook: Library, chat: MessageSquareText, doc: FileText, answer: BookmarkCheck }

  return (
    <div className="flex h-screen overflow-hidden bg-cream">
      <Sidebar
        recentNotebooks={notebooks}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed((v) => !v)}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <Topbar onOpenSearch={() => setSearchOpen(true)} onOpenMobileMenu={() => setMobileOpen(true)} />
        <main className="flex-1 overflow-y-auto">
          <Outlet context={{ refreshNotebooks: loadNotebooks }} />
        </main>
      </div>

      <SearchModal
        open={searchOpen}
        onClose={() => setSearchOpen(false)}
        items={searchItems}
        iconMap={iconFor}
        placeholder="Search notebooks, documents and answers…"
        onSelect={(to) => navigate(to)}
        footer={
          <button
            type="button"
            onClick={() => {
              setSearchOpen(false)
              navigate('/documents')
            }}
            className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-muted hover:text-ink hover:bg-cream transition-colors"
          >
            <Search size={15} aria-hidden="true" />
            Search all documents…
          </button>
        }
      />

      {/* Global toast notifications (see common/Toast) */}
      <ToastHost />
    </div>
  )
}
