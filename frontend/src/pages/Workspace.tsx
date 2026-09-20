import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './Workspace.css'

/**
 * RAG Workspace — the main authenticated page.
 *
 * Loaded after a successful login. Reads the access_token from localStorage
 * (never the password — it is not stored anywhere) and sends it as
 * "Authorization: Bearer <token>" on every protected request.
 *
 * Layout:
 * - Header:    logo, username (from /api/auth/me), logout
 * - Sidebar:   "Chats" list (previous sessions, New Chat, delete) +
 *              "My Documents" list (original filename when known, chunk count,
 *              falling back to the document_id) + upload
 * - Main area: ask-a-question box, conversation (saved + live messages),
 *              sources with similarity scores for the live answer
 *
 * Chat history: the first question of a new chat auto-creates a session
 * (POST /api/chat/sessions); every WebSocket start frame then carries its
 * session_id so the backend persists each completed Q&A. Opening a session
 * loads its stored messages; deleting removes it server-side.
 *
 * On any 401 the token is removed and the user is sent back to the Login page
 * through App.tsx's temporary view switch (onLogout).
 */

const API_BASE = 'http://127.0.0.1:8002'
// WebSocket endpoint for streaming question-answering (same backend, ws scheme).
const WS_CHAT_URL = API_BASE.replace(/^http/, 'ws') + '/api/chat/ws'
const TOKEN_KEY = 'access_token'
const NOTEBOOK_ID = 'default'
// localStorage key for the document_id → original filename map (see FilenameMap).
const FILENAME_KEY = 'rag_document_filenames'

// Accept files the backend supports (PDF, DOCX, XLSX, XLS, TXT, MD, CSV).
const ACCEPTED_TYPES = '.pdf,.docx,.xlsx,.xls,.txt,.md,.csv'
const MAX_UPLOAD_MB = 20

/** One entry of GET /api/documents — only ids and counts (no filename). */
interface DocumentSummary {
  document_id: string
  chunk_count: number
}

/**
 * document_id → original filename, remembered from upload responses.
 * GET /api/documents does not return filenames, so the frontend keeps this
 * local map; entries without a cached name fall back to the document_id.
 */
type FilenameMap = Record<string, string>

/** Shape of GET /api/documents. */
interface DocumentsResponse {
  documents: DocumentSummary[]
  total: number
}

/** Shape of POST /api/documents/upload. */
interface UploadResponse {
  document_id: string
  filename: string
  file_type: string
  status: string
  page_count: number
  chunk_count: number
  chunks_stored: number
  error_message: string | null
}

/** One retrieved chunk in a chat answer. */
interface ChatSource {
  chunk_id: number
  document_id: string
  chunk_index: number
  similarity_score: number | null
  snippet: string
}

/** Shape of the answer display state (from POST /api/chat or the WS flow). */
interface ChatResponse {
  question: string
  answer: string
  sources: ChatSource[]
  found_context: boolean
  /** The WS protocol does not send a model field — only REST does. */
  model?: string
}

/** One session in GET /api/chat/sessions. */
interface SessionSummary {
  id: number
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

/** Shape of GET /api/chat/sessions. */
interface SessionsResponse {
  sessions: SessionSummary[]
  total: number
}

/** One stored message in GET /api/chat/sessions/{id}. */
interface SessionMessage {
  id: number
  role: string
  content: string
  created_at: string
}

/** Shape of GET /api/chat/sessions/{id}. */
interface SessionDetailResponse {
  id: number
  title: string
  created_at: string
  updated_at: string
  messages: SessionMessage[]
}

/**
 * One message in the on-screen conversation.
 *
 * Historical messages loaded from the backend carry content only (no
 * sources); sources exist just for the live streaming answer.
 */
interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
  sources?: ChatSource[]
  /** Mirrors the WS context frame; undefined when retrieval never answered. */
  foundContext?: boolean
}

/**
 * Server → client WebSocket messages from /api/chat/ws.
 * Mirrors backend app/schemas/ws.py (context · answer_chunk · complete · error).
 */
interface WsContextMessage {
  type: 'context'
  found_context: boolean
  sources: ChatSource[]
}

interface WsAnswerChunkMessage {
  type: 'answer_chunk'
  chunk: string
}

interface WsCompleteMessage {
  type: 'complete'
}

interface WsErrorMessage {
  type: 'error'
  detail: string
}

type WsServerMessage =
  | WsContextMessage
  | WsAnswerChunkMessage
  | WsCompleteMessage
  | WsErrorMessage

/**
 * Build a friendly error message from a failed fetch Response.
 * Mirrors the backend: 400 bad request · 401 expired token · 403 ·
 * 413 file too large · 422 validation/processing · 500 server error.
 */
async function toApiErrorMessage(res: Response): Promise<string> {
  // Try to read FastAPI's { detail: "..." } error body if there is one.
  let detail: unknown = null
  try {
    const body = (await res.json()) as { detail?: unknown } | null
    detail = body?.detail
  } catch {
    // Non-JSON body — fall through to the status-based messages below.
  }

  if (res.status === 400) return typeof detail === 'string' ? detail : 'Bad request. Please check your input.'
  if (res.status === 401) return 'Your session has expired. Please log in again.'
  if (res.status === 403) return 'You do not have permission to do that.'
  if (res.status === 413) return 'That file is too large. The maximum upload size is 20 MB.'
  if (res.status === 422) {
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      const first = detail[0] as { msg?: string } | undefined
      if (first?.msg) return `Validation error: ${first.msg}.`
    }
    return 'Validation error: please check your input and try again.'
  }
  if (res.status >= 500) return 'The server had a problem. Please try again in a moment.'

  return typeof detail === 'string' ? detail : 'Something went wrong. Please try again.'
}

/** Read the filename map from localStorage (empty object if missing/corrupt). */
function loadFilenameMap(): FilenameMap {
  try {
    const raw = localStorage.getItem(FILENAME_KEY)
    return raw ? (JSON.parse(raw) as FilenameMap) : {}
  } catch {
    return {}
  }
}

/** Persist the filename map (best-effort — filenames are cosmetic). */
function saveFilenameMap(map: FilenameMap): void {
  try {
    localStorage.setItem(FILENAME_KEY, JSON.stringify(map))
  } catch {
    // Ignore: storage unavailable or full.
  }
}

/** Small inline SVGs (no icon library needed — same approach as Login/Signup). */

function SparkleIcon() {
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8" />
    </svg>
  )
}

function DocIcon() {
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6M9 13h6M9 17h6" />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M17 8l-5-5-5 5M12 3v12" />
    </svg>
  )
}

function SendIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M22 2 11 13M22 2l-7 20-4-9-9-4z" />
    </svg>
  )
}

function ChatIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      <path d="M8 9h8M8 13h5" />
    </svg>
  )
}

function PlusIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 5v14M5 12h14" />
    </svg>
  )
}

function TrashIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
    </svg>
  )
}

function LogoutIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <path d="M16 17l5-5-5-5M21 12H9" />
    </svg>
  )
}

function SpinnerIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      aria-hidden="true"
      className="ws-spin"
    >
      <path d="M21 12a9 9 0 1 1-6.2-8.56" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M20 6 9 17l-5-5" />
    </svg>
  )
}

function AlertIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 8v4M12 16h.01" />
    </svg>
  )
}

/** Props from App.tsx's temporary view switch. */
interface WorkspaceProps {
  username: string
  onLogout: () => void
}

export default function Workspace({ username, onLogout }: WorkspaceProps) {
  // --- Authentication state ---
  const token = localStorage.getItem(TOKEN_KEY)

  // --- User state ---
  const [meError, setMeError] = useState<string | null>(null)

  // --- Documents state ---
  const [documents, setDocuments] = useState<DocumentSummary[] | null>(null)
  // document_id → original filename (saved from upload responses, kept in localStorage).
  const [filenames, setFilenames] = useState<FilenameMap>(loadFilenameMap)
  const [documentsError, setDocumentsError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<{ kind: 'success' | 'error'; text: string } | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // --- Chat state ---
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [askError, setAskError] = useState<string | null>(null)
  // Live metadata for the answer currently streaming (sources, found_context).
  // The conversation itself lives in `messages`; `chatResult` is kept only
  // while a stream is in flight.
  const [chatResult, setChatResult] = useState<ChatResponse | null>(null)

  // --- Chat history state (Stage 4) ---
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null)
  const [sessionsError, setSessionsError] = useState<string | null>(null)
  const [sessionsBusy, setSessionsBusy] = useState(false)
  // null = fresh unsaved chat; the session is auto-created on the first Ask.
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null)
  const [activeSessionTitle, setActiveSessionTitle] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [loadingSession, setLoadingSession] = useState(false)
  // Track the session a stream belongs to, so late frames from a cancelled
  // stream can never be appended to a different conversation.
  const streamSessionRef = useRef<number | null>(null)

  // Active streaming socket, so it can be closed on unmount.
  const wsRef = useRef<WebSocket | null>(null)
  // Prevents state updates after close/unmount during the async WS session.
  const wsSessionRef = useRef(false)

  // Close an in-flight streaming socket when the Workspace unmounts.
  useEffect(() => {
    return () => {
      wsSessionRef.current = false
      streamSessionRef.current = null
      const ws = wsRef.current
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close()
      }
      wsRef.current = null
    }
  }, [])

  /** Shared 401 handling: drop the token and return to Login. */
  const handleUnauthorized = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY)
    onLogout()
  }, [onLogout])

  /** GET /api/documents — refreshes the sidebar list. */
  const refreshDocuments = useCallback(async () => {
    if (!token) {
      handleUnauthorized()
      return
    }
    try {
      const res = await fetch(`${API_BASE}/api/documents`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.status === 401) {
        handleUnauthorized()
        return
      }
      if (!res.ok) {
        throw new Error(await toApiErrorMessage(res))
      }
      const data = (await res.json()) as DocumentsResponse
      setDocuments(data.documents)
      setDocumentsError(null)
    } catch (err) {
      setDocumentsError(
        err instanceof TypeError
          ? 'Unable to connect to the server. Please make sure the backend is running.'
          : err instanceof Error
            ? err.message
            : 'Could not load your documents.',
      )
    }
  }, [token, handleUnauthorized])

  /** GET /api/chat/sessions — refreshes the sidebar chats list. */
  const refreshSessions = useCallback(async () => {
    if (!token) {
      handleUnauthorized()
      return
    }
    try {
      const res = await fetch(`${API_BASE}/api/chat/sessions`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.status === 401) {
        handleUnauthorized()
        return
      }
      if (!res.ok) {
        throw new Error(await toApiErrorMessage(res))
      }
      const data = (await res.json()) as SessionsResponse
      setSessions(data.sessions)
      setSessionsError(null)
    } catch (err) {
      setSessionsError(
        err instanceof TypeError
          ? 'Unable to connect to the server. Please make sure the backend is running.'
          : err instanceof Error
            ? err.message
            : 'Could not load your chats.',
      )
    }
  }, [token, handleUnauthorized])

  /**
   * Cancel any in-flight WS stream. Used when opening another session,
   * starting a New Chat, deleting the active session, or on unmount paths.
   * Late frames are ignored via wsSessionRef + streamSessionRef.
   */
  const cancelStream = useCallback(() => {
    wsSessionRef.current = false
    streamSessionRef.current = null
    const ws = wsRef.current
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      try {
        ws.close()
      } catch {
        // Already closing/closed.
      }
    }
    if (wsRef.current === ws) {
      wsRef.current = null
    }
    setAsking(false)
  }, [])

  /** Start a fresh unsaved chat: cancel any stream and clear the conversation. */
  const handleNewChat = useCallback(() => {
    cancelStream()
    setActiveSessionId(null)
    setActiveSessionTitle(null)
    setMessages([])
    setChatResult(null)
    setAskError(null)
    setQuestion('')
  }, [cancelStream])

  /** Open a previous session: load its stored messages into the main area. */
  const handleOpenSession = useCallback(
    async (sessionId: number) => {
      if (!token) {
        handleUnauthorized()
        return
      }
      cancelStream()
      setLoadingSession(true)
      setAskError(null)
      try {
        const res = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.status === 401) {
          handleUnauthorized()
          return
        }
        if (!res.ok) {
          throw new Error(await toApiErrorMessage(res))
        }
        const data = (await res.json()) as SessionDetailResponse
        setActiveSessionId(data.id)
        setActiveSessionTitle(data.title)
        setMessages(
          data.messages.map((m) => ({
            role: m.role === 'assistant' ? 'assistant' : 'user',
            content: m.content,
          })),
        )
        setChatResult(null)
      } catch (err) {
        setAskError(
          err instanceof TypeError
            ? 'Unable to connect to the server. Please make sure the backend is running.'
            : err instanceof Error
              ? err.message
              : 'Could not load the conversation.',
        )
      } finally {
        setLoadingSession(false)
      }
    },
    [token, handleUnauthorized, cancelStream],
  )

  /** Delete a session after confirmation; if active, reset to a fresh chat. */
  const handleDeleteSession = useCallback(
    async (sessionId: number) => {
      if (!token) {
        handleUnauthorized()
        return
      }
      if (!window.confirm('Delete this chat? Its messages are removed permanently.')) {
        return
      }
      setSessionsBusy(true)
      try {
        const res = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}`, {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.status === 401) {
          handleUnauthorized()
          return
        }
        if (!res.ok) {
          throw new Error(await toApiErrorMessage(res))
        }
        // Refresh the list; if the active session was deleted, start fresh.
        await refreshSessions()
        if (activeSessionId === sessionId) {
          handleNewChat()
        }
      } catch (err) {
        setAskError(
          err instanceof TypeError
            ? 'Unable to connect to the server. Please make sure the backend is running.'
            : err instanceof Error
              ? err.message
              : 'Could not delete the chat.',
        )
      } finally {
        setSessionsBusy(false)
      }
    },
    [token, handleUnauthorized, refreshSessions, activeSessionId, handleNewChat],
  )

  /** Load the profile and the document list once when the Workspace opens. */
  useEffect(() => {
    let cancelled = false

    async function loadInitialData() {
      if (!token) {
        handleUnauthorized()
        return
      }

      // Step 1: verify the token and load the username via /api/auth/me.
      try {
        const meRes = await fetch(`${API_BASE}/api/auth/me`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (meRes.status === 401) {
          handleUnauthorized()
          return
        }
        if (!meRes.ok) {
          throw new Error(await toApiErrorMessage(meRes))
        }
        // We only need to verify the token here — App.tsx already has the
        // username from Login. Keep the response read (dispose the body).
        await meRes.json()
      } catch (err) {
        if (!cancelled) {
          setMeError(
            err instanceof TypeError
              ? 'Unable to connect to the server. Please make sure the backend is running.'
              : err instanceof Error
                ? err.message
                : 'Could not load your profile.',
          )
        }
      }

      // Step 2: load the document list and the chat sessions for the sidebar.
      if (!cancelled) {
        await Promise.all([refreshDocuments(), refreshSessions()])
      }
    }

    loadInitialData()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /** POST /api/documents/upload with FormData, then refresh the list. */
  const handleUpload = async (file: File) => {
    if (!token) {
      handleUnauthorized()
      return
    }

    // Small client-side guard (the backend is the authority).
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setUploadStatus({ kind: 'error', text: `"${file.name}" is larger than 20 MB.` })
      return
    }

    setUploading(true)
    setUploadStatus(null)

    try {
      const body = new FormData()
      body.append('file', file)
      body.append('notebook_id', NOTEBOOK_ID)

      const res = await fetch(`${API_BASE}/api/documents/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body,
      })

      if (res.status === 401) {
        handleUnauthorized()
        return
      }
      if (!res.ok) {
        throw new Error(await toApiErrorMessage(res))
      }

      const data = (await res.json()) as UploadResponse
      // Remember the original filename so the sidebar can display it
      // (GET /api/documents only returns the document_id).
      if (data.document_id && data.filename) {
        const next = { ...filenames, [data.document_id]: data.filename }
        setFilenames(next)
        saveFilenameMap(next)
      }
      setUploadStatus({
        kind: 'success',
        text: `Uploaded "${data.filename}" — ${data.chunks_stored} chunks stored.`,
      })
      // Show the new document in the sidebar right away.
      await refreshDocuments()
    } catch (err) {
      setUploadStatus({
        kind: 'error',
        text:
          err instanceof TypeError
            ? 'Unable to connect to the server. Please make sure the backend is running.'
            : err instanceof Error
              ? err.message
              : 'Upload failed. Please try again.',
      })
    } finally {
      setUploading(false)
      // Allow picking the same file again (the input's onChange only fires on change).
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  /**
   * Ask a question via the streaming WebSocket /api/chat/ws.
   *
   * Flow: if no session is active yet, one is auto-created first
   * (POST /api/chat/sessions). The start frame then carries the session_id so
   * the backend persists the completed Q&A. Frames append to the on-screen
   * conversation; sources attach to the live assistant message only.
   */
  const handleAsk = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (asking) return

    const trimmed = question.trim()
    if (!trimmed) {
      setAskError('Please type a question first.')
      return
    }

    if (!token) {
      handleUnauthorized()
      return
    }

    setAsking(true)
    setAskError(null)

    // --- Auto-create the session on the first Ask of a new chat ---
    let sessionId = activeSessionId
    if (sessionId === null) {
      try {
        const res = await fetch(`${API_BASE}/api/chat/sessions`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.status === 401) {
          handleUnauthorized()
          return
        }
        if (!res.ok) {
          throw new Error(await toApiErrorMessage(res))
        }
        const created = (await res.json()) as SessionSummary
        sessionId = created.id
        setActiveSessionId(created.id)
        setActiveSessionTitle(created.title)
        setSessions((prev) =>
          prev ? [{ ...created, message_count: 0 }, ...prev] : [{ ...created, message_count: 0 }],
        )
      } catch (err) {
        setAsking(false)
        setAskError(
          err instanceof TypeError
            ? 'Unable to connect to the server. Please make sure the backend is running.'
            : err instanceof Error
              ? err.message
              : 'Could not start a new chat.',
        )
        return
      }
    }

    // --- Append the user message + an empty assistant placeholder ---
    setMessages((prev) => [...prev, { role: 'user', content: trimmed }, { role: 'assistant', content: '' }])
    setQuestion('')
    setChatResult(null)
    wsSessionRef.current = true
    streamSessionRef.current = sessionId

    let ws: WebSocket
    try {
      ws = new WebSocket(WS_CHAT_URL)
    } catch {
      setAsking(false)
      wsSessionRef.current = false
      setAskError('Unable to connect to the server. Please make sure the backend is running.')
      return
    }
    wsRef.current = ws

    ws.onopen = () => {
      if (!wsSessionRef.current) return
      ws.send(JSON.stringify({ type: 'start', token, question: trimmed, session_id: sessionId }))
    }

    ws.onmessage = (event: MessageEvent) => {
      if (!wsSessionRef.current) return

      let msg: WsServerMessage
      try {
        msg = JSON.parse(event.data) as WsServerMessage
      } catch {
        return // ignore malformed frames
      }

      switch (msg.type) {
        case 'context':
          // Open the answer section and show sources as soon as retrieval lands.
          setChatResult({
            question: trimmed,
            answer: '',
            sources: msg.sources ?? [],
            found_context: msg.found_context,
          })
          // Mark the live placeholder for the no-context rendering path.
          if (msg.found_context === false) {
            setMessages((prev) => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last && last.role === 'assistant' && last.content === '') {
                next[next.length - 1] = { ...last, foundContext: false }
              }
              return next
            })
          }
          break

        case 'answer_chunk':
          // Progressively append the streamed answer text.
          if (msg.chunk) {
            setChatResult((prev) =>
              prev ? { ...prev, answer: prev.answer + msg.chunk } : prev,
            )
            setMessages((prev) => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last && last.role === 'assistant') {
                next[next.length - 1] = { ...last, content: last.content + msg.chunk }
              }
              return next
            })
          }
          break

        case 'complete':
          finishStreaming(ws)
          // The backend owns the stored title/counts — refresh the sidebar.
          refreshSessions()
          break

        case 'error': {
          // Auth failures behave like the REST 401 path: drop token, return to Login.
          if (msg.detail.startsWith('Authentication failed')) {
            handleUnauthorized()
            return
          }
          setAskError(msg.detail || 'Something went wrong. Please try again.')
          finishStreaming(ws)
          break
        }
      }
    }

    ws.onerror = () => {
      if (!wsSessionRef.current) return
      setAskError('Unable to connect to the server. Please make sure the backend is running.')
      finishStreaming(ws)
    }

    ws.onclose = () => {
      // If the server closed early (before complete/error), surface a friendly
      // error; after a normal finish this is a no-op because the session flag
      // and asking state were already reset by finishStreaming.
      if (!wsSessionRef.current) return
      setAskError('The connection to the server was closed before the answer finished.')
      finishStreaming(ws)
    }
  }

  /** End the streaming session: reset loading state and close the socket. */
  function finishStreaming(ws: WebSocket) {
    wsSessionRef.current = false
    streamSessionRef.current = null
    setAsking(false)
    try {
      ws.close()
    } catch {
      // Already closing/closed.
    }
    if (wsRef.current === ws) {
      wsRef.current = null
    }
  }

  return (
    <div className="ws-page">
      {/* ---------- Header ---------- */}
      <header className="ws-header">
        <div className="ws-header-inner">
          <span className="logo">
            <span className="logo-mark" aria-hidden="true">
              <SparkleIcon />
            </span>
            RAG System
          </span>

          <div className="ws-header-actions">
            <span className="ws-username" title={`Logged in as ${username}`}>
              {username}
            </span>
            <button type="button" className="btn btn-outline ws-logout" onClick={onLogout}>
              <LogoutIcon />
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* ---------- Body: sidebar + main ---------- */}
      <div className="ws-body">
        {/* ---------- Sidebar: My Documents ---------- */}
        <aside className="ws-sidebar">
          {/* ---------- Sidebar: Chats (history) ---------- */}
          <div className="ws-chats-head">
            <h2>Chats</h2>
            <button
              type="button"
              className="btn btn-outline ws-new-chat-btn"
              onClick={handleNewChat}
              disabled={asking || loadingSession}
            >
              <PlusIcon />
              New Chat
            </button>
          </div>

          {sessionsError && (
            <div className="ws-chats-status ws-chats-error" role="alert">
              <AlertIcon />
              <span>{sessionsError}</span>
            </div>
          )}

          <ul className="ws-chat-list">
            {sessions === null && !sessionsError && (
              <li className="ws-chat-list-empty">
                <SpinnerIcon />
                Loading chats…
              </li>
            )}

            {sessions !== null && sessions.length === 0 && !sessionsError && (
              <li className="ws-chat-list-empty">
                <ChatIcon />
                No chats yet. Ask your first question to start one.
              </li>
            )}

            {sessions !== null &&
              sessions.map((session) => (
                <li
                  key={session.id}
                  className={`ws-chat-item${session.id === activeSessionId ? ' ws-chat-item-active' : ''}`}
                >
                  <button
                    type="button"
                    className="ws-chat-open"
                    onClick={() => handleOpenSession(session.id)}
                    disabled={loadingSession}
                    title={session.title}
                  >
                    <span className="ws-chat-icon" aria-hidden="true">
                      <ChatIcon />
                    </span>
                    <span className="ws-chat-info">
                      <span className="ws-chat-title">{session.title}</span>
                      <span className="ws-chat-meta">
                        {session.message_count}{' '}
                        {session.message_count === 1 ? 'message' : 'messages'}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    className="ws-chat-delete"
                    onClick={() => handleDeleteSession(session.id)}
                    disabled={sessionsBusy}
                    aria-label={`Delete chat: ${session.title}`}
                    title="Delete chat"
                  >
                    <TrashIcon />
                  </button>
                </li>
              ))}
          </ul>

          {/* ---------- Sidebar: My Documents ---------- */}
          <div className="ws-sidebar-head">
            <h2>My Documents</h2>
            <p className="ws-sidebar-count">
              {documents === null ? 'Loading…' : `${documents.length} total`}
            </p>
          </div>

          {/* Upload control */}
          <div className="ws-upload">
            <input
              ref={fileInputRef}
              id="ws-file-input"
              type="file"
              accept={ACCEPTED_TYPES}
              className="ws-file-input"
              disabled={uploading}
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) handleUpload(file)
              }}
            />
            <label htmlFor="ws-file-input" className={`btn btn-primary ws-upload-btn${uploading ? ' ws-btn-disabled' : ''}`}>
              {uploading ? (
                <>
                  <SpinnerIcon />
                  Uploading…
                </>
              ) : (
                <>
                  <UploadIcon />
                  Upload Document
                </>
              )}
            </label>
            <p className="ws-upload-hint">PDF, DOCX, XLSX, TXT, MD, CSV · max 20 MB</p>
          </div>

          {uploadStatus && (
            <div
              className={`ws-upload-status ${uploadStatus.kind === 'success' ? 'ws-status-success' : 'ws-status-error'}`}
              role="status"
            >
              {uploadStatus.kind === 'success' ? <CheckIcon /> : <AlertIcon />}
              <span>{uploadStatus.text}</span>
            </div>
          )}

          {/* Document list */}
          <ul className="ws-doc-list">
            {documents === null && (
              <li className="ws-doc-empty">
                <SpinnerIcon />
                Loading documents…
              </li>
            )}

            {documents !== null && documentsError && (
              <li className="ws-doc-empty ws-doc-error">
                <AlertIcon />
                {documentsError}
              </li>
            )}

            {documents !== null && !documentsError && documents.length === 0 && (
              <li className="ws-doc-empty">
                <DocIcon />
                No documents yet. Upload your first file to get started.
              </li>
            )}

            {documents !== null &&
              !documentsError &&
              documents.map((doc) => {
                // Display the remembered original filename when we have one
                // (saved from the upload response); otherwise fall back to the
                // document_id — names are never invented.
                const displayName = filenames[doc.document_id] ?? doc.document_id
                return (
                <li key={doc.document_id} className="ws-doc-item">
                  <span className="ws-doc-icon" aria-hidden="true">
                    <DocIcon />
                  </span>
                  <div className="ws-doc-info">
                    {/* title keeps the real document_id available on hover. */}
                    <p className="ws-doc-id" title={doc.document_id}>
                      {displayName}
                    </p>
                    <p className="ws-doc-meta">
                      {doc.chunk_count} {doc.chunk_count === 1 ? 'chunk' : 'chunks'}
                    </p>
                  </div>
                </li>
                )
              })}
          </ul>
        </aside>

        {/* ---------- Main area: Ask your documents ---------- */}
        <main className="ws-main">
          <h1>Ask your documents</h1>
          <p className="ws-main-subtitle">
            Ask a question about your uploaded documents and get a grounded answer
            with citations.
          </p>

          {activeSessionTitle && (
            <p className="ws-active-session" title={activeSessionTitle}>
              Chat: {activeSessionTitle}
            </p>
          )}

          <form className="ws-ask-form" onSubmit={handleAsk}>
            <textarea
              className="ws-question"
              placeholder="e.g. What are the key points covered in my documents?"
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={asking}
              aria-label="Your question"
            />
            <button type="submit" className="btn btn-primary ws-ask-btn" disabled={asking || !question.trim()}>
              {asking ? (
                <>
                  <SpinnerIcon />
                  Searching your documents…
                </>
              ) : (
                <>
                  <SendIcon />
                  Ask
                </>
              )}
            </button>
          </form>

          {askError && (
            <div className="form-banner form-banner-error" role="alert">
              {askError}
            </div>
          )}

          {/* Profile load failure (non-401) — visible but not blocking chat. */}
          {meError && (
            <div className="form-banner form-banner-error" role="alert">
              {meError}
            </div>
          )}

          {/* ---------- Loading a previous session ---------- */}
          {loadingSession && (
            <div className="ws-session-loading">
              <SpinnerIcon />
              Loading conversation…
            </div>
          )}

          {/* ---------- Conversation ---------- */}
          {!loadingSession && messages.length > 0 && (
            <section className="ws-conversation" aria-live="polite">
              {messages.map((msg, index) => {
                const isLast = index === messages.length - 1
                const isStreamingBubble =
                  isLast && asking && msg.role === 'assistant'
                // Sources exist only for the live answer (chatResult); the
                // backend does not store citations with historical messages.
                const liveSources = isLast && chatResult ? chatResult.sources : []

                if (msg.role === 'user') {
                  return (
                    <div key={index} className="ws-msg ws-msg-user">
                      <p className="ws-msg-text">{msg.content}</p>
                    </div>
                  )
                }

                return (
                  <div key={index} className="ws-msg ws-msg-assistant">
                    {msg.content === '' &&
                      isStreamingBubble &&
                      msg.foundContext === undefined && (
                        <span className="ws-msg-streaming">
                          <SpinnerIcon />
                          Searching your documents…
                        </span>
                      )}

                    {msg.content === '' && msg.foundContext === false && (
                      /* RAG rule: no relevant context found — say so clearly. */
                      <div className="ws-answer ws-answer-missing">
                        <div className="ws-answer-head">
                          <span className="ws-answer-badge ws-badge-missing" aria-hidden="true">
                            <AlertIcon />
                          </span>
                          <h2>No relevant information found</h2>
                        </div>
                        <p className="ws-answer-text">
                          The available documents do not contain information relevant to
                          your question. Try rephrasing it, or upload a document that
                          covers this topic.
                        </p>
                      </div>
                    )}

                    {msg.content !== '' && (
                      <>
                        <div className="ws-answer-head">
                          <span className="ws-answer-badge" aria-hidden="true">
                            <CheckIcon />
                          </span>
                          <h2>Answer</h2>
                          {/* The WS protocol sends no model field — show it only when present. */}
                          {isLast && chatResult?.model && (
                            <span className="ws-model">Model: {chatResult.model}</span>
                          )}
                        </div>

                        <p className="ws-answer-text">{msg.content}</p>

                        {/* ---------- Sources / citations (live answer only) ---------- */}
                        {liveSources.length > 0 && (
                          <div className="ws-sources">
                            <h3>Sources</h3>
                            <ul className="ws-source-list">
                              {liveSources.map((source) => (
                                <li key={source.chunk_id} className="ws-source-item">
                                  <div className="ws-source-meta">
                                    <span className="ws-source-doc" title={source.document_id}>
                                      {source.document_id}
                                    </span>
                                    <span className="ws-source-chunk">chunk {source.chunk_index}</span>
                                    {source.similarity_score !== null && (
                                      <span className="ws-source-score">
                                        {Math.round(source.similarity_score * 100)}% match
                                      </span>
                                    )}
                                  </div>
                                  <blockquote className="ws-source-snippet">{source.snippet}</blockquote>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )
              })}
            </section>
          )}

          {/* ---------- Empty chat state ---------- */}
          {!loadingSession && messages.length === 0 && !asking && !askError && (
            <div className="ws-chat-empty">
              <span className="ws-chat-empty-icon" aria-hidden="true">
                <SparkleIcon />
              </span>
              <p>Your answer will appear here.</p>
              <p className="ws-chat-empty-hint">
                {documents !== null && documents.length === 0
                  ? 'Upload a document first, then ask your question.'
                  : 'Ask anything about the content of your documents.'}
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
