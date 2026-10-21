/**
 * Central API client for the RAG System backend.
 *
 * - Base URL comes from VITE_API_BASE_URL (see .env.example)
 * - JWT token is stored in localStorage under "access_token"
 * - Helper for JSON requests with auth + friendly error mapping
 */

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ??
  'http://localhost:8000'

export const TOKEN_KEY = 'access_token'

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

/** Backend error shape: { detail: string | Array<{ msg: string }> } */
type BackendError = { detail?: unknown }

async function detailToMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body = (await res.json()) as BackendError
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail) && body.detail.length > 0) {
      const first = body.detail[0] as { msg?: string }
      if (first?.msg) return first.msg.replace(/^Value error,?\s*/i, '')
    }
  } catch {
    // non-JSON body
  }
  return fallback
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE'
  body?: unknown
  auth?: boolean
}

/** Small JSON fetch wrapper with auth header + friendly errors. */
export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, auth = true } = options

  const headers: Record<string, string> = { Accept: 'application/json' }
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (auth) {
    const token = getToken()
    if (token) headers['Authorization'] = `Bearer ${token}`
  }

  let res: Response
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch {
    // fetch() rejects with TypeError on network failure / server down
    throw new ApiError(0, 'Unable to connect to the server. Please make sure the backend is running.')
  }

  if (!res.ok) {
    if (res.status === 401) {
      // Session expired / invalid token
      clearToken()
      throw new ApiError(401, 'Your session has expired. Please log in again.')
    }
    const fallbackMap: Record<number, string> = {
      400: 'Invalid request. Please check your input.',
      403: 'You do not have access to this resource.',
      404: 'Not found.',
      413: 'File too large. Maximum size is 20 MB.',
      422: 'Processing failed. Please check the file and try again.',
      500: 'Server error. Please try again later.',
      503: 'Service temporarily unavailable. The AI backend may not be configured.',
    }
    throw new ApiError(res.status, await detailToMessage(res, fallbackMap[res.status] ?? 'Something went wrong. Please try again.'))
  }

  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

// ------------------------------------------------------------------- types

export interface MeUser {
  id: number
  username: string
  email: string
  is_active: boolean
  created_at: string
}

export interface Notebook {
  id: number
  name: string
  description: string
  color: string
  is_pinned: boolean
  source_count: number
  chunk_count: number
  question_count: number
  created_at: string
  updated_at: string
}

export interface DocumentMeta {
  id: number
  notebook_id: number
  document_uuid: string
  filename: string
  file_type: string
  size_bytes: number
  page_count: number
  chunk_count: number
  status: 'ready' | 'processing' | 'failed'
  error_message: string | null
  created_at: string
  updated_at: string
}

export interface NotebookDetail extends Notebook {
  documents: DocumentMeta[]
  notes: Note[]
  saved_answers: SavedAnswer[]
}

export interface Note {
  id: number
  notebook_id: number | null
  title: string
  content: string
  tags: string[]
  is_pinned: boolean
  created_at: string
  updated_at: string
}

export interface ChatSource {
  chunk_id: number
  document_id: string
  chunk_index: number
  similarity_score: number | null
  snippet: string
}

export interface SavedAnswer {
  id: number
  notebook_id: number | null
  question: string
  answer: string
  sources: ChatSource[]
  source_count: number
  created_at: string
}

export interface ChatResponse {
  question: string
  answer: string
  sources: ChatSource[]
  found_context: boolean
  model: string
}

export interface Stats {
  notebooks: number
  documents: number
  questions: number
  ai_insights: number
  total_chunks: number
}

export interface UploadResult {
  document_id: string
  filename: string
  file_type: string
  status: string
  page_count: number
  chunk_count: number
  chunks_stored: number
  error_message: string | null
}

// ------------------------------------------------------------------- auth

export async function register(username: string, email: string, password: string): Promise<MeUser> {
  return request<MeUser>('/api/auth/register', { method: 'POST', body: { username, email, password }, auth: false })
}

export async function login(username: string, password: string): Promise<string> {
  const res = await request<{ access_token: string }>('/api/auth/login', {
    method: 'POST',
    body: { username, password },
    auth: false,
  })
  setToken(res.access_token)
  return res.access_token
}

export async function getMe(): Promise<MeUser> {
  return request<MeUser>('/api/auth/me')
}

// --------------------------------------------------------------- notebooks

export async function listNotebooks(): Promise<Notebook[]> {
  const res = await request<{ notebooks: Notebook[]; total: number }>('/api/notebooks')
  return res.notebooks
}

export async function createNotebook(name: string, description = '', color = 'lavender'): Promise<Notebook> {
  return request<Notebook>('/api/notebooks', { method: 'POST', body: { name, description, color } })
}

export async function getNotebook(id: number): Promise<NotebookDetail> {
  return request<NotebookDetail>(`/api/notebooks/${id}`)
}

export async function updateNotebook(id: number, patch: Partial<Pick<Notebook, 'name' | 'description' | 'color' | 'is_pinned'>>): Promise<Notebook> {
  return request<Notebook>(`/api/notebooks/${id}`, { method: 'PATCH', body: patch })
}

export async function deleteNotebook(id: number): Promise<void> {
  await request<void>(`/api/notebooks/${id}`, { method: 'DELETE' })
}

// --------------------------------------------------------------- documents

export async function listDocuments(): Promise<DocumentMeta[]> {
  const res = await request<{ documents: DocumentMeta[]; total: number }>('/api/documents')
  return res.documents
}

export async function deleteDocument(id: number): Promise<void> {
  await request<void>(`/api/documents/${id}`, { method: 'DELETE' })
}

/** Upload a file to the existing backend endpoint (multipart/form-data). */
export async function uploadDocument(file: File, notebookId: number | string): Promise<UploadResult> {
  const token = getToken()
  const headers: Record<string, string> = {}
  if (token) headers['Authorization'] = `Bearer ${token}`

  const form = new FormData()
  form.append('file', file)
  form.append('notebook_id', String(notebookId))

  let res: Response
  try {
    res = await fetch(`${API_BASE_URL}/api/documents/upload`, {
      method: 'POST',
      headers,
      body: form,
    })
  } catch {
    throw new ApiError(0, 'Unable to connect to the server. Please make sure the backend is running.')
  }

  if (!res.ok) {
    const fallbackMap: Record<number, string> = {
      400: 'Unsupported or empty file. Please use PDF, DOCX, XLSX, TXT, MD or CSV.',
      401: 'Your session has expired. Please log in again.',
      413: 'File too large. Maximum size is 20 MB.',
      422: 'Could not process this document. It may be corrupted or have no extractable text.',
      500: 'Server error while processing the document.',
      503: 'AI backend not configured (missing API key on the server).',
    }
    throw new ApiError(res.status, await detailToMessage(res, fallbackMap[res.status] ?? 'Upload failed. Please try again.'))
  }
  return (await res.json()) as UploadResult
}

/**
 * Upload a file with real, browser-reported upload progress (XMLHttpRequest).
 *
 * ``onProgress`` receives 0-100 as the multipart body is transferred to the
 * backend. After the transfer reaches 100%, the backend still has to validate,
 * extract text, chunk and embed the document — the returned promise (and the
 * UploadResult) is the only confirmation that processing succeeded.
 */
export function uploadDocumentWithProgress(
  file: File,
  notebookId: number | string,
  onProgress?: (percent: number) => void,
): Promise<UploadResult> {
  const token = getToken()
  const form = new FormData()
  form.append('file', file)
  form.append('notebook_id', String(notebookId))

  return new Promise<UploadResult>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE_URL}/api/documents/upload`)
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    xhr.responseType = 'json'
    xhr.timeout = 300_000 // keep long embedding pipelines from hanging forever

    xhr.upload.onprogress = (e) => {
      if (onProgress && e.lengthComputable) {
        onProgress(Math.min(100, Math.round((e.loaded / e.total) * 100)))
      }
    }
    xhr.upload.onload = () => onProgress?.(100)

    const fail = (status: number, message: string) => reject(new ApiError(status, message))

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as UploadResult)
        return
      }
      const fallbackMap: Record<number, string> = {
        400: 'Unsupported or empty file. Please use PDF, DOCX, XLSX, TXT, MD or CSV.',
        401: 'Your session has expired. Please log in again.',
        413: 'File too large. Maximum size is 20 MB.',
        422: 'Could not process this document. It may be corrupted or have no extractable text.',
        500: 'Server error while processing the document.',
        503: 'AI backend not configured (missing API key on the server).',
      }
      const detail =
        (xhr.response as { detail?: unknown } | null)?.detail
      let msg = fallbackMap[xhr.status] ?? 'Upload failed. Please try again.'
      if (typeof detail === 'string') msg = detail
      fail(xhr.status, msg)
    }
    xhr.ontimeout = () => fail(0, 'The upload timed out. Please try again later.')
    xhr.onerror = () => fail(0, 'Unable to connect to the server. Please make sure the backend is running.')
    xhr.onabort = () => fail(0, 'Upload cancelled.')

    xhr.send(form)
  })
}

// ------------------------------------------------------------------- notes

export async function listNotes(): Promise<Note[]> {
  const res = await request<{ notes: Note[]; total: number }>('/api/notes')
  return res.notes
}

export async function createNote(title: string, content: string, tags: string[], notebookId: number | null): Promise<Note> {
  return request<Note>('/api/notes', { method: 'POST', body: { title, content, tags, notebook_id: notebookId } })
}

export async function updateNote(id: number, patch: Partial<Pick<Note, 'title' | 'content' | 'tags' | 'is_pinned'>>): Promise<Note> {
  return request<Note>(`/api/notes/${id}`, { method: 'PATCH', body: patch })
}

export async function deleteNote(id: number): Promise<void> {
  await request<void>(`/api/notes/${id}`, { method: 'DELETE' })
}

// ------------------------------------------------------------ saved answers

export async function listSavedAnswers(): Promise<SavedAnswer[]> {
  const res = await request<{ saved_answers: SavedAnswer[]; total: number }>('/api/saved-answers')
  return res.saved_answers
}

export async function saveAnswer(question: string, answer: string, sources: ChatSource[], notebookId: number | null): Promise<SavedAnswer> {
  return request<SavedAnswer>('/api/saved-answers', {
    method: 'POST',
    body: { question, answer, sources, notebook_id: notebookId },
  })
}

export async function deleteSavedAnswer(id: number): Promise<void> {
  await request<void>(`/api/saved-answers/${id}`, { method: 'DELETE' })
}

// ------------------------------------------------------------------- chat

/**
 * Ask a RAG question scoped to ONE notebook (mandatory).
 *
 * Every chat request carries the selected ``notebookId``; the backend enforces
 * ownership and strictly restricts retrieval + generation to that notebook's
 * documents — chunks from other notebooks are never used, and unanswered
 * questions return an honest "not found" response instead of guessing.
 */
export async function askQuestion(
  question: string,
  notebookId: number,
  topK?: number,
  minSimilarity?: number,
): Promise<ChatResponse> {
  return request<ChatResponse>('/api/chat', {
    method: 'POST',
    body: {
      question,
      notebook_id: notebookId,
      ...(topK !== undefined ? { top_k: topK } : {}),
      ...(minSimilarity !== undefined ? { min_similarity: minSimilarity } : {}),
    },
  })
}

// ------------------------------------------------------------------- stats

export async function getStats(): Promise<Stats> {
  return request<Stats>('/api/stats')
}
