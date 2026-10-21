/**
 * Client-side file validation and display helpers for the upload flows.
 *
 * The backend remains the source of truth (it re-validates extension, MIME
 * type, magic bytes and real size with its own configurable 20 MB limit), but
 * validating here gives users instant feedback before anything is sent.
 */
import {
  FileSpreadsheet,
  FileType2,
  FileText,
  type LucideIcon,
} from 'lucide-react'

/** Mirrors backend MAX_UPLOAD_SIZE_MB; overridable via VITE_MAX_UPLOAD_SIZE_MB. */
const _configuredMb = Number(import.meta.env.VITE_MAX_UPLOAD_SIZE_MB as string | undefined)
export const MAX_UPLOAD_SIZE_MB =
  Number.isFinite(_configuredMb) && _configuredMb > 0 ? _configuredMb : 20
export const MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

export const SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.xls', '.txt', '.md', '.csv'] as const
export type SupportedExtension = (typeof SUPPORTED_EXTENSIONS)[number]

export const FILE_ACCEPT_ATTR = SUPPORTED_EXTENSIONS.join(',')
export const SUPPORTED_EXTENSIONS_LABEL = 'PDF · DOCX · XLSX · TXT · Markdown · CSV'

export function fileExtension(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot).toLowerCase()
}

export function isSupportedFile(name: string): boolean {
  return (SUPPORTED_EXTENSIONS as readonly string[]).includes(fileExtension(name))
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** Friendly type label for display ("PDF", "DOCX", "MD", …). */
export function typeLabel(ext: string): string {
  const e = ext.replace('.', '').toUpperCase()
  if (e === 'MD') return 'MD'
  if (e === 'XLS') return 'XLS'
  return e || 'FILE'
}

/**
 * Validate one file. Returns an error message when the file should be
 * rejected, or null when it may be uploaded.
 */
export function validateFile(file: File): string | null {
  if (!file.name) return 'The file has no name and cannot be uploaded.'
  if (!isSupportedFile(file.name)) {
    return `"${file.name}" is not a supported type. Use ${SUPPORTED_EXTENSIONS_LABEL}.`
  }
  if (file.size === 0) {
    return `"${file.name}" is empty.`
  }
  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    return `"${file.name}" is ${formatBytes(file.size)} — larger than the ${MAX_UPLOAD_SIZE_MB} MB limit.`
  }
  return null
}

/** An icon class for a file type after ingest (used by source lists). */
export function iconForFileType(fileType: string): LucideIcon {
  const t = (fileType || '').toLowerCase()
  if (['xlsx', 'xls', 'csv'].includes(t)) return FileSpreadsheet
  if (t === '') return FileType2
  return FileText
}
