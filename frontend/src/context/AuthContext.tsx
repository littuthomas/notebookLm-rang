import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  clearToken,
  getToken,
  getMe,
  login as apiLogin,
  register,
  resetRefreshState,
} from '../services/api'
import type { MeUser } from '../services/api'

/**
 * Auth context with refresh-aware session handling:
 * - On mount, verifies any stored JWT via GET /api/auth/me
 * - login() stores the token and loads the profile
 * - logout() clears the token, the profile, and the single-flight refresh state
 */

interface AuthState {
  user: MeUser | null
  initializing: boolean
  login: (username: string, password: string) => Promise<MeUser>
  registerAndLogin: (username: string, email: string, password: string) => Promise<MeUser>
  logout: () => void
}

const AuthContext = createContext<AuthState | null>(null)


export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<MeUser | null>(null)
  const [initializing, setInitializing] = useState(true)

  useEffect(() => {
    let cancelled = false
    async function restore() {
      if (!getToken()) {
        setInitializing(false)
        return
      }
      try {
        const me = await getMe()
        if (!cancelled) setUser(me)
      } catch {
        // Invalid/expired token — clear it silently
        clearToken()
        if (!cancelled) setUser(null)
      } finally {
        if (!cancelled) setInitializing(false)
      }
    }
    void restore()
    return () => {
      cancelled = true
    }
  }, [])

  const login = useCallback(async (username: string, password: string) => {
    await apiLogin(username, password)
    const me = await getMe()
    setUser(me)
    return me
  }, [])

  const registerAndLogin = useCallback(async (username: string, email: string, password: string) => {
    await register(username, email, password)
    await apiLogin(username, password)
    const me = await getMe()
    setUser(me)
    return me
  }, [])

  const logout = useCallback(() => {
    clearToken()
    resetRefreshState()
    setUser(null)
  }, [])

  const value = useMemo(
    () => ({ user, initializing, login, registerAndLogin, logout }),
    [user, initializing, login, registerAndLogin, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}


export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}

