const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");

/**
 * Resolves an API path against the configured VITE_API_BASE_URL.
 * In development or when VITE_API_BASE_URL is not set, returns the original relative path.
 * When VITE_API_BASE_URL is provided, prefixes it cleanly without duplicate slashes.
 */
export function apiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${BASE_URL}${normalizedPath}`;
}
