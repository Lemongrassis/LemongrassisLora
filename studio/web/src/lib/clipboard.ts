/** Copy text, falling back to a hidden textarea + execCommand when the async
 * Clipboard API is missing or denied (embedded webviews, non-secure LAN
 * origins, permission policies). Resolves false only when both fail. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch { /* fall through to the legacy path */ }
  const ta = document.createElement('textarea')
  ta.value = text
  ta.setAttribute('readonly', '')
  ta.style.position = 'fixed'
  ta.style.top = '0'
  ta.style.left = '0'
  ta.style.opacity = '0'
  document.body.appendChild(ta)
  const prevFocus = document.activeElement as HTMLElement | null
  try {
    ta.select()
    return document.execCommand('copy')
  } catch {
    return false
  } finally {
    ta.remove()
    prevFocus?.focus?.()
  }
}
