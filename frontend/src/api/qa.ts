export interface QASource {
  content: string
  source: string
  score: number
  type: string
  metadata?: Record<string, unknown>
}

export interface RetrieveStep {
  name: string
  label: string
  hits: number
  cost_ms: number
  detail: string
}

export interface QAMeta {
  intent: string
  confidence: number
  reasoning_steps: string[]
  sources: QASource[]
  retrieve_steps?: RetrieveStep[]
}

export type SSEEvent =
  | ({ type: 'meta' } & QAMeta)
  | { type: 'token'; content: string }
  | { type: 'done'; reasoning_steps: string[] }
  | { type: 'error'; message: string }

export async function* askStream(question: string): AsyncGenerator<SSEEvent> {
  const resp = await fetch('/api/qa/ask_stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!resp.ok || !resp.body) {
    throw new Error(`HTTP ${resp.status}`)
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''

    for (const raw of events) {
      const parsed = parseSSEBlock(raw)
      if (parsed) yield parsed
    }
  }

  if (buffer.trim()) {
    const parsed = parseSSEBlock(buffer)
    if (parsed) yield parsed
  }
}

function parseSSEBlock(block: string): SSEEvent | null {
  const lines = block.split('\n')
  let eventType = ''
  let data = ''
  for (const line of lines) {
    if (line.startsWith('event: ')) eventType = line.slice(7).trim()
    else if (line.startsWith('data: ')) data += line.slice(6)
  }
  if (!eventType || !data) return null
  try {
    return JSON.parse(data) as SSEEvent
  } catch {
    return null
  }
}
