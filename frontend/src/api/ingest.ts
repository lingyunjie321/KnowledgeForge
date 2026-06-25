export interface IngestResult {
  file_name: string
  chunks_count: number
  entities_count: number
  relations_count: string
  status: string
}

export async function uploadDocument(file: File): Promise<IngestResult> {
  const form = new FormData()
  form.append('file', file)
  const r = await fetch('/api/ingest/upload', { method: 'POST', body: form })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${r.status}`)
  }
  return r.json()
}

export interface Stats {
  vector_store: {
    backend?: string
    total_vectors?: number
  }
  knowledge_graph: {
    total_entities?: number
    total_relations?: number
  }
}

export async function fetchStats(): Promise<Stats> {
  const r = await fetch('/api/admin/stats')
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}
