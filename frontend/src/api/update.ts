export interface UpdateDiff {
  added_count?: number
  removed_count?: number
  change_ratio?: number
  is_major_change?: boolean
  affected_chunks?: string[]
}

export interface UpdateResult {
  file_path: string
  vectors_added: number
  vectors_deleted: number
  entities_added: number
  relations_added: number
  diff: UpdateDiff
  success: boolean
  processing_time_ms: number
}

export async function triggerUpdate(filePath: string, changeType: string): Promise<UpdateResult> {
  const r = await fetch('/api/admin/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_path: filePath, change_type: changeType }),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${r.status}`)
  }
  return r.json()
}
