export interface GraphEntity {
  name: string
  type: string
  description: string
  source: string
  version: number
}

export interface GraphRelation {
  source: string
  relation: string
  target: string
  confidence: number
  source_doc: string
}

export interface GraphData {
  entities: GraphEntity[]
  relations: GraphRelation[]
  available: boolean
}

export async function fetchGraphData(limit = 200): Promise<GraphData> {
  const r = await fetch(`/api/graph/data?limit=${limit}`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}
