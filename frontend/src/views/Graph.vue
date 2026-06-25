<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue'
import ForceGraph from 'force-graph'
import { fetchGraphData, type GraphData } from '../api/graph'

const containerRef = ref<HTMLElement | null>(null)
const loading = ref(true)
const errorMsg = ref('')
const stats = ref<{ entities: number; relations: number }>({ entities: 0, relations: 0 })
let graphInstance: any = null

// force-graph 默认导出在 TS 里是 class 但语义上是工厂函数
const forceGraph = ForceGraph as unknown as (el: HTMLElement) => any

const TYPE_COLORS: Record<string, string> = {
  person: '#3b82f6',
  organization: '#8b5cf6',
  location: '#10b981',
  technology: '#f59e0b',
  concept: '#ef4444',
  event: '#06b6d4',
}
const DEFAULT_COLOR = '#6b7280'

async function loadAndRender() {
  loading.value = true
  errorMsg.value = ''
  try {
    const data: GraphData = await fetchGraphData(300)
    if (!data.available) {
      errorMsg.value = 'Neo4j 未连接，图谱检索降级运行中。启动 Neo4j 后即可查看。'
      loading.value = false
      return
    }
    stats.value = { entities: data.entities.length, relations: data.relations.length }

    const nodeSet = new Set(data.entities.map((e) => e.name))
    const nodes = data.entities.map((e) => ({
      id: e.name,
      name: e.name,
      type: e.type,
      description: e.description,
      color: TYPE_COLORS[e.type?.toLowerCase()] ?? DEFAULT_COLOR,
    }))
    const links = data.relations
      .filter((r) => nodeSet.has(r.source) && nodeSet.has(r.target))
      .map((r) => ({
        source: r.source,
        target: r.target,
        label: r.relation,
        confidence: r.confidence,
      }))

    if (!containerRef.value) return
    if (graphInstance) {
      graphInstance._destructor()
      graphInstance = null
    }

    if (nodes.length === 0) {
      errorMsg.value = '图谱为空，先上传文档入库后再查看。'
      loading.value = false
      return
    }

    graphInstance = forceGraph(containerRef.value)
      .graphData({ nodes, links })
      .nodeLabel('name')
      .nodeColor('color')
      .nodeRelSize(6)
      .linkLabel('label')
      .linkColor(() => 'rgba(107, 114, 128, 0.4)')
      .linkWidth(1)
      .linkDirectionalArrowLength(4)
      .linkDirectionalArrowRelPos(1)
      .cooldownTicks(200)
      .onNodeClick((node: any) => {
        if (node.description) {
          alert(`${node.name} (${node.type})\n\n${node.description}`)
        }
      })
    loading.value = false
  } catch (e) {
    errorMsg.value = (e as Error).message
    loading.value = false
  }
}

onMounted(loadAndRender)
onBeforeUnmount(() => {
  if (graphInstance) {
    graphInstance._destructor()
    graphInstance = null
  }
})
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-4">
      <div>
        <h2 class="text-lg font-medium text-gray-900">知识图谱</h2>
        <p class="text-sm text-gray-500 mt-0.5">
          Neo4j 实体关系可视化 · 力导向布局
        </p>
      </div>
      <button
        @click="loadAndRender"
        :disabled="loading"
        class="px-3 py-1.5 text-sm rounded-lg border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50"
      >
        {{ loading ? '加载中' : '刷新' }}
      </button>
    </div>

    <div class="grid grid-cols-2 gap-3 mb-4">
      <div class="bg-white border border-gray-200 rounded-lg px-4 py-3">
        <div class="text-xs text-gray-500">实体数</div>
        <div class="text-2xl font-medium text-gray-900 mt-1">{{ stats.entities }}</div>
      </div>
      <div class="bg-white border border-gray-200 rounded-lg px-4 py-3">
        <div class="text-xs text-gray-500">关系数</div>
        <div class="text-2xl font-medium text-gray-900 mt-1">{{ stats.relations }}</div>
      </div>
    </div>

    <div class="flex flex-wrap gap-3 mb-4 text-xs">
      <span
        v-for="(color, type) in TYPE_COLORS"
        :key="type"
        class="flex items-center gap-1.5 px-2 py-1 rounded bg-white border border-gray-200"
      >
        <span class="w-2.5 h-2.5 rounded-full" :style="{ background: color }" />
        <span class="text-gray-600 capitalize">{{ type }}</span>
      </span>
    </div>

    <div
      ref="containerRef"
      class="bg-white border border-gray-200 rounded-lg"
      style="height: 520px"
    />

    <div v-if="errorMsg" class="mt-3 p-3 rounded-lg bg-amber-50 border border-amber-200 text-sm text-amber-800">
      {{ errorMsg }}
    </div>
  </div>
</template>
