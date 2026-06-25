<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { fetchStats, type Stats } from '../api/ingest'

const stats = ref<Stats | null>(null)
const loading = ref(false)
const errorMsg = ref('')

async function load() {
  loading.value = true
  errorMsg.value = ''
  try {
    stats.value = await fetchStats()
  } catch (e) {
    errorMsg.value = (e as Error).message
  } finally {
    loading.value = false
  }
}

onMounted(load)

interface Card {
  label: string
  value: number | string
  hint: string
  color: string
}

function cards(): Card[] {
  if (!stats.value) return []
  return [
    {
      label: '向量索引',
      value: stats.value.vector_store.total_vectors ?? 0,
      hint: 'ChromaDB',
      color: 'bg-blue-500',
    },
    {
      label: '图谱实体',
      value: stats.value.knowledge_graph.total_entities ?? 0,
      hint: 'Neo4j',
      color: 'bg-purple-500',
    },
    {
      label: '图谱关系',
      value: stats.value.knowledge_graph.total_relations ?? 0,
      hint: 'Neo4j',
      color: 'bg-pink-500',
    },
    {
      label: '向量后端',
      value: stats.value.vector_store.backend ?? '--',
      hint: '当前后端',
      color: 'bg-emerald-500',
    },
  ]
}
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <div>
        <h2 class="text-lg font-medium text-gray-900">系统概览</h2>
        <p class="text-sm text-gray-500 mt-0.5">知识库运行状态与数据指标</p>
      </div>
      <button
        @click="load"
        :disabled="loading"
        class="px-3 py-1.5 text-sm rounded-lg border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50"
      >
        {{ loading ? '加载中' : '刷新' }}
      </button>
    </div>

    <div class="grid grid-cols-2 gap-4">
      <div
        v-for="(card, i) in cards()"
        :key="i"
        class="bg-white border border-gray-200 rounded-xl p-5"
      >
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full" :class="card.color" />
          <span class="text-xs text-gray-500">{{ card.hint }}</span>
        </div>
        <div class="text-3xl font-medium text-gray-900 mt-3">{{ card.value }}</div>
        <div class="text-sm text-gray-600 mt-1">{{ card.label }}</div>
      </div>
    </div>

    <div v-if="errorMsg" class="mt-4 p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
      {{ errorMsg }}
    </div>

    <div class="mt-6 bg-white border border-gray-200 rounded-xl p-5">
      <h3 class="text-sm font-medium text-gray-900 mb-3">技术栈</h3>
      <ul class="text-sm text-gray-600 space-y-1.5">
        <li>· 后端：Python 3.12 + FastAPI + LangGraph + LangChain</li>
        <li>· 向量库：ChromaDB（主）/ PGVector（备）</li>
        <li>· 图库：Neo4j</li>
        <li>· LLM：DeepSeek（OpenAI API 兼容）</li>
        <li>· Embedding：三档可切换（local / api / disabled）</li>
        <li>· 前端：Vue 3 + Vite + TypeScript + Tailwind</li>
      </ul>
    </div>
  </div>
</template>
