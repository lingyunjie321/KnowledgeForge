<script setup lang="ts">
import { ref, onMounted } from 'vue'

const healthy = ref<boolean | null>(null)
const serviceName = ref('')

async function checkHealth() {
  try {
    const r = await fetch('/api/health')
    const d = await r.json()
    healthy.value = d.status === 'ok'
    serviceName.value = d.service || ''
  } catch {
    healthy.value = false
  }
}

onMounted(() => {
  checkHealth()
  setInterval(checkHealth, 15000)
})

const navItems = [
  { to: '/qa', label: '智能问答' },
  { to: '/graph', label: '知识图谱' },
  { to: '/upload', label: '文档入库' },
  { to: '/update', label: '增量更新' },
  { to: '/dashboard', label: '系统概览' },
]
</script>

<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="border-b bg-white">
      <div class="mx-auto max-w-5xl px-6 py-4 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-medium">KF</div>
          <div>
            <h1 class="text-base font-medium text-gray-900">KnowledgeForge</h1>
            <p class="text-xs text-gray-500">GraphRAG 混合检索 · 知识图谱推理</p>
          </div>
        </div>
        <div class="flex items-center gap-2 text-sm">
          <span
            class="w-2 h-2 rounded-full"
            :class="healthy === true ? 'bg-emerald-500' : healthy === false ? 'bg-red-500' : 'bg-gray-300'"
          />
          <span class="text-gray-600">
            {{ healthy === true ? (serviceName || '系统正常') : healthy === false ? '服务离线' : '连接中' }}
          </span>
        </div>
      </div>
    </header>

    <nav class="border-b bg-white">
      <div class="mx-auto max-w-5xl px-6 flex gap-1">
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="px-4 py-3 text-sm font-medium border-b-2 -mb-px transition-colors"
          active-class="border-indigo-600 text-indigo-600"
          inactive-class="border-transparent text-gray-600 hover:text-gray-900"
        >
          {{ item.label }}
        </RouterLink>
      </div>
    </nav>

    <main class="flex-1 mx-auto max-w-5xl w-full px-6 py-8">
      <RouterView />
    </main>
  </div>
</template>
